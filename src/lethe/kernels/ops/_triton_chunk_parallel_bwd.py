"""Chunk-parallel-carry SISO backward scan (the long-L backward speedup lever).

The serial-L backward (``_triton_bwd_scan``) walks one program over all L: a
forward checkpoint sweep then a single reverse sweep, both O(L). That serial
critical path is what leaves the *backward* ~3.5x behind the official kernel at
L=16K even though the forward already reassociates. This module reassociates the
backward the same way the forward was: both states it carries are linear
recurrences.

  - forward state ``h_t = a_t*h_{t-1} + bu_t`` — reused verbatim from the
    forward module (``_chunk_reduce_kernel`` + ``_carry_scan``) to get the state
    ``hin`` entering each chunk, so every chunk can recompute its in-chunk
    forward states independently.
  - reverse adjoint ``g_t = dy_t*C_t + a_{t+1}*g_{t+1}`` — the mirror-image
    linear scan (newest-first). Its chunked carry is the reverse twin of the
    forward: a local per-chunk reverse reduce (``_rev_reduce_kernel``, parallel
    across chunks) emits each chunk's carry-out ``Sg``; an O(L/K) torch carry
    (``_rev_carry_scan``) threads ``Gin`` over the nc chunk boundaries; the
    per-chunk readout (``_chunk_bwd_readout_kernel``) seeds its reverse sweep
    from ``Gin[c]``. The chunk decay ``A_chunk = prod_j a`` is the *same* product
    the forward carry uses, so it is produced once.

The readout kernel's inner body — the six gradient expressions, grouping by
grouping (``gm=(g*h_{t-1})*a`` once; the two delta paths as separate
N-reductions) — is byte-for-byte the serial kernel's reverse-sweep body; only
the two carries (forward ``hin``, reverse ``Gin``) are supplied externally
instead of threaded in registers, which is what lets all nc chunks run at once.
The reassociation stays inside the eps*sqrt(chain)*scale band the gates allow;
``test_chunk_parallel_bwd_scan_replica`` pins the algebra fp64-exact / fp32
in-band before any hardware.

Memory: the per-chunk forward recompute scratch (``hbuf``) is sized for all
chunks at once (O(B*L*D*block_n)) — the cost of parallelising what the serial
kernel reused across chunks. At small-batch/long-L (the regime this targets)
this is a few GB and fits; the mode selector / autotuner only picks it where it
does. All C2 invariants carry verbatim: fp32 compute (upcast at load, round once
per gradient output), libdevice exp/log1p, softplus threshold 20, masked padded
lanes, int64 offset bases, no tl.dot, no atomics (deterministic torch reductions
of per-program partials).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.ops._resource_meta import collect_resource_meta
from lethe.kernels.ops._triton_chunk_parallel_fwd import (
    MAX_BLOCK_N,
    _carry_scan,
    _chunk_reduce_kernel,
)
from lethe.kernels.ops.forward_chunked_scan import _chunk_parallel_bwd_scratch_bytes

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)
# Register-tile budget (elements) for the [block_d, block_n] fp32 tiles; block_d
# shrinks at large block_n so a Mamba-3 d_state=128 tile doesn't spill. Mirrors
# the serial backward; validated by scratch/bwd_n128_sweep.
_BWD_TILE_BUDGET = 2048


@triton.jit  # type: ignore[untyped-decorator]
def _rev_reduce_kernel(  # type: ignore[no-untyped-def]
    delta_ptr,
    a_ptr,
    c_ptr,
    dy_ptr,
    sg_ptr,
    seq_len,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 1 (reverse): per (batch, chunk, D-block) local reverse scan from a
    zero carry; emit the chunk's reverse carry-out ``Sg = a_{c,0} * gloc_{c,0}``.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )

    t0 = pid_c * chunk_len
    # Walk newest-first. abar_prev carries a_{j+1} (the multiplier for the older
    # step); starting it at 0 makes the j=K-1 step take a zero incoming carry
    # (local), and g starts at exact zero so 0*0 mints no NaN.
    g = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    abar_prev = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    for jj in range(chunk_len):
        j = chunk_len - 1 - jj
        uld = pid_b.to(tl.int64) * seq_len * d_model + (t0 + j).to(tl.int64) * d_model + offs_d
        bln = pid_b.to(tl.int64) * seq_len * n_state + (t0 + j).to(tl.int64) * n_state + offs_n
        dlt = tl.load(delta_ptr + uld, mask=mask_d, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)
        dy_t = tl.load(dy_ptr + uld, mask=mask_d, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        dyc = tl.where(mask_dn, dy_t[:, None] * c_t[None, :], 0.0)
        g = tl.where(mask_dn, dyc + abar_prev * g, 0.0)
        abar_prev = abar

    # After the loop abar_prev = a_{c,0}; g = gloc_{c,0}.
    sg = tl.where(mask_dn, abar_prev * g, 0.0)
    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    tl.store(sg_ptr + state_off, sg, mask=mask_dn)


def _rev_carry_scan(sg: Tensor, adecay: Tensor) -> Tensor:
    """Phase 2 (reverse): serial carry over the nc boundaries, newest-first.

    ``Gin[c]`` is the additive adjoint carry entering chunk c at its newest step:
    Gin[nc-1]=0, Gin[c] = adecay[c+1]*Gin[c+1] + sg[c+1]. O(nc), not O(L). The
    decay ``adecay`` is the same per-chunk product the forward carry consumes.
    """
    batch, n_chunks, d_model, n_state = sg.shape
    gin = torch.empty_like(sg)
    carry = torch.zeros(batch, d_model, n_state, dtype=sg.dtype, device=sg.device)
    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        gin[:, c] = carry
        carry = adecay[:, c] * carry + sg[:, c]
    return gin


@triton.jit  # type: ignore[untyped-decorator]
def _chunk_bwd_readout_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    dy_ptr,
    hin_ptr,  # [B, nc, D, N] forward state entering each chunk
    gin_ptr,  # [B, nc, D, N] reverse carry entering each chunk
    grad_u_ptr,
    grad_delta_ptr,
    ga_part_ptr,  # [B, nc, D, N] fp32, reduced over (batch, chunk) by the launcher
    gb_part_ptr,  # [B, nDb, L, N] fp32, reduced over nDb
    gc_part_ptr,  # [B, nDb, L, N] fp32, reduced over nDb
    gd_part_ptr,  # [B, nc, D] fp32, reduced over (batch, chunk)
    hbuf_ptr,  # [B * nc * nDb * chunk_len * BLOCK_D * BLOCK_N] fp32 scratch
    seq_len,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 3 + readout: recompute the chunk's forward states from ``hin``, then
    reverse-sweep from ``Gin`` computing all six gradients — the serial reverse
    body, run per-chunk in parallel.
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    n_d_blocks = tl.num_programs(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    offs_tile = tl.arange(0, BLOCK_D)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    prog = (pid_b.to(tl.int64) * n_chunks + pid_c) * n_d_blocks + pid_d
    hbuf_prog = hbuf_ptr + prog * chunk_len * BLOCK_D * BLOCK_N + offs_tile

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]

    t0 = pid_c * chunk_len
    uld_base = (pid_b.to(tl.int64) * seq_len + t0) * d_model + offs_d
    bln_base = (pid_b.to(tl.int64) * seq_len + t0) * n_state + offs_n

    # ----- Forward recompute from hin, storing the PRE-update state per step.
    h_prev = tl.load(hin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    h_prev = tl.where(mask_dn, h_prev, 0.0)
    for j in range(chunk_len):
        tl.store(hbuf_prog + j * BLOCK_D * BLOCK_N, h_prev)
        u_t = tl.load(u_ptr + uld_base + j * d_model, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_base + j * d_model, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_base + j * n_state, mask=mask_n, other=0.0).to(tl.float32)
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bb = dbar[:, None] * b_t[None, :]
        h_prev = tl.where(mask_dn, abar * h_prev + bb * u_t[:, None], 0.0)

    # ----- Reverse sweep, seeded from the chunk's incoming adjoint carry.
    h_cur = h_prev
    ag_carry = tl.load(gin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    ag_carry = tl.where(mask_dn, ag_carry, 0.0)
    ga_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    pid_bd = (pid_b * n_d_blocks + pid_d).to(tl.int64)
    gbc_base = (pid_bd * seq_len + t0) * n_state + offs_n
    for jj in range(chunk_len):
        j = chunk_len - 1 - jj
        uld = uld_base + j * d_model
        bln = bln_base + j * n_state

        dy_t = tl.load(dy_ptr + uld, mask=mask_d, other=0.0).to(tl.float32)
        u_t = tl.load(u_ptr + uld, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bb = dbar[:, None] * b_t[None, :]
        h_tm1 = tl.load(hbuf_prog + j * BLOCK_D * BLOCK_N)

        g = tl.where(mask_dn, dy_t[:, None] * c_t[None, :] + ag_carry, 0.0)

        gc_t = tl.sum(tl.where(mask_dn, dy_t[:, None] * h_cur, 0.0), axis=0)
        tl.store(gc_part_ptr + gbc_base + j * n_state, gc_t, mask=mask_n)

        gb_t = tl.sum(tl.where(mask_dn, (g * u_t[:, None]) * dbar[:, None], 0.0), axis=0)
        tl.store(gb_part_ptr + gbc_base + j * n_state, gb_t, mask=mask_n)

        gu_t = tl.sum(tl.where(mask_dn, g * bb, 0.0), axis=1) + d_skip * dy_t
        tl.store(grad_u_ptr + uld, gu_t.to(grad_u_ptr.dtype.element_ty), mask=mask_d)

        gm = (g * h_tm1) * abar
        ddbar = tl.sum(tl.where(mask_dn, gm * a, 0.0), axis=1) + tl.sum(
            tl.where(mask_dn, (g * u_t[:, None]) * b_t[None, :], 0.0), axis=1
        )
        z = libdevice.exp(dlt)
        dsig = tl.where(dlt > _SOFTPLUS_THRESHOLD, 1.0, z / (z + 1.0))
        tl.store(
            grad_delta_ptr + uld,
            (ddbar * dsig).to(grad_delta_ptr.dtype.element_ty),
            mask=mask_d,
        )

        ga_acc += tl.where(mask_dn, gm * dbar[:, None], 0.0)
        gd_acc += dy_t * u_t
        ag_carry = tl.where(mask_dn, abar * g, 0.0)
        h_cur = h_tm1

    tl.store(ga_part_ptr + state_off, ga_acc, mask=mask_dn)
    gd_off = (pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d
    tl.store(gd_part_ptr + gd_off, gd_acc, mask=mask_d)


def launch_backward_chunk_parallel_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    dy: Tensor,
    *,
    chunk_len: int,
    config: KernelConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Chunk-parallel backward scan. Inputs must be CUDA tensors of one dtype.

    Returns ``(grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D)`` in the
    input dtypes — same contract as the serial ``launch_backward_scan``.
    ``chunk_len`` (K) must divide L; it is the chunk-parallel granularity.
    ``config`` overrides block_d/num_warps/num_stages as in the serial launcher.
    BLOCK_N is pinned to the full state dim, a correctness constraint.
    """
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]
    if seq_len % chunk_len != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_len {chunk_len}")
    n_chunks = seq_len // chunk_len

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    # Same register-tile budget as the serial backward: shrink block_d at large
    # block_n so a Mamba-3 d_state=128 tile doesn't spill (2x at N=128;
    # bwd_n128_sweep). Unchanged at N=16; correctness-invariant (tiling only).
    block_d = min(64, max(16, _BWD_TILE_BUDGET // block_n))
    if config is not None and config.block_d is not None:
        block_d = config.block_d
    n_d_blocks = triton.cdiv(d_model, block_d)

    # The per-chunk forward-recompute scratch is sized for ALL chunks at once
    # (the serial kernel reuses one K=16-capped buffer; here chunk_len can be 512
    # and nc*chunk_len=L, so hbuf is O(B*L*D*block_n) — the whole state tensor).
    # Guard before allocating so a too-large shape/config raises instead of an
    # opaque CUDA OOM mid-backward (and so config scoring records it honestly).
    scratch_bytes = _chunk_parallel_bwd_scratch_bytes(batch, seq_len, d_model, n_state, chunk_len)
    if u.is_cuda:
        free_bytes, _ = torch.cuda.mem_get_info(u.device)
        if scratch_bytes > 0.7 * free_bytes:
            raise ValueError(
                f"chunk_parallel backward scratch ~{scratch_bytes / 1e9:.1f} GB exceeds "
                f"{0.7 * free_bytes / 1e9:.1f} GB free; use serial scan_mode or smaller chunk_len"
            )

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    d_c = d_skip.contiguous()
    dy_c = dy.contiguous()

    dev = u.device
    grad_u = torch.empty_like(u_c)
    grad_delta = torch.empty_like(delta_c)
    ga_part = torch.empty(batch, n_chunks, d_model, n_state, dtype=torch.float32, device=dev)
    gb_part = torch.empty(batch, n_d_blocks, seq_len, n_state, dtype=torch.float32, device=dev)
    gc_part = torch.empty(batch, n_d_blocks, seq_len, n_state, dtype=torch.float32, device=dev)
    gd_part = torch.empty(batch, n_chunks, d_model, dtype=torch.float32, device=dev)

    sloc = torch.empty(batch, n_chunks, d_model, n_state, dtype=torch.float32, device=dev)
    adecay = torch.empty_like(sloc)
    sg = torch.empty_like(sloc)
    hbuf = torch.empty(
        batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n,
        dtype=torch.float32,
        device=dev,
    )

    grid = (batch, n_chunks, n_d_blocks)
    # block_d<=16 only at the large-state regime (block_n=128); num_warps=2 wins
    # there (bwd_n128_sweep), all other shapes unchanged.
    num_warps = 2 if block_d <= 16 else (4 if block_d * block_n >= 512 else 2)
    if config is not None and config.num_warps is not None:
        num_warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages

    # Forward chunk reduce (reused) -> per-chunk end state + decay -> hin.
    _chunk_reduce_kernel[grid](
        u_c, delta_c, a_c, b_c, sloc, adecay,
        seq_len, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=num_warps, **extra,
    )  # fmt: skip
    hin = _carry_scan(sloc, adecay)

    # Reverse chunk reduce -> per-chunk adjoint carry-out -> Gin.
    _rev_reduce_kernel[grid](
        delta_c, a_c, c_c, dy_c, sg,
        seq_len, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=num_warps, **extra,
    )  # fmt: skip
    gin = _rev_carry_scan(sg, adecay)

    _chunk_bwd_readout_kernel[grid](
        u_c, delta_c, a_c, b_c, c_c, d_c, dy_c, hin, gin,
        grad_u, grad_delta, ga_part, gb_part, gc_part, gd_part, hbuf,
        seq_len, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=num_warps, **extra,
    )  # fmt: skip

    grad_a = ga_part.sum(dim=(0, 1)).to(a.dtype)
    grad_b = gb_part.sum(dim=1).to(b.dtype)
    grad_c = gc_part.sum(dim=1).to(c.dtype)
    grad_d = gd_part.sum(dim=(0, 1)).to(d_skip.dtype)
    return grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d


def resource_meta() -> dict[str, int] | None:
    """Resource envelope over all compiled chunk-parallel backward kernels."""
    rm: dict[str, int] = {}
    for kern in (_chunk_reduce_kernel, _rev_reduce_kernel, _chunk_bwd_readout_kernel):
        meta = collect_resource_meta(kern)
        if meta is None:
            continue
        for key, val in meta.items():
            rm[key] = max(rm.get(key, 0), val)
    return rm or None
