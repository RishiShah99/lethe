"""Chunk-parallel-carry C6 fused-block backward (the long-L training lever).

The serial C6 backward (``_triton_fused_block_bwd``) walks two serial-L sweeps:
``_fwd_stage_kernel`` register-carries the scan state ``h`` across chunks, and
``_bwd_sweep_kernel`` register-carries the adjoint ``ag = a_bar*g``. Both are the
linear recurrences C1/C2 carry, so both reassociate the same way — this extends
the long-L lever from the SISO backward (C2) to the fused training block, where
the serial backward is still ~3.5x behind official at L=16K.

Pipeline (the two chunk-free kernels — RMSNorm backward, gather-form grad_x —
are reused verbatim from the serial module):

  1. ``_fwd_reduce_kernel``   — per (b, chunk, D-block) local forward
     conv+SiLU+scan from a zero state -> chunk end state ``Sloc`` and decay
     ``Adecay = prod_j a_bar``.
  2. ``_carry_scan`` (torch)  — O(L/K) forward carry -> ``hin`` entering each chunk.
  3. ``_fwd_ys_kernel``       — per chunk, recompute h from ``hin``, stage ``ys``.
  4. ``_norm_bwd_kernel``     — (reused) ``ys`` -> ``dys`` + norm-weight partials.
  5. ``_rev_reduce_kernel``   — per chunk local reverse scan (dys*C) from a zero
     carry -> reverse carry-out ``Sg = a_{c,0}*gloc_{c,0}``.
  6. ``_rev_carry_scan`` (torch) — O(L/K) newest-first carry (the *same*
     ``Adecay``) -> ``Gin`` entering each chunk's newest step.
  7. ``_chunk_bwd_sweep_kernel`` — per chunk, recompute h from ``hin`` then
     reverse-sweep from ``Gin``, emitting all nine gradients' partials. Its inner
     body is byte-for-byte the serial ``_bwd_sweep_kernel`` reverse body; only
     the two carries are supplied externally instead of register-threaded, which
     is what lets all nc chunks run at once.
  8. ``_conv_x_bwd_kernel``   — (reused) the gather-form grad_x.

Memory: the per-chunk forward-recompute scratch (hbuf/zbuf/cvbuf) and the
per-chunk reduction partials are sized for all chunks at once (O(B*L*D*block_n)
for hbuf) — the cost of parallelising what the serial kernel reused across
chunks. At small-batch/long-L (the regime this targets) it fits; the mode
selector only picks it there. ``test_chunk_parallel_fused_bwd_replica`` pins the
algebra fp64-exact / fp32 in-band before hardware. All C2/C6 invariants carry
verbatim: fp32 internally, libdevice exp/log1p, softplus threshold 20, masked
padded lanes exactly zero, int64 offset bases, no tl.dot, no atomics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops._resource_meta import collect_resource_meta
from flash_mamba_rl.kernels.ops._triton_bwd_scan import MAX_BLOCK_N
from flash_mamba_rl.kernels.ops._triton_chunk_parallel_bwd import _rev_carry_scan
from flash_mamba_rl.kernels.ops._triton_chunk_parallel_fwd import _carry_scan
from flash_mamba_rl.kernels.ops._triton_fused_block import (
    _MAX_BLOCK_D_NORM,
    _NORM_TILE,
    MAX_CONV_K,
)
from flash_mamba_rl.kernels.ops._triton_fused_block_bwd import (
    _conv_x_bwd_kernel,
    _norm_bwd_kernel,
)

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)


@triton.jit  # type: ignore[untyped-decorator]
def _fwd_reduce_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    sloc_ptr,
    adecay_ptr,
    l_out,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_k = offs_k < CONV_K
    mask_dn = mask_d[:, None] & mask_n[None, :]
    mask_dk = mask_d[:, None] & mask_k[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    conv_w = tl.load(
        conv_w_ptr + offs_d[:, None] * CONV_K + offs_k[None, :], mask=mask_dk, other=0.0
    ).to(tl.float32)
    conv_b = tl.load(conv_b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    dec = tl.where(mask_dn, 1.0, 0.0)
    seq_in = l_out + CONV_K - 1
    t0 = pid_c * chunk_len
    x_off = pid_b.to(tl.int64) * seq_in * d_model + t0.to(tl.int64) * d_model + offs_d
    od_off = pid_b.to(tl.int64) * l_out * d_model + t0.to(tl.int64) * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * l_out * n_state + t0.to(tl.int64) * n_state + offs_n

    for _j in range(chunk_len):
        xw = tl.load(
            x_ptr + x_off[:, None] + offs_k[None, :] * d_model, mask=mask_dk, other=0.0
        ).to(tl.float32)
        conv = tl.sum(conv_w * xw, axis=1) + conv_b
        z = conv * (1.0 / (1.0 + libdevice.exp(-conv)))
        dlt = tl.load(delta_ptr + od_off, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * z)[:, None] * b_t[None, :]
        h = tl.where(mask_dn, abar * h + bu, 0.0)
        dec = tl.where(mask_dn, dec * abar, 0.0)
        x_off += d_model
        od_off += d_model
        bln_off += n_state

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    tl.store(sloc_ptr + state_off, h, mask=mask_dn)
    tl.store(adecay_ptr + state_off, dec, mask=mask_dn)


@triton.jit  # type: ignore[untyped-decorator]
def _fwd_ys_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    hin_ptr,
    ys_ptr,
    l_out,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_k = offs_k < CONV_K
    mask_dn = mask_d[:, None] & mask_n[None, :]
    mask_dk = mask_d[:, None] & mask_k[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    conv_w = tl.load(
        conv_w_ptr + offs_d[:, None] * CONV_K + offs_k[None, :], mask=mask_dk, other=0.0
    ).to(tl.float32)
    conv_b = tl.load(conv_b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    h = tl.load(hin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    h = tl.where(mask_dn, h, 0.0)

    seq_in = l_out + CONV_K - 1
    t0 = pid_c * chunk_len
    x_off = pid_b.to(tl.int64) * seq_in * d_model + t0.to(tl.int64) * d_model + offs_d
    od_off = pid_b.to(tl.int64) * l_out * d_model + t0.to(tl.int64) * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * l_out * n_state + t0.to(tl.int64) * n_state + offs_n

    for _j in range(chunk_len):
        xw = tl.load(
            x_ptr + x_off[:, None] + offs_k[None, :] * d_model, mask=mask_dk, other=0.0
        ).to(tl.float32)
        conv = tl.sum(conv_w * xw, axis=1) + conv_b
        z = conv * (1.0 / (1.0 + libdevice.exp(-conv)))
        dlt = tl.load(delta_ptr + od_off, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * z)[:, None] * b_t[None, :]
        h = tl.where(mask_dn, abar * h + bu, 0.0)
        ys_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * z
        tl.store(ys_ptr + od_off, ys_t, mask=mask_d)
        x_off += d_model
        od_off += d_model
        bln_off += n_state


@triton.jit  # type: ignore[untyped-decorator]
def _rev_reduce_kernel(  # type: ignore[no-untyped-def]
    delta_ptr,
    a_ptr,
    c_ptr,
    dys_ptr,
    sg_ptr,
    l_out,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
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
    g = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    abar_prev = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    for jj in range(chunk_len):
        j = chunk_len - 1 - jj
        od = pid_b.to(tl.int64) * l_out * d_model + (t0 + j).to(tl.int64) * d_model + offs_d
        bln = pid_b.to(tl.int64) * l_out * n_state + (t0 + j).to(tl.int64) * n_state + offs_n
        dlt = tl.load(delta_ptr + od, mask=mask_d, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)
        dys_t = tl.load(dys_ptr + od, mask=mask_d, other=0.0).to(tl.float32)
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        dyc = tl.where(mask_dn, dys_t[:, None] * c_t[None, :], 0.0)
        g = tl.where(mask_dn, dyc + abar_prev * g, 0.0)
        abar_prev = abar

    sg = tl.where(mask_dn, abar_prev * g, 0.0)
    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]
    tl.store(sg_ptr + state_off, sg, mask=mask_dn)


@triton.jit  # type: ignore[untyped-decorator]
def _chunk_bwd_sweep_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    dys_ptr,
    hin_ptr,
    gin_ptr,
    grad_delta_ptr,
    dconv_ptr,
    ga_part_ptr,  # [B, nc, D, N] fp32
    gb_part_ptr,  # [B, nDb, L_out, N] fp32
    gc_part_ptr,  # [B, nDb, L_out, N] fp32
    gd_part_ptr,  # [B, nc, D] fp32
    gw_part_ptr,  # [B, nc, D, CONV_K] fp32
    gcb_part_ptr,  # [B, nc, D] fp32
    hbuf_ptr,
    zbuf_ptr,
    cvbuf_ptr,
    l_out,
    d_model,
    n_state,
    n_chunks,
    chunk_len,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d = tl.program_id(2)
    n_d_blocks = tl.num_programs(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_k = offs_k < CONV_K
    mask_dn = mask_d[:, None] & mask_n[None, :]
    mask_dk = mask_d[:, None] & mask_k[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    conv_w = tl.load(
        conv_w_ptr + offs_d[:, None] * CONV_K + offs_k[None, :], mask=mask_dk, other=0.0
    ).to(tl.float32)
    conv_b = tl.load(conv_b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    offs_tile = tl.arange(0, BLOCK_D)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offs_d_row = tl.arange(0, BLOCK_D)
    prog = (pid_b.to(tl.int64) * n_chunks + pid_c) * n_d_blocks + pid_d
    hbuf_prog = hbuf_ptr + prog * chunk_len * BLOCK_D * BLOCK_N + offs_tile
    zbuf_prog = zbuf_ptr + prog * chunk_len * BLOCK_D + offs_d_row
    cvbuf_prog = cvbuf_ptr + prog * chunk_len * BLOCK_D + offs_d_row

    state_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * n_state
    state_off = state_off + offs_n[None, :]

    seq_in = l_out + CONV_K - 1
    t0 = pid_c * chunk_len
    x_base = (pid_b.to(tl.int64) * seq_in + t0) * d_model + offs_d
    od_base = (pid_b.to(tl.int64) * l_out + t0) * d_model + offs_d
    bln_base = (pid_b.to(tl.int64) * l_out + t0) * n_state + offs_n

    # ----- Forward recompute from hin, staging h_{t-1}, z, conv per step.
    h_prev = tl.load(hin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    h_prev = tl.where(mask_dn, h_prev, 0.0)
    for j in range(chunk_len):
        tl.store(hbuf_prog + j * BLOCK_D * BLOCK_N, h_prev)
        xw = tl.load(
            x_ptr + x_base[:, None] + (j * d_model + offs_k[None, :] * d_model),
            mask=mask_dk,
            other=0.0,
        ).to(tl.float32)
        conv = tl.sum(conv_w * xw, axis=1) + conv_b
        z = conv * (1.0 / (1.0 + libdevice.exp(-conv)))
        tl.store(cvbuf_prog + j * BLOCK_D, conv)
        tl.store(zbuf_prog + j * BLOCK_D, z)
        dlt = tl.load(delta_ptr + od_base + j * d_model, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln_base + j * n_state, mask=mask_n, other=0.0).to(tl.float32)
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bb = dbar[:, None] * b_t[None, :]
        h_prev = tl.where(mask_dn, abar * h_prev + bb * z[:, None], 0.0)

    # ----- Reverse sweep seeded from the chunk's incoming adjoint carry.
    h_cur = h_prev
    ag_carry = tl.load(gin_ptr + state_off, mask=mask_dn, other=0.0).to(tl.float32)
    ag_carry = tl.where(mask_dn, ag_carry, 0.0)
    ga_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    gw_acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
    gcb_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    pid_bd = (pid_b * n_d_blocks + pid_d).to(tl.int64)
    gbc_base = (pid_bd * l_out + t0) * n_state + offs_n
    for jj in range(chunk_len):
        j = chunk_len - 1 - jj
        od = od_base + j * d_model
        bln = bln_base + j * n_state

        dys_t = tl.load(dys_ptr + od, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + od, mask=mask_d, other=0.0).to(tl.float32)
        b_t = tl.load(b_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln, mask=mask_n, other=0.0).to(tl.float32)
        z = tl.load(zbuf_prog + j * BLOCK_D)
        conv = tl.load(cvbuf_prog + j * BLOCK_D)

        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
        abar = libdevice.exp(dbar[:, None] * a)
        bb = dbar[:, None] * b_t[None, :]
        h_tm1 = tl.load(hbuf_prog + j * BLOCK_D * BLOCK_N)

        g = tl.where(mask_dn, dys_t[:, None] * c_t[None, :] + ag_carry, 0.0)

        gc_t = tl.sum(tl.where(mask_dn, dys_t[:, None] * h_cur, 0.0), axis=0)
        tl.store(gc_part_ptr + gbc_base + j * n_state, gc_t, mask=mask_n)

        gb_t = tl.sum(tl.where(mask_dn, (g * z[:, None]) * dbar[:, None], 0.0), axis=0)
        tl.store(gb_part_ptr + gbc_base + j * n_state, gb_t, mask=mask_n)

        gz_t = tl.sum(tl.where(mask_dn, g * bb, 0.0), axis=1) + d_skip * dys_t
        sig = 1.0 / (1.0 + libdevice.exp(-conv))
        dconv_t = (gz_t * sig) * (1.0 + conv * (1.0 - sig))
        tl.store(dconv_ptr + od, dconv_t, mask=mask_d)

        xw = tl.load(
            x_ptr + x_base[:, None] + (j * d_model + offs_k[None, :] * d_model),
            mask=mask_dk,
            other=0.0,
        ).to(tl.float32)
        gw_acc += tl.where(mask_dk, dconv_t[:, None] * xw, 0.0)
        gcb_acc += dconv_t

        gm = (g * h_tm1) * abar
        ddbar = tl.sum(tl.where(mask_dn, gm * a, 0.0), axis=1) + tl.sum(
            tl.where(mask_dn, (g * z[:, None]) * b_t[None, :], 0.0), axis=1
        )
        zexp = libdevice.exp(dlt)
        dsig = tl.where(dlt > _SOFTPLUS_THRESHOLD, 1.0, zexp / (zexp + 1.0))
        tl.store(
            grad_delta_ptr + od,
            (ddbar * dsig).to(grad_delta_ptr.dtype.element_ty),
            mask=mask_d,
        )

        ga_acc += tl.where(mask_dn, gm * dbar[:, None], 0.0)
        gd_acc += dys_t * z
        ag_carry = tl.where(mask_dn, abar * g, 0.0)
        h_cur = h_tm1

    tl.store(ga_part_ptr + state_off, ga_acc, mask=mask_dn)
    gd_off = (pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d
    tl.store(gd_part_ptr + gd_off, gd_acc, mask=mask_d)
    gw_off = ((pid_b.to(tl.int64) * n_chunks + pid_c) * d_model + offs_d[:, None]) * CONV_K
    gw_off = gw_off + offs_k[None, :]
    tl.store(gw_part_ptr + gw_off, gw_acc, mask=mask_dk)
    tl.store(gcb_part_ptr + gd_off, gcb_acc, mask=mask_d)


def launch_fused_block_backward_chunk_parallel(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    norm_weight: Tensor,
    dy: Tensor,
    eps: float,
    *,
    chunk_len: int,
    config: KernelConfig | None = None,
) -> tuple[Tensor, ...]:
    """Chunk-parallel C6 backward. Inputs must be CUDA tensors of one dtype.

    Returns the nine gradients in ``FusedBlockGrads`` field order, each in its
    input's dtype — same contract as the serial ``launch_fused_block_backward``.
    ``chunk_len`` (K) must divide L_out; it is the chunk-parallel granularity.
    """
    batch, seq_in, d_model = x.shape
    conv_k = conv_weight.shape[-1]
    n_state = a.shape[1]
    l_out = seq_in - (conv_k - 1)

    if conv_k > MAX_CONV_K:
        raise ValueError(f"conv kernel size {conv_k} exceeds window budget {MAX_CONV_K}")
    if l_out < 1:
        raise ValueError(f"sequence length {seq_in} shorter than conv window {conv_k}")
    if l_out % chunk_len != 0:
        raise ValueError(f"output length {l_out} must be divisible by chunk_len {chunk_len}")
    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))
    if config is not None and config.block_d is not None:
        block_d = config.block_d
    block_k = triton.next_power_of_2(conv_k)
    n_chunks = l_out // chunk_len
    n_d_blocks = triton.cdiv(d_model, block_d)

    # Per-chunk forward-recompute scratch sized for all chunks at once — O(B*L*D*
    # block_n), the whole state tensor (the serial kernel reuses a K=16 buffer).
    # Guard before allocating so a too-large shape/config raises instead of an
    # opaque CUDA OOM (and config scoring records it honestly).
    hbuf_elems = batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n
    if x.is_cuda:
        free_bytes, _ = torch.cuda.mem_get_info(x.device)
        if hbuf_elems * 4 > 0.7 * free_bytes:
            raise ValueError(
                f"chunk_parallel C6 backward scratch ~{hbuf_elems * 4 / 1e9:.1f} GB exceeds "
                f"{0.7 * free_bytes / 1e9:.1f} GB free; use serial scan_mode or smaller chunk_len"
            )

    x_c = x.contiguous()
    conv_w_c = conv_weight.contiguous()
    conv_b_c = conv_bias.contiguous()
    delta_c = delta.contiguous()
    a_c = a.contiguous()
    b_c = b.contiguous()
    c_c = c.contiguous()
    dskip_c = d_skip.contiguous()
    norm_w_c = norm_weight.contiguous()
    dy_c = dy.contiguous()

    dev = x.device
    f32 = torch.float32
    ys = torch.empty(batch, l_out, d_model, dtype=f32, device=dev)
    dys = torch.empty(batch, l_out, d_model, dtype=f32, device=dev)
    dconv = torch.empty(batch, l_out, d_model, dtype=f32, device=dev)
    grad_delta = torch.empty_like(delta_c)
    grad_x = torch.empty_like(x_c)

    sloc = torch.empty(batch, n_chunks, d_model, n_state, dtype=f32, device=dev)
    adecay = torch.empty_like(sloc)
    sg = torch.empty_like(sloc)
    hbuf = torch.empty(
        batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n, dtype=f32, device=dev
    )
    zbuf = torch.empty(batch * n_chunks * n_d_blocks * chunk_len * block_d, dtype=f32, device=dev)
    cvbuf = torch.empty_like(zbuf)
    ga_part = torch.empty(batch, n_chunks, d_model, n_state, dtype=f32, device=dev)
    gb_part = torch.empty(batch, n_d_blocks, l_out, n_state, dtype=f32, device=dev)
    gc_part = torch.empty(batch, n_d_blocks, l_out, n_state, dtype=f32, device=dev)
    gd_part = torch.empty(batch, n_chunks, d_model, dtype=f32, device=dev)
    gw_part = torch.empty(batch, n_chunks, d_model, conv_k, dtype=f32, device=dev)
    gcb_part = torch.empty(batch, n_chunks, d_model, dtype=f32, device=dev)

    grid = (batch, n_chunks, n_d_blocks)
    warps = 8 if block_d * block_n >= 512 else 2
    if config is not None and config.num_warps is not None:
        warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages

    _fwd_reduce_kernel[grid](
        x_c, conv_w_c, conv_b_c, delta_c, a_c, b_c, sloc, adecay,
        l_out, d_model, n_state, n_chunks, chunk_len,
        CONV_K=conv_k, BLOCK_K=block_k, BLOCK_D=block_d, BLOCK_N=block_n,
        num_warps=warps, **extra,
    )  # fmt: skip
    hin = _carry_scan(sloc, adecay)

    _fwd_ys_kernel[grid](
        x_c, conv_w_c, conv_b_c, delta_c, a_c, b_c, c_c, dskip_c, hin, ys,
        l_out, d_model, n_state, n_chunks, chunk_len,
        CONV_K=conv_k, BLOCK_K=block_k, BLOCK_D=block_d, BLOCK_N=block_n,
        num_warps=warps, **extra,
    )  # fmt: skip

    block_d_norm = min(_MAX_BLOCK_D_NORM, triton.next_power_of_2(d_model))
    block_t = max(1, _NORM_TILE // block_d_norm)
    n_t_blocks = triton.cdiv(l_out, block_t)
    gnw_part = torch.empty(batch, n_t_blocks, d_model, dtype=f32, device=dev)
    num_warps_norm = 4 if block_t * block_d_norm >= 512 else 2
    _norm_bwd_kernel[(batch, n_t_blocks)](
        ys, dy_c, norm_w_c, dys, gnw_part, l_out, d_model, eps,
        BLOCK_T=block_t, BLOCK_D=block_d_norm, num_warps=num_warps_norm,
    )  # fmt: skip

    _rev_reduce_kernel[grid](
        delta_c, a_c, c_c, dys, sg,
        l_out, d_model, n_state, n_chunks, chunk_len,
        BLOCK_D=block_d, BLOCK_N=block_n, num_warps=warps, **extra,
    )  # fmt: skip
    gin = _rev_carry_scan(sg, adecay)

    _chunk_bwd_sweep_kernel[grid](
        x_c, conv_w_c, conv_b_c, delta_c, a_c, b_c, c_c, dskip_c, dys, hin, gin,
        grad_delta, dconv, ga_part, gb_part, gc_part, gd_part, gw_part, gcb_part,
        hbuf, zbuf, cvbuf,
        l_out, d_model, n_state, n_chunks, chunk_len,
        CONV_K=conv_k, BLOCK_K=block_k, BLOCK_D=block_d, BLOCK_N=block_n,
        num_warps=warps, **extra,
    )  # fmt: skip

    block_t_x = max(1, _NORM_TILE // block_d_norm)
    grid_x = (batch, triton.cdiv(seq_in, block_t_x), triton.cdiv(d_model, block_d_norm))
    _conv_x_bwd_kernel[grid_x](
        dconv, conv_w_c, grad_x, seq_in, l_out, d_model,
        CONV_K=conv_k, BLOCK_T=block_t_x, BLOCK_D=block_d_norm, num_warps=num_warps_norm,
    )  # fmt: skip

    grad_a = ga_part.sum(dim=(0, 1)).to(a.dtype)
    grad_b = gb_part.sum(dim=1).to(b.dtype)
    grad_c = gc_part.sum(dim=1).to(c.dtype)
    grad_d = gd_part.sum(dim=(0, 1)).to(d_skip.dtype)
    grad_w = gw_part.sum(dim=(0, 1)).unsqueeze(1).to(conv_weight.dtype)
    grad_cb = gcb_part.sum(dim=(0, 1)).to(conv_bias.dtype)
    grad_nw = gnw_part.sum(dim=(0, 1)).to(norm_weight.dtype)
    return (
        grad_x,
        grad_w,
        grad_cb,
        grad_delta,
        grad_a,
        grad_b,
        grad_c,
        grad_d,
        grad_nw,
    )


def resource_meta() -> dict[str, int] | None:
    """Max-envelope resource metadata over the chunk-parallel C6 kernels."""
    merged: dict[str, int] = {}
    kernels = (
        _fwd_reduce_kernel,
        _fwd_ys_kernel,
        _norm_bwd_kernel,
        _rev_reduce_kernel,
        _chunk_bwd_sweep_kernel,
        _conv_x_bwd_kernel,
    )
    for jit_fn in kernels:
        meta = collect_resource_meta(jit_fn)
        if meta is None:
            continue
        for key, val in meta.items():
            merged[key] = max(merged.get(key, 0), val)
    return merged or None
