"""Triton kernel for the SISO selective-scan backward (C2).

This is the op the official ``mamba_ssm`` cannot compile on Blackwell
(#904): its chunked backward feeds ``tl.dot``, and Triton's eager
TMEM-promotion pass (pre-#9093) promotes the LHS operand past sm_100's
512-element TMEM budget at every ``num_warps >= 4`` config. This kernel is
structured so that pass never engages: the recurrence is carried in
registers as elementwise FMAs and ``tl.sum`` reductions, no ``tl.dot``,
no MMA, no TMEM operands, so it compiles at num_warps 2/4/8 alike on
sm_100. The bench driver records both kernels' compile behaviour side
by side.

Import this module only when ``triton`` is installed and a CUDA device is
the target, the public dispatcher in ``backward_selective_scan.py`` guards
both. Layout assumptions (enforced by the launcher via ``.contiguous()``):

    u, delta, dy, grad_u, grad_delta : [B, L, D]   row-major
    A                                : [D, N]
    B, C                             : [B, L, N]
    D_skip                           : [D]

Parallelisation: one program per (batch, D-block), as in the C1 forward.
The cross-program reductions the backward introduces (grad_B and grad_C
sum over D; grad_A and grad_D sum over (batch, L)) are written as
per-program partial buffers and reduced by a deterministic ``torch.sum``
in the launcher. No atomics anywhere: ORD-02 requires byte-identical
repeated calls, and float-atomic accumulation order is run-dependent.

State recompute: the reverse sweep needs h_{t-1} at every step. Storing
all states is O(B*L*D*N), terabytes at training shapes, and inverting
the recurrence (h_{t-1} = (h_t - dbar*u*B) / abar) divides by abar, which
underflows to exact zero for saturated delta. Instead: a forward sweep
stores the state *entering* each K-step chunk (checkpoint buffer), then
chunks are processed newest-first, the K in-chunk pre-update states are
recomputed from the checkpoint into a scratch buffer, then the reverse
sweep walks the chunk. K is capped small so the scratch working set
(B*D*K*N fp32) stays L2-resident (see ``_chunk_k``); the checkpoint
buffer is touched once per element.

Gradient expressions mirror torch.autograd's dataflow through the eager
forward grouping-by-grouping, not merely algebraically: EXC-01 compares
NaN/Inf masks against the autograd oracle, and Inf*0 products mint NaNs in
whichever factoring you choose. Example: at t=0 (h_{-1}=0) autograd's
grad_delta_bar via the exp path is ((g*h_{t-1})*abar)*A = NaN*... for
g=Inf, while the algebraically-equal g*(A*abar*h_{t-1} + u*B) gives ±Inf.
The kernel therefore computes ``gm = (g*h_{t-1})*abar`` once and keeps the
two delta paths as separate N-reductions, exactly like autograd's two
AccumulateGrad contributions.

All arithmetic is fp32 regardless of input dtype (upcast at load, one
round at store); partial buffers and the launcher's reductions stay fp32
and round once at the very end. Softplus and its derivative match
``torch.nn.functional.softplus`` exactly (linear above threshold 20;
derivative z/(z+1) with z=exp(x), 1 above threshold). libdevice exp/log1p
preserve denormals, ex2.approx flushes subnormal outputs and splits the
EXC masks (see the C1 module docstring).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.ops._resource_meta import collect_resource_meta

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

# torch.nn.functional.softplus switches to the identity above this threshold.
_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)

# One CTA holds the whole state dim; N above this needs a multi-block design.
MAX_BLOCK_N = 128
# Register-tile budget (elements) for the per-program [block_d, block_n] fp32
# tiles the serial-L recurrence holds; block_d shrinks at large block_n so a
# Mamba-3 d_state=128 tile doesn't spill.
_BWD_TILE_BUDGET = 2048

# Recompute-chunk cap. Working set of the in-chunk state scratch is
# B*D*K*N fp32 across all programs; K=16 keeps it ~34 MB at the training
# shape (B=8, D=4096, N=16), resident in B200's 126 MB L2, so the
# store-then-reload round trip never pays DRAM. Larger K trades scratch
# traffic against checkpoint size; this is a v2 autotuning lever.
MAX_CHUNK_K = 16


def _chunk_k(seq_len: int) -> int:
    """Largest power-of-2 divisor of seq_len, capped at MAX_CHUNK_K."""
    return min(seq_len & (-seq_len), MAX_CHUNK_K)


@triton.jit  # type: ignore[untyped-decorator]
def _bwd_scan_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    dy_ptr,
    grad_u_ptr,
    grad_delta_ptr,
    ga_part_ptr,  # [B, D, N] fp32, reduced over batch by the launcher
    gb_part_ptr,  # [B, nDb, L, N] fp32, reduced over nDb by the launcher
    gc_part_ptr,  # [B, nDb, L, N] fp32, reduced over nDb by the launcher
    gd_part_ptr,  # [B, D] fp32, reduced over batch by the launcher
    ckpt_ptr,  # [B * nDb * n_chunks * BLOCK_D * BLOCK_N] fp32 scratch
    hbuf_ptr,  # [B * nDb * CHUNK_K * BLOCK_D * BLOCK_N] fp32 scratch
    seq_len,
    d_model,
    n_state,
    n_chunks,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    n_d_blocks = tl.num_programs(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    # Program-owned scratch tiles need no masks: padded lanes hold zeros by
    # construction (every value stored there went through a mask_dn where).
    offs_tile = tl.arange(0, BLOCK_D)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    pid_bd = (pid_b * n_d_blocks + pid_d).to(tl.int64)
    ckpt_prog = ckpt_ptr + pid_bd * n_chunks * BLOCK_D * BLOCK_N + offs_tile
    hbuf_prog = hbuf_ptr + pid_bd * CHUNK_K * BLOCK_D * BLOCK_N + offs_tile

    # Time-invariant operands, loaded once.
    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    # ----- Phase 1: forward sweep, checkpointing the state entering each chunk.
    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    uld_off = pid_b.to(tl.int64) * seq_len * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * seq_len * n_state + offs_n
    for c in range(n_chunks):
        # c is int32: promote before the tile-stride product (it crosses
        # 2^31 once the per-program ckpt region exceeds 8 GiB).
        tl.store(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N, h)  # type: ignore[attr-defined]
        for _j in range(CHUNK_K):
            u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
            dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
            b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

            dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
            abar = libdevice.exp(dbar[:, None] * a)
            bb = dbar[:, None] * b_t[None, :]
            # Padding lanes must stay exactly zero through the whole update
            # (C1 lesson): a NaN minted in a padded lane would poison the
            # real lanes' N-reductions downstream.
            h = tl.where(mask_dn, abar * h + bb * u_t[:, None], 0.0)

            uld_off += d_model
            bln_off += n_state

    # ----- Phase 2: chunks newest-first; recompute in-chunk states, then
    # reverse-sweep the chunk. ag_carry = abar_{t+1} * g_{t+1} crosses chunk
    # boundaries in registers because t descends globally.
    ag_carry = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    ga_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        t0 = c * CHUNK_K

        # Forward recompute from the checkpoint, storing the PRE-update
        # state h_{t-1} per step: the reverse sweep then reads h_{t-1}
        # uniformly from the scratch (the j=0 entry is the checkpoint
        # itself) and walks h_t backward in a register.
        h_prev = tl.load(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N)
        # Fold t0 into the int64 batch term before multiplying by the row
        # stride: a bare int32 t0 * d_model overflows past L*D ~ 2^31 (the
        # C1 invariant; j * d_model stays tiny, j < CHUNK_K).
        uld_base = (pid_b.to(tl.int64) * seq_len + t0) * d_model + offs_d
        bln_base = (pid_b.to(tl.int64) * seq_len + t0) * n_state + offs_n
        for j in range(CHUNK_K):
            tl.store(hbuf_prog + j * BLOCK_D * BLOCK_N, h_prev)
            u_t = tl.load(u_ptr + uld_base + j * d_model, mask=mask_d, other=0.0).to(tl.float32)
            dlt = tl.load(delta_ptr + uld_base + j * d_model, mask=mask_d, other=0.0).to(tl.float32)
            b_t = tl.load(b_ptr + bln_base + j * n_state, mask=mask_n, other=0.0).to(tl.float32)
            dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
            abar = libdevice.exp(dbar[:, None] * a)
            bb = dbar[:, None] * b_t[None, :]
            h_prev = tl.where(mask_dn, abar * h_prev + bb * u_t[:, None], 0.0)

        # Reverse sweep over the chunk. h_cur enters as the chunk-end state.
        h_cur = h_prev
        gbc_base = (pid_bd * seq_len + t0) * n_state + offs_n
        for jj in range(CHUNK_K):
            j = CHUNK_K - 1 - jj
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

            # g_t = dL/dh_t. dy in padded lanes loads as 0, but Inf*0 across
            # the (d, n) broadcast would mint NaNs in padded n-lanes, mask.
            g = tl.where(mask_dn, dy_t[:, None] * c_t[None, :] + ag_carry, 0.0)

            # grad_C partial: sum over this program's D-slice of dy_t * h_t.
            gc_t = tl.sum(tl.where(mask_dn, dy_t[:, None] * h_cur, 0.0), axis=0)
            tl.store(gc_part_ptr + gbc_base + j * n_state, gc_t, mask=mask_n)

            # grad_B partial: autograd's grouping is (g*u)*dbar.
            gb_t = tl.sum(tl.where(mask_dn, (g * u_t[:, None]) * dbar[:, None], 0.0), axis=0)
            tl.store(gb_part_ptr + gbc_base + j * n_state, gb_t, mask=mask_n)

            # grad_u: sum_n g*b_bar, plus the skip path D*dy.
            gu_t = tl.sum(tl.where(mask_dn, g * bb, 0.0), axis=1) + d_skip * dy_t
            tl.store(grad_u_ptr + uld, gu_t.to(grad_u_ptr.dtype.element_ty), mask=mask_d)

            # grad_delta_bar: two separate N-reductions, mirroring autograd's
            # two AccumulateGrad paths (via exp and via b_bar). The shared
            # gm = (g*h_{t-1})*abar is also the grad_A integrand. gm needs no
            # mask of its own: g and h_tm1 are both pre-masked to exact zero
            # in padded lanes, so gm is 0 there regardless of abar, that
            # zero-ness is load-bearing for the re-masked uses below.
            gm = (g * h_tm1) * abar
            ddbar = tl.sum(tl.where(mask_dn, gm * a, 0.0), axis=1) + tl.sum(
                tl.where(mask_dn, (g * u_t[:, None]) * b_t[None, :], 0.0), axis=1
            )
            # Softplus derivative, matching aten's softplus_backward branch.
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
            # Dead at j=0 (the next chunk re-seeds h_cur from its recompute).
            h_cur = h_tm1

    # Per-(batch, D-block) partials for the launcher's deterministic sums.
    ga_off = pid_b.to(tl.int64) * d_model * n_state + offs_d[:, None] * n_state + offs_n[None, :]
    tl.store(ga_part_ptr + ga_off, ga_acc, mask=mask_dn)
    gd_off = pid_b.to(tl.int64) * d_model + offs_d
    tl.store(gd_part_ptr + gd_off, gd_acc, mask=mask_d)


def launch_backward_scan(
    u: Tensor,
    delta: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d_skip: Tensor,
    dy: Tensor,
    *,
    num_warps: int | None = None,
    config: KernelConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Launch the Triton backward scan. Inputs must be CUDA tensors of one dtype.

    Returns ``(grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D)`` in the
    corresponding input dtypes. ``config`` overrides the autotuner's searched
    knobs (block_d, chunk_k, num_warps, num_stages); ``num_warps`` is the
    legacy bench-sweep hook for the #904 compile-behaviour comparison. A None
    config or None field keeps the shipped heuristic, so the default path is
    byte-for-byte the pre-autotune launch.
    """
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    # Cap the per-program [block_d, block_n] fp32 register tile at a budget so a
    # large state (Mamba-3 d_state=128) shrinks block_d instead of spilling: the
    # serial-L recurrence holds several such tiles in registers, and at a fixed
    # block_d=64 a block_n=128 tile spills hard (measured 2x slower; bd=64
    # nw=2 at N=128 was 6x worse). Unchanged at N=16
    # (2048//16=128 -> min(64,128)=64), correctness-invariant (tiling only).
    block_d = min(64, max(16, _BWD_TILE_BUDGET // block_n))
    if config is not None and config.block_d is not None:
        block_d = config.block_d
    chunk_k = _chunk_k(seq_len)
    if config is not None and config.chunk_k is not None:
        chunk_k = config.chunk_k
        # A direct override that does not divide seq_len drops the tail chunk
        # (uninitialised grads); the default _chunk_k always divides.
        if seq_len % chunk_k != 0:
            raise ValueError(f"chunk_k override {chunk_k} must divide seq_len {seq_len}")
    n_chunks = seq_len // chunk_k
    n_d_blocks = triton.cdiv(d_model, block_d)

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
    # Partials and scratch stay fp32 regardless of input dtype: internal
    # compute is fp32, and the single round to the input dtype happens after
    # the launcher's reduction (one round at the output, the PRC contract).
    ga_part = torch.empty(batch, d_model, n_state, dtype=torch.float32, device=dev)
    gb_part = torch.empty(batch, n_d_blocks, seq_len, n_state, dtype=torch.float32, device=dev)
    gc_part = torch.empty(batch, n_d_blocks, seq_len, n_state, dtype=torch.float32, device=dev)
    gd_part = torch.empty(batch, d_model, dtype=torch.float32, device=dev)
    ckpt = torch.empty(
        batch * n_d_blocks * n_chunks * block_d * block_n, dtype=torch.float32, device=dev
    )
    hbuf = torch.empty(
        batch * n_d_blocks * chunk_k * block_d * block_n, dtype=torch.float32, device=dev
    )

    grid = (batch, n_d_blocks)
    # block_d<=16 only at the large-state regime (block_n=128 => block_d=16),
    # where the sweep finds num_warps=2 beats 4 (fewer warps, less per-thread
    # overhead on the small-block_d serial loop); all other shapes unchanged.
    warps = (
        num_warps
        if num_warps is not None
        else (2 if block_d <= 16 else (4 if block_d * block_n >= 512 else 2))
    )
    if config is not None and config.num_warps is not None:
        warps = config.num_warps
    extra: dict[str, int] = {}
    if config is not None and config.num_stages is not None:
        extra["num_stages"] = config.num_stages
    _bwd_scan_kernel[grid](
        u_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        d_c,
        dy_c,
        grad_u,
        grad_delta,
        ga_part,
        gb_part,
        gc_part,
        gd_part,
        ckpt,
        hbuf,
        seq_len,
        d_model,
        n_state,
        n_chunks,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CHUNK_K=chunk_k,
        num_warps=warps,
        **extra,
    )

    # Deterministic cross-program reductions (fixed shapes -> fixed reduction
    # trees; byte-identical across runs, unlike float atomics).
    grad_a = ga_part.sum(dim=0).to(a.dtype)
    grad_b = gb_part.sum(dim=1).to(b.dtype)
    grad_c = gc_part.sum(dim=1).to(c.dtype)
    grad_d = gd_part.sum(dim=0).to(d_skip.dtype)
    return grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d


def resource_meta() -> dict[str, int] | None:
    """Resource envelope across all compiled specialisations of the kernel."""
    return collect_resource_meta(_bwd_scan_kernel)
