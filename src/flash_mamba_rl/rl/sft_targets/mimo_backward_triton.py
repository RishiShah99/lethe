"""Triton kernel for the Mamba-3 MIMO selective-scan backward.

No open implementation of this op exists (the official repo ships only
forward + decode-step kernels for MIMO). No ``tl.dot`` — every contraction
is elementwise FMAs + ``tl.sum``, so the sm_100 TMEM-promotion pass never
engages. No atomics: cross-program sums go through per-program partial
buffers reduced by deterministic ``torch.sum`` in the launcher (ORD-02).
The op has no transcendentals (``dt``/``alpha`` arrive precomputed).

Key structural fact: ``alpha`` is rank-independent and the readout
distributes identically over ranks, so the backward state gradient is the
same for every rank — one carry ``g[p, n]``, and the aggregated state obeys
``h_agg_t = alpha_t * h_agg_{t-1} + sum_r (dt_t * B_t^r) * x_r_t^r``.

Layout (enforced via ``.contiguous()``):

    x, dy, grad_x              : [B, L, H, P]   row-major
    B, C, grad_B, grad_C       : [B, L, R, H, N]
    dt, alpha                  : [B, L, H]
    mimo_x, mimo_o             : [H, R, P]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# One program holds the whole headdim; P above this needs a multi-block design.
MAX_BLOCK_P = 128
# static_range unrolls the rank loop R times per step; Mamba-3 uses R in {1,2,4,8}.
MAX_RANK = 16
MAX_BLOCK_N = 64

# Recompute-chunk cap: the in-chunk scratch working set is
# n_programs * K * BLOCK_P * BLOCK_N fp32; K=16 at (B=8, H=64, P=64,
# N=128, BLOCK_N=64) is ~268 MB — above B200's 126 MB L2, so K (and
# BLOCK_N) are autotuning levers; correctness never depends on K.
MAX_CHUNK_K = 16


def _chunk_k(seq_len: int) -> int:
    """Largest power-of-2 divisor of seq_len, capped at MAX_CHUNK_K."""
    return min(seq_len & (-seq_len), MAX_CHUNK_K)


@triton.jit  # type: ignore[untyped-decorator]
def _mimo_bwd_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    b_ptr,
    c_ptr,
    dt_ptr,
    alpha_ptr,
    mimo_x_ptr,
    mimo_o_ptr,
    dy_ptr,
    grad_b_ptr,
    grad_c_ptr,
    gx_part_ptr,  # [nNb, B, L, H, P] fp32, reduced over nNb by the launcher
    gdt_part_ptr,  # [nNb, B, L, H] fp32, reduced over nNb
    galpha_part_ptr,  # [nNb, B, L, H] fp32, reduced over nNb
    gmx_part_ptr,  # [nNb, B, H, R, P] fp32, reduced over (nNb, B)
    gmo_part_ptr,  # [nNb, B, H, R, P] fp32, reduced over (nNb, B)
    ckpt_ptr,  # [B * H * nNb * n_chunks * BLOCK_P * BLOCK_N] fp32 scratch
    hbuf_ptr,  # [B * H * nNb * CHUNK_K * BLOCK_P * BLOCK_N] fp32 scratch
    seq_len,
    nheads,
    headdim,
    n_state,
    n_chunks,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    n_n_blocks = tl.num_programs(2)

    offs_p = tl.arange(0, BLOCK_P)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    mask_p = offs_p < headdim
    mask_n = offs_n < n_state
    mask_pn = mask_p[:, None] & mask_n[None, :]

    offs_tile = tl.arange(0, BLOCK_P)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    pid_flat = ((pid_b * nheads + pid_h) * n_n_blocks + pid_n).to(tl.int64)
    ckpt_prog = ckpt_ptr + pid_flat * n_chunks * BLOCK_P * BLOCK_N + offs_tile
    hbuf_prog = hbuf_ptr + pid_flat * CHUNK_K * BLOCK_P * BLOCK_N + offs_tile

    # ----- Phase 1: forward sweep, checkpointing h_agg entering each chunk.
    h = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    # Fold batch into the int64 base before multiplying by row strides:
    # bare int32 products overflow past L*R*H*N ~ 2^31 (C2 invariant).
    xrow = (pid_b.to(tl.int64) * seq_len * nheads + pid_h) * headdim + offs_p
    brow = (pid_b.to(tl.int64) * seq_len * R * nheads + pid_h) * n_state + offs_n
    srow = pid_b.to(tl.int64) * seq_len * nheads + pid_h
    x_step = nheads * headdim
    b_step = R * nheads * n_state
    b_rstride = nheads * n_state
    s_step = nheads

    for c in range(n_chunks):
        tl.store(ckpt_prog + c * BLOCK_P * BLOCK_N, h)
        for _j in range(CHUNK_K):
            x_t = tl.load(x_ptr + xrow, mask=mask_p, other=0.0).to(tl.float32)
            dt_t = tl.load(dt_ptr + srow).to(tl.float32)
            alpha_t = tl.load(alpha_ptr + srow).to(tl.float32)
            upd = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
            for r in tl.static_range(R):
                b_r = tl.load(b_ptr + brow + r * b_rstride, mask=mask_n, other=0.0).to(tl.float32)
                mx_r = tl.load(
                    mimo_x_ptr + (pid_h * R + r) * headdim + offs_p, mask=mask_p, other=0.0
                ).to(tl.float32)
                upd += ((x_t * mx_r)[:, None]) * ((dt_t * b_r)[None, :])
            # Padding lanes must stay exactly zero through the whole update
            # (C1 lesson): a NaN minted there poisons the real lanes'
            # reductions downstream.
            h = tl.where(mask_pn, alpha_t * h + upd, 0.0)
            xrow += x_step
            brow += b_step
            srow += s_step

    # ----- Phase 2: chunks newest-first; recompute in-chunk pre-update
    # states, then reverse-sweep. ag_carry = alpha_{t+1} * g_{t+1} crosses
    # chunk boundaries in registers because t descends globally.
    ag_carry = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
    gmx_acc = tl.zeros((BLOCK_R, BLOCK_P), dtype=tl.float32)
    gmo_acc = tl.zeros((BLOCK_R, BLOCK_P), dtype=tl.float32)

    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        t0 = c * CHUNK_K

        h_prev = tl.load(ckpt_prog + c * BLOCK_P * BLOCK_N)
        xbase = ((pid_b.to(tl.int64) * seq_len + t0) * nheads + pid_h) * headdim + offs_p
        bbase = ((pid_b.to(tl.int64) * seq_len + t0) * R * nheads + pid_h) * n_state + offs_n
        sbase = (pid_b.to(tl.int64) * seq_len + t0) * nheads + pid_h
        for j in range(CHUNK_K):
            tl.store(hbuf_prog + j * BLOCK_P * BLOCK_N, h_prev)
            x_t = tl.load(x_ptr + xbase + j * x_step, mask=mask_p, other=0.0).to(tl.float32)
            dt_t = tl.load(dt_ptr + sbase + j * s_step).to(tl.float32)
            alpha_t = tl.load(alpha_ptr + sbase + j * s_step).to(tl.float32)
            upd = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
            for r in tl.static_range(R):
                b_r = tl.load(
                    b_ptr + bbase + j * b_step + r * b_rstride, mask=mask_n, other=0.0
                ).to(tl.float32)
                mx_r = tl.load(
                    mimo_x_ptr + (pid_h * R + r) * headdim + offs_p, mask=mask_p, other=0.0
                ).to(tl.float32)
                upd += ((x_t * mx_r)[:, None]) * ((dt_t * b_r)[None, :])
            h_prev = tl.where(mask_pn, alpha_t * h_prev + upd, 0.0)

        h_cur = h_prev
        for jj in range(CHUNK_K):
            j = CHUNK_K - 1 - jj
            xoff = xbase + j * x_step
            boff = bbase + j * b_step
            soff = sbase + j * s_step

            dy_t = tl.load(dy_ptr + xoff, mask=mask_p, other=0.0).to(tl.float32)
            x_t = tl.load(x_ptr + xoff, mask=mask_p, other=0.0).to(tl.float32)
            dt_t = tl.load(dt_ptr + soff).to(tl.float32)
            alpha_t = tl.load(alpha_ptr + soff).to(tl.float32)
            h_tm1 = tl.load(hbuf_prog + j * BLOCK_P * BLOCK_N)

            # g_t = dL/dh_agg_t (rank-uniform). dy in padded lanes loads as
            # 0, but Inf*0 across broadcasts mints NaN in padded lanes — mask.
            gsum = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)
            for r in tl.static_range(R):
                c_r = tl.load(c_ptr + boff + r * b_rstride, mask=mask_n, other=0.0).to(tl.float32)
                mo_r = tl.load(
                    mimo_o_ptr + (pid_h * R + r) * headdim + offs_p, mask=mask_p, other=0.0
                ).to(tl.float32)
                gy_r = dy_t * mo_r
                gsum += gy_r[:, None] * c_r[None, :]
            g = tl.where(mask_pn, gsum + ag_carry, 0.0)

            gdt_t = 0.0
            gx_t = tl.zeros((BLOCK_P,), dtype=tl.float32)
            for r in tl.static_range(R):
                b_r = tl.load(b_ptr + boff + r * b_rstride, mask=mask_n, other=0.0).to(tl.float32)
                c_r = tl.load(c_ptr + boff + r * b_rstride, mask=mask_n, other=0.0).to(tl.float32)
                mx_r = tl.load(
                    mimo_x_ptr + (pid_h * R + r) * headdim + offs_p, mask=mask_p, other=0.0
                ).to(tl.float32)
                mo_r = tl.load(
                    mimo_o_ptr + (pid_h * R + r) * headdim + offs_p, mask=mask_p, other=0.0
                ).to(tl.float32)
                gy_r = dy_t * mo_r
                xr_r = x_t * mx_r
                tmp1_r = dt_t * b_r

                gc_r = tl.sum(tl.where(mask_pn, gy_r[:, None] * h_cur, 0.0), axis=0)
                tl.store(
                    grad_c_ptr + boff + r * b_rstride,
                    gc_r.to(grad_c_ptr.dtype.element_ty),
                    mask=mask_n,
                )
                yraw_r = tl.sum(tl.where(mask_pn, h_cur * c_r[None, :], 0.0), axis=1)

                # Autograd's grouping: grad_tmp1 = sum_p(g * x_r), then * dt
                # for grad_B and * B (summed) for grad_dt.
                gt1_r = tl.sum(tl.where(mask_pn, g * xr_r[:, None], 0.0), axis=0)
                tl.store(
                    grad_b_ptr + boff + r * b_rstride,
                    (gt1_r * dt_t).to(grad_b_ptr.dtype.element_ty),
                    mask=mask_n,
                )
                # Unmasked sums here and at galpha_t below are safe only
                # because g, gt1_r and h_tm1 are zeroed at padded lanes
                # upstream (g at its tl.where, h via the recompute mask) —
                # re-check if any of those producers change.
                gdt_t += tl.sum(gt1_r * b_r)
                gxr_r = tl.sum(tl.where(mask_pn, g * tmp1_r[None, :], 0.0), axis=1)
                gx_t += gxr_r * mx_r

                # Per-rank accumulators persist across t as [BLOCK_R, BLOCK_P]
                # tiles; static_range r selects its row via the lane mask.
                row_r = offs_r == r
                gmx_acc += tl.where(row_r[:, None], (gxr_r * x_t)[None, :], 0.0)
                gmo_acc += tl.where(row_r[:, None], (dy_t * yraw_r)[None, :], 0.0)

            galpha_t = tl.sum(g * h_tm1)
            part_off = ((pid_n * tl.num_programs(0) + pid_b).to(tl.int64) * seq_len + t0 + j) * (
                nheads
            ) + pid_h
            tl.store(gdt_part_ptr + part_off, gdt_t)
            tl.store(galpha_part_ptr + part_off, galpha_t)
            tl.store(gx_part_ptr + part_off * headdim + offs_p, gx_t, mask=mask_p)

            ag_carry = tl.where(mask_pn, alpha_t * g, 0.0)
            h_cur = h_tm1

    mrow = ((pid_n * tl.num_programs(0) + pid_b).to(tl.int64) * nheads + pid_h) * R * headdim
    moffs = offs_r[:, None] * headdim + offs_p[None, :]
    mask_rp = (offs_r[:, None] < R) & mask_p[None, :]
    tl.store(gmx_part_ptr + mrow + moffs, gmx_acc, mask=mask_rp)
    tl.store(gmo_part_ptr + mrow + moffs, gmo_acc, mask=mask_rp)


def launch_mimo_backward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
    *,
    num_warps: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Launch the Triton MIMO backward on CUDA tensors.

    Dispatch keys on ``x`` (device, dtype); the kernel upcasts every load to
    fp32 and each gradient stores in its own input's dtype — mixed-dtype
    inputs are tolerated identically to the eager path, not validated.
    Returns ``(grad_x, grad_B, grad_C, grad_dt, grad_alpha, grad_mimo_x,
    grad_mimo_o)`` in the corresponding input dtypes. ``num_warps`` overrides
    the launch config for the bench's compile-behaviour sweep.
    """
    batch, seq_len, nheads, headdim = x.shape
    rank = B.shape[2]
    n_state = B.shape[4]

    block_p = triton.next_power_of_2(headdim)
    if block_p > MAX_BLOCK_P:
        raise ValueError(f"headdim={headdim} exceeds single-block budget {MAX_BLOCK_P}")
    if rank > MAX_RANK:
        raise ValueError(f"rank={rank} exceeds unroll budget {MAX_RANK}")
    block_r = triton.next_power_of_2(rank)
    block_n = min(MAX_BLOCK_N, triton.next_power_of_2(n_state))
    n_n_blocks = triton.cdiv(n_state, block_n)
    chunk_k = _chunk_k(seq_len)
    n_chunks = seq_len // chunk_k

    x_c = x.contiguous()
    b_c = B.contiguous()
    c_c = C.contiguous()
    dt_c = dt.contiguous()
    alpha_c = alpha.contiguous()
    mx_c = mimo_x.contiguous()
    mo_c = mimo_o.contiguous()
    dy_c = dy.contiguous()

    dev = x.device
    grad_b = torch.empty_like(b_c)
    grad_c = torch.empty_like(c_c)
    # Partials and scratch stay fp32: internal compute is fp32 and the single
    # round to the input dtype happens after the launcher's reduction.
    gx_part = torch.empty(
        n_n_blocks, batch, seq_len, nheads, headdim, dtype=torch.float32, device=dev
    )
    gdt_part = torch.empty(n_n_blocks, batch, seq_len, nheads, dtype=torch.float32, device=dev)
    galpha_part = torch.empty(n_n_blocks, batch, seq_len, nheads, dtype=torch.float32, device=dev)
    gmx_part = torch.empty(
        n_n_blocks, batch, nheads, rank, headdim, dtype=torch.float32, device=dev
    )
    gmo_part = torch.empty(
        n_n_blocks, batch, nheads, rank, headdim, dtype=torch.float32, device=dev
    )
    n_programs = batch * nheads * n_n_blocks
    # ckpt is HBM-resident O(B*H*nNb*(L/CHUNK_K)*BLOCK_P*BLOCK_N) fp32 — ~2 GB
    # at (B8, L4096, H32, P64, N128); only hbuf's in-chunk slice is L2-hot.
    ckpt = torch.empty(n_programs * n_chunks * block_p * block_n, dtype=torch.float32, device=dev)
    hbuf = torch.empty(n_programs * chunk_k * block_p * block_n, dtype=torch.float32, device=dev)

    grid = (batch, nheads, n_n_blocks)
    # B200 resource envelope sits at the 255-reg ceiling with <=194 B spill
    # — zero headroom: anything that adds register pressure (wider MAX_RANK
    # unroll, larger BLOCK_P) must re-check RES-02 and the spill column.
    warps = num_warps if num_warps is not None else (4 if block_p * block_n >= 512 else 2)
    _mimo_bwd_kernel[grid](
        x_c,
        b_c,
        c_c,
        dt_c,
        alpha_c,
        mx_c,
        mo_c,
        dy_c,
        grad_b,
        grad_c,
        gx_part,
        gdt_part,
        galpha_part,
        gmx_part,
        gmo_part,
        ckpt,
        hbuf,
        seq_len,
        nheads,
        headdim,
        n_state,
        n_chunks,
        R=rank,
        BLOCK_R=block_r,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
        CHUNK_K=chunk_k,
        num_warps=warps,
    )

    # Deterministic cross-program reductions (fixed shapes -> fixed reduction
    # trees; byte-identical across runs, unlike float atomics).
    grad_x = gx_part.sum(dim=0).to(x.dtype)
    grad_dt = gdt_part.sum(dim=0).to(dt.dtype)
    grad_alpha = galpha_part.sum(dim=0).to(alpha.dtype)
    grad_mimo_x = gmx_part.sum(dim=(0, 1)).to(mimo_x.dtype)
    grad_mimo_o = gmo_part.sum(dim=(0, 1)).to(mimo_o.dtype)
    return grad_x, grad_b, grad_c, grad_dt, grad_alpha, grad_mimo_x, grad_mimo_o


def _mimo_forward_eager(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
) -> Tensor:
    batch, seqlen, nheads, headdim = x.shape
    rank = B.shape[2]
    d_state = B.shape[4]

    mimo_x_bc = mimo_x.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)
    x_r = x.unsqueeze(2) * mimo_x_bc

    h = torch.zeros(batch, rank, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)
    for t in range(seqlen):
        alpha_t = alpha[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        dt_t = dt[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        B_t = B[:, t, :, :, :].unsqueeze(3)
        x_r_t = x_r[:, t, :, :, :].unsqueeze(-1)
        h = alpha_t * h + dt_t * B_t * x_r_t
        h_agg = h.sum(dim=1)
        C_t = C[:, t, :, :, :].unsqueeze(3)
        y_raw = (h_agg.unsqueeze(1) * C_t).sum(-1)
        mimo_o_bc = mimo_o.permute(1, 0, 2).unsqueeze(0)
        y[:, t, :, :] = (y_raw * mimo_o_bc).sum(1)
    return y


def _mimo_backward_eager(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x, B, C, dt, alpha, mimo_x, mimo_o, dy = (
            t.to(torch.float32) for t in (x, B, C, dt, alpha, mimo_x, mimo_o, dy)
        )
    leaves = [t.detach().requires_grad_(True) for t in (x, B, C, dt, alpha, mimo_x, mimo_o)]
    y = _mimo_forward_eager(*leaves)
    grads = torch.autograd.grad(outputs=y, inputs=leaves, grad_outputs=dy)
    g_x, g_b, g_c, g_dt, g_alpha, g_mx, g_mo = grads
    return (
        g_x.to(out_dtype),
        g_b.to(out_dtype),
        g_c.to(out_dtype),
        g_dt.to(out_dtype),
        g_alpha.to(out_dtype),
        g_mx.to(out_dtype),
        g_mo.to(out_dtype),
    )


def mimo_backward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Mamba-3 MIMO SSM backward pass.

    Args/shapes: ``x``/``dy`` [B, L, H, P], ``B``/``C`` [B, L, R, H, N],
    ``dt``/``alpha`` [B, L, H], ``mimo_x``/``mimo_o`` [H, R, P].
    Returns ``(grad_x, grad_B, grad_C, grad_dt, grad_alpha, grad_mimo_x,
    grad_mimo_o)`` matching corresponding input shapes and dtypes.
    """
    # Device residency: non-CUDA (and fp64) inputs take the eager path.
    if not (x.is_cuda and x.dtype in (torch.float32, torch.float16, torch.bfloat16)):
        return _mimo_backward_eager(x, B, C, dt, alpha, mimo_x, mimo_o, dy)
    return launch_mimo_backward(x, B, C, dt, alpha, mimo_x, mimo_o, dy)
