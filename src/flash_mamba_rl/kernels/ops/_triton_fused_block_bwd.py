"""Triton kernels for the Mamba fused-block backward (C6) — the training path.

Import this module only when ``triton`` is installed and a CUDA device is
the target — the public dispatcher in ``fused_block_backward.py`` guards
both. Layout assumptions (enforced by the launcher via ``.contiguous()``):

    x, grad_x                 : [B, L, D]      L = L_out + K - 1 (pre-padded)
    conv_weight               : [D, 1, K]
    delta, dy, grad_delta     : [B, L_out, D]
    A : [D, N]   B, C : [B, L_out, N]   conv_bias, D_skip, norm_weight : [D]

Four kernels, one public op. The structure is forced, not chosen:

- The reverse scan sweep consumes ``dys`` (the RMSNorm-backward output),
  and ``dys`` at any (b, t) needs the *complete* ``ys`` row — which only
  exists after the forward sweep of every program has finished. That is a
  global barrier, and Triton has no grid sync, so the forward sweep and
  the reverse sweep are separate launches (where C2 fused them: its
  upstream gradient was given, not computed).
- The RMSNorm backward needs cross-D reductions per (b, t) — the same
  constraint that split C5 into two kernels (atomics are forbidden,
  ORD-02).
- The conv input-gradient is written in gather form (one kernel, parallel
  over every (b, s, d): grad_x[s] = sum_k w[k] * dconv[s-k]) instead of
  scatter-accumulating inside the serial sweep — no read-modify-write, no
  accumulation hazard, embarrassingly parallel.

Pipeline (stagings are fp32; one round per gradient at the very end):
  1. ``_fwd_stage_kernel``   — C5's fused conv+SiLU+scan forward, plus
     per-chunk state checkpoints (C2's scheme) and the ``ys`` staging.
  2. ``_norm_bwd_kernel``    — per (batch, t-block): recompute rms from
     ``ys``, emit ``dys`` and per-program norm-weight partials.
  3. ``_bwd_sweep_kernel``   — C2's newest-first chunked reverse sweep
     with the conv+SiLU recompute riding the in-chunk forward pass (z and
     conv are scratch-buffered per chunk, so the reverse walk re-reads
     them instead of re-deriving); emits grad_delta directly, dconv
     staging, and per-program partials for A/B/C/D/conv_w/conv_bias.
  4. ``_conv_x_bwd_kernel``  — the gather-form grad_x.
  Launcher then runs deterministic ``torch.sum`` reductions over the
  partial buffers (fixed shapes -> fixed reduction trees, ORD-02).

C2 invariants carried verbatim: no tl.dot (the #904 TMEM-promotion pass
never engages), no atomics, int64 offset bases with int32 per-step
increments, fp32 everywhere internally, libdevice exp/log1p, masked padded
lanes exactly zero, never invert the recurrence (checkpoint + in-chunk
recompute), gradient expressions mirror autograd's dataflow grouping (the
EXC-01 lesson):

- RMSNorm backward follows autograd's chain through mul/div/sqrt/mean/pow:
  ``dys = (dy*w)/r + (2*ys) * ((-sdw/r^2) / (2r) / D)`` with
  ``sdw = sum_d (dy*w)*ys``. Factoring 1/r^2 out of the sdw sum is
  mask-equivalent in every reachable case (r >= sqrt(eps) > 0 for finite
  ys; an Inf in ys makes both forms NaN through Inf/Inf; the one excluded
  window is all-finite overflow of the undivided sdw sum, unreachable at
  the gate contract's randn scales) — pinned by the replica's non-finite
  tests.
- SiLU': ``(dz * sig) * (1 + conv * (1 - sig))`` — aten silu_backward's
  grouping.
- The scan adjoint is C2's, with z (the SiLU output) in place of u.
- The sweep's in-chunk recompute keeps C2's ``b_bar = dbar*B`` grouping
  (the dataflow the gradient expressions consume); stage 1 keeps C5's
  ``(dbar*z)*B`` so the staged ``ys`` reproduces the forward bit-for-bit.
  The two trajectories diverge at ULP level — regrouping a product cannot
  change NaN/Inf class, and the divergence sits inside the calibrated
  tolerances.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from flash_mamba_rl.kernels.ops._resource_meta import collect_resource_meta
from flash_mamba_rl.kernels.ops._triton_bwd_scan import MAX_BLOCK_N, _chunk_k
from flash_mamba_rl.kernels.ops._triton_fused_block import (
    _MAX_BLOCK_D_NORM,
    _NORM_TILE,
    MAX_CONV_K,
)

try:  # moved between minor versions
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)


@triton.jit  # type: ignore[untyped-decorator]
def _fwd_stage_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    ys_ptr,
    ckpt_ptr,  # [B * nDb * n_chunks * BLOCK_D * BLOCK_N] fp32 scratch
    l_out,
    d_model,
    n_state,
    n_chunks,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    n_d_blocks = tl.num_programs(1)

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
    pid_bd = (pid_b * n_d_blocks + pid_d).to(tl.int64)
    ckpt_prog = ckpt_ptr + pid_bd * n_chunks * BLOCK_D * BLOCK_N + offs_tile

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    seq_in = l_out + CONV_K - 1
    x_off = pid_b.to(tl.int64) * seq_in * d_model + offs_d
    od_off = pid_b.to(tl.int64) * l_out * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * l_out * n_state + offs_n

    for c in range(n_chunks):
        # c is int32: promote before the tile-stride product (it crosses
        # 2^31 once the per-program ckpt region exceeds 8 GiB).
        tl.store(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N, h)  # type: ignore[attr-defined]
        for _j in range(CHUNK_K):
            xw = tl.load(
                x_ptr + x_off[:, None] + offs_k[None, :] * d_model, mask=mask_dk, other=0.0
            ).to(tl.float32)
            conv = tl.sum(conv_w * xw, axis=1) + conv_b
            z = conv * (1.0 / (1.0 + libdevice.exp(-conv)))

            dlt = tl.load(delta_ptr + od_off, mask=mask_d, other=0.0).to(tl.float32)
            b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
            c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

            dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
            a_bar = libdevice.exp(dbar[:, None] * a)
            bu = (dbar * z)[:, None] * b_t[None, :]
            h = tl.where(mask_dn, a_bar * h + bu, 0.0)

            ys_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * z
            tl.store(ys_ptr + od_off, ys_t, mask=mask_d)

            x_off += d_model
            od_off += d_model
            bln_off += n_state


@triton.jit  # type: ignore[untyped-decorator]
def _norm_bwd_kernel(  # type: ignore[no-untyped-def]
    ys_ptr,
    dy_ptr,
    w_ptr,
    dys_ptr,
    gnw_part_ptr,  # [B, nTb, D] fp32, reduced over (B, nTb) by the launcher
    l_out,
    d_model,
    eps,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    n_t_blocks = tl.num_programs(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < l_out
    row_off = (pid_b.to(tl.int64) * l_out + offs_t) * d_model

    ssq = tl.zeros((BLOCK_T,), dtype=tl.float32)
    sdw = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for d0 in range(0, d_model, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_model
        mask = mask_t[:, None] & mask_d[None, :]
        ys = tl.load(ys_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        ssq += tl.sum(ys * ys, axis=1)
        # div-backward integrand (dy*w)*ys; the 1/r^2 factor is applied
        # after the loop (mask-equivalent, see module docstring).
        sdw += tl.sum(tl.where(mask, (dy * w[None, :]) * ys, 0.0), axis=1)

    r = libdevice.sqrt(ssq / d_model + eps)
    # autograd chain: dr = -sdw/r^2 ; dm = dr/(2r) ; per-element 2*ys*dm/D.
    dm_over_d = (-sdw / (r * r)) / (2.0 * r) / d_model

    pid_bt = pid_b.to(tl.int64) * n_t_blocks + pid_t
    gnw_acc_base = gnw_part_ptr + pid_bt * d_model
    for d0 in range(0, d_model, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_model
        mask = mask_t[:, None] & mask_d[None, :]
        ys = tl.load(ys_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0)
        dy = tl.load(dy_ptr + row_off[:, None] + offs_d[None, :], mask=mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        dys = (dy * w[None, :]) / r[:, None] + (2.0 * ys) * dm_over_d[:, None]
        tl.store(dys_ptr + row_off[:, None] + offs_d[None, :], dys, mask=mask)

        # grad_norm_weight integrand dy * (ys/r); masked rows contribute
        # exact zeros to the per-program partial.
        gnw_t = tl.sum(tl.where(mask, dy * (ys / r[:, None]), 0.0), axis=0)
        tl.store(gnw_acc_base + offs_d, gnw_t, mask=mask_d)


@triton.jit  # type: ignore[untyped-decorator]
def _bwd_sweep_kernel(  # type: ignore[no-untyped-def]
    x_ptr,
    conv_w_ptr,
    conv_b_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    dys_ptr,
    grad_delta_ptr,
    dconv_ptr,  # [B, L_out, D] fp32 staging for the grad_x gather kernel
    ga_part_ptr,  # [B, D, N] fp32
    gb_part_ptr,  # [B, nDb, L_out, N] fp32
    gc_part_ptr,  # [B, nDb, L_out, N] fp32
    gd_part_ptr,  # [B, D] fp32
    gw_part_ptr,  # [B, D, CONV_K] fp32
    gcb_part_ptr,  # [B, D] fp32
    ckpt_ptr,
    hbuf_ptr,  # [B * nDb * CHUNK_K * BLOCK_D * BLOCK_N] fp32 scratch
    zbuf_ptr,  # [B * nDb * CHUNK_K * BLOCK_D] fp32 scratch (z per in-chunk step)
    cvbuf_ptr,  # [B * nDb * CHUNK_K * BLOCK_D] fp32 scratch (conv pre-activation)
    l_out,
    d_model,
    n_state,
    n_chunks,
    CONV_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CHUNK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    n_d_blocks = tl.num_programs(1)

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
    pid_bd = (pid_b * n_d_blocks + pid_d).to(tl.int64)
    ckpt_prog = ckpt_ptr + pid_bd * n_chunks * BLOCK_D * BLOCK_N + offs_tile
    hbuf_prog = hbuf_ptr + pid_bd * CHUNK_K * BLOCK_D * BLOCK_N + offs_tile
    zbuf_prog = zbuf_ptr + pid_bd * CHUNK_K * BLOCK_D + tl.arange(0, BLOCK_D)
    cvbuf_prog = cvbuf_ptr + pid_bd * CHUNK_K * BLOCK_D + tl.arange(0, BLOCK_D)

    seq_in = l_out + CONV_K - 1

    ag_carry = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    ga_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    gw_acc = tl.zeros((BLOCK_D, BLOCK_K), dtype=tl.float32)
    gcb_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        t0 = c * CHUNK_K

        h_prev = tl.load(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N)
        x_base = (pid_b.to(tl.int64) * seq_in + t0) * d_model + offs_d
        od_base = (pid_b.to(tl.int64) * l_out + t0) * d_model + offs_d
        bln_base = (pid_b.to(tl.int64) * l_out + t0) * n_state + offs_n
        for j in range(CHUNK_K):
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

        h_cur = h_prev
        gbc_base = (pid_bd * l_out + t0) * n_state + offs_n
        for jj in range(CHUNK_K):
            j = CHUNK_K - 1 - jj
            od = od_base + j * d_model
            bln = bln_base + j * n_state

            dys_t = tl.load(dys_ptr + od, mask=mask_d, other=0.0)
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

            # grad_z (C2's grad_u with z as the scan input), then SiLU' and
            # the conv-weight/bias integrands — the x window is reloaded
            # from L2 for gw (same columns the recompute pass just read).
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

    ga_off = pid_b.to(tl.int64) * d_model * n_state + offs_d[:, None] * n_state + offs_n[None, :]
    tl.store(ga_part_ptr + ga_off, ga_acc, mask=mask_dn)
    gd_off = pid_b.to(tl.int64) * d_model + offs_d
    tl.store(gd_part_ptr + gd_off, gd_acc, mask=mask_d)
    gw_off = pid_b.to(tl.int64) * d_model * CONV_K + offs_d[:, None] * CONV_K + offs_k[None, :]
    tl.store(gw_part_ptr + gw_off, gw_acc, mask=mask_dk)
    tl.store(gcb_part_ptr + gd_off, gcb_acc, mask=mask_d)


@triton.jit  # type: ignore[untyped-decorator]
def _conv_x_bwd_kernel(  # type: ignore[no-untyped-def]
    dconv_ptr,
    conv_w_ptr,
    grad_x_ptr,
    seq_in,
    l_out,
    d_model,
    CONV_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < seq_in
    mask_d = offs_d < d_model

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    dconv_base = pid_b.to(tl.int64) * l_out * d_model
    for k in tl.static_range(CONV_K):
        src_t = offs_t - k
        mask_src = mask_t & (src_t >= 0) & (src_t < l_out)
        v = tl.load(
            dconv_ptr + dconv_base + src_t[:, None].to(tl.int64) * d_model + offs_d[None, :],
            mask=mask_src[:, None] & mask_d[None, :],
            other=0.0,
        )
        w_k = tl.load(conv_w_ptr + offs_d * CONV_K + k, mask=mask_d, other=0.0).to(tl.float32)
        acc += v * w_k[None, :]

    gx_off = pid_b.to(tl.int64) * seq_in * d_model + offs_t[:, None].to(tl.int64) * d_model
    tl.store(
        grad_x_ptr + gx_off + offs_d[None, :],
        acc.to(grad_x_ptr.dtype.element_ty),
        mask=mask_t[:, None] & mask_d[None, :],
    )


def launch_fused_block_backward(
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
    num_warps: int | None = None,
) -> tuple[Tensor, ...]:
    """Launch the four-kernel backward. Inputs must be CUDA tensors of one dtype.

    Returns the nine gradients in ``FusedBlockGrads`` field order, each in
    its input's dtype. ``num_warps`` overrides the two serial-sweep
    kernels' heuristic (bench hook); the norm-backward and gather kernels
    keep their own.

    Checkpoint scratch is ``4 * B * ceil(D/64) * n_chunks * BLOCK_D *
    BLOCK_N`` bytes with ``n_chunks = l_out / chunk_k(l_out)``; chunk_k is
    the largest power-of-2 divisor of l_out capped at 16, so an odd l_out
    collapses it to 1 and inflates the buffer 16x vs the even-L envelope
    (the public op's chunk_size divisibility check keeps default callers
    out of that regime).
    """
    batch, seq_in, d_model = x.shape
    conv_k = conv_weight.shape[-1]
    n_state = a.shape[1]
    l_out = seq_in - (conv_k - 1)

    if conv_k > MAX_CONV_K:
        raise ValueError(f"conv kernel size {conv_k} exceeds window budget {MAX_CONV_K}")
    if l_out < 1:
        raise ValueError(f"sequence length {seq_in} shorter than conv window {conv_k}")
    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))
    block_k = triton.next_power_of_2(conv_k)
    chunk_k = _chunk_k(l_out)
    n_chunks = l_out // chunk_k
    n_d_blocks = triton.cdiv(d_model, block_d)

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
    ckpt = torch.empty(batch * n_d_blocks * n_chunks * block_d * block_n, dtype=f32, device=dev)
    hbuf = torch.empty(batch * n_d_blocks * chunk_k * block_d * block_n, dtype=f32, device=dev)
    zbuf = torch.empty(batch * n_d_blocks * chunk_k * block_d, dtype=f32, device=dev)
    cvbuf = torch.empty(batch * n_d_blocks * chunk_k * block_d, dtype=f32, device=dev)
    ga_part = torch.empty(batch, d_model, n_state, dtype=f32, device=dev)
    gb_part = torch.empty(batch, n_d_blocks, l_out, n_state, dtype=f32, device=dev)
    gc_part = torch.empty(batch, n_d_blocks, l_out, n_state, dtype=f32, device=dev)
    gd_part = torch.empty(batch, d_model, dtype=f32, device=dev)
    gw_part = torch.empty(batch, d_model, conv_k, dtype=f32, device=dev)
    gcb_part = torch.empty(batch, d_model, dtype=f32, device=dev)

    grid_sweep = (batch, n_d_blocks)
    # num_warps=8 measured fastest at every (shape, dtype) cell on B200
    # (scratch/c6_warp_probe.py) — and at num_warps<=4 ptxas collapses the
    # half-dtype _bwd_sweep_kernel specs to 32 registers with 1.4-2.6 KB
    # spill (train bf16: 722 ms at nw=4 vs 114 ms at nw=8); nw=8 compiles
    # every spec at >=219 regs with <=172 B spill.
    warps = num_warps if num_warps is not None else (8 if block_d * block_n >= 512 else 2)
    _fwd_stage_kernel[grid_sweep](
        x_c,
        conv_w_c,
        conv_b_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        dskip_c,
        ys,
        ckpt,
        l_out,
        d_model,
        n_state,
        n_chunks,
        CONV_K=conv_k,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CHUNK_K=chunk_k,
        num_warps=warps,
    )

    block_d_norm = min(_MAX_BLOCK_D_NORM, triton.next_power_of_2(d_model))
    block_t = max(1, _NORM_TILE // block_d_norm)
    n_t_blocks = triton.cdiv(l_out, block_t)
    gnw_part = torch.empty(batch, n_t_blocks, d_model, dtype=f32, device=dev)
    grid_norm = (batch, n_t_blocks)
    num_warps_norm = 4 if block_t * block_d_norm >= 512 else 2
    _norm_bwd_kernel[grid_norm](
        ys,
        dy_c,
        norm_w_c,
        dys,
        gnw_part,
        l_out,
        d_model,
        eps,
        BLOCK_T=block_t,
        BLOCK_D=block_d_norm,
        num_warps=num_warps_norm,
    )

    _bwd_sweep_kernel[grid_sweep](
        x_c,
        conv_w_c,
        conv_b_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        dskip_c,
        dys,
        grad_delta,
        dconv,
        ga_part,
        gb_part,
        gc_part,
        gd_part,
        gw_part,
        gcb_part,
        ckpt,
        hbuf,
        zbuf,
        cvbuf,
        l_out,
        d_model,
        n_state,
        n_chunks,
        CONV_K=conv_k,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        CHUNK_K=chunk_k,
        num_warps=warps,
    )

    block_t_x = max(1, _NORM_TILE // block_d_norm)
    grid_x = (batch, triton.cdiv(seq_in, block_t_x), triton.cdiv(d_model, block_d_norm))
    _conv_x_bwd_kernel[grid_x](
        dconv,
        conv_w_c,
        grad_x,
        seq_in,
        l_out,
        d_model,
        CONV_K=conv_k,
        BLOCK_T=block_t_x,
        BLOCK_D=block_d_norm,
        num_warps=num_warps_norm,
    )

    grad_a = ga_part.sum(dim=0).to(a.dtype)
    grad_b = gb_part.sum(dim=1).to(b.dtype)
    grad_c = gc_part.sum(dim=1).to(c.dtype)
    grad_d = gd_part.sum(dim=0).to(d_skip.dtype)
    grad_w = gw_part.sum(dim=0).unsqueeze(1).to(conv_weight.dtype)
    grad_cb = gcb_part.sum(dim=0).to(conv_bias.dtype)
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
    """Max-envelope resource metadata over all four kernels' specialisations."""
    merged: dict[str, int] | None = None
    for jit_fn in (_fwd_stage_kernel, _norm_bwd_kernel, _bwd_sweep_kernel, _conv_x_bwd_kernel):
        meta = collect_resource_meta(jit_fn)
        if meta is None:
            continue
        if merged is None:
            merged = dict(meta)
        else:
            for key, val in meta.items():
                merged[key] = max(merged.get(key, 0), val)
    return merged
