"""Triton kernel for the SISO selective-scan backward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

try:
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - depends on installed triton
    from triton.language.extra.cuda import libdevice

# torch.nn.functional.softplus switches to the identity above this threshold.
_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)

# One CTA holds the whole state dim; N above this needs a multi-block design.
MAX_BLOCK_N = 128

# Recompute-chunk cap.
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

    # Scratch tiles need no mask here: padding lanes are already zero from prior masked stores.
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
        # Promote c to int64 before the stride product; it can cross 2^31 past an 8 GiB ckpt region.
        tl.store(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N, h)  # type: ignore[attr-defined]
        for _j in range(CHUNK_K):
            u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
            dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
            b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

            dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))
            abar = libdevice.exp(dbar[:, None] * a)
            bb = dbar[:, None] * b_t[None, :]
            # Padding lanes must stay zero (C1 lesson) or a stray NaN poisons the real N-reductions.
            h = tl.where(mask_dn, abar * h + bb * u_t[:, None], 0.0)

            uld_off += d_model
            bln_off += n_state

    # ----- Phase 2: chunks newest-first, recompute states, then reverse-sweep for gradients.
    ag_carry = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    ga_acc = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for ci in range(n_chunks):
        c = n_chunks - 1 - ci
        t0 = c * CHUNK_K

        # Recompute from the checkpoint, storing each step's pre-update h_{t-1} for the reverse sweep.
        h_prev = tl.load(ckpt_prog + c.to(tl.int64) * BLOCK_D * BLOCK_N)
        # Fold t0 into int64 before the row-stride multiply; int32 t0*d_model overflows past L*D~2^31 (C1).
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

            # g_t = dL/dh_t.
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

            # grad_delta_bar: two N-reductions (via exp, via b_bar); the shared gm also feeds grad_A.
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
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Launch the Triton backward scan."""
    batch, seq_len, d_model = u.shape
    n_state = a.shape[1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))
    chunk_k = _chunk_k(seq_len)
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
    # Partials/scratch stay fp32; input dtype is rounded once after reduction (PRC contract).
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
    warps = num_warps if num_warps is not None else (4 if block_d * block_n >= 512 else 2)
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
    )

    # Deterministic reductions: fixed shapes give fixed reduction trees, byte-identical unlike atomics.
    grad_a = ga_part.sum(dim=0).to(a.dtype)
    grad_b = gb_part.sum(dim=1).to(b.dtype)
    grad_c = gc_part.sum(dim=1).to(c.dtype)
    grad_d = gd_part.sum(dim=0).to(d_skip.dtype)
    return grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d


def _scan_forward_eager(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor
) -> Tensor:
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]
    delta_bar = torch.nn.functional.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)
    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    for t in range(seq_len):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]
    return y


def _backward_eager(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, dy: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    out_dtype = u.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        u, delta, A, B, C, D, dy = (t.to(torch.float32) for t in (u, delta, A, B, C, D, dy))
    leaves = [t.detach().requires_grad_(True) for t in (u, delta, A, B, C, D)]
    y = _scan_forward_eager(*leaves)
    grads = torch.autograd.grad(outputs=y, inputs=leaves, grad_outputs=dy)
    g_u, g_delta, g_a, g_b, g_c, g_d = grads
    return (
        g_u.to(out_dtype),
        g_delta.to(out_dtype),
        g_a.to(out_dtype),
        g_b.to(out_dtype),
        g_c.to(out_dtype),
        g_d.to(out_dtype),
    )


def backward_selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    chunk_size: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Selective-scan backward pass (SISO, Mamba-1 recurrence)."""
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    # Device residency: non-CUDA (and fp64) inputs take the eager path.
    if not (u.is_cuda and u.dtype in (torch.float32, torch.float16, torch.bfloat16)):
        return _backward_eager(u, delta, A, B, C, D, dy)
    return launch_backward_scan(u, delta, A, B, C, D, dy)
