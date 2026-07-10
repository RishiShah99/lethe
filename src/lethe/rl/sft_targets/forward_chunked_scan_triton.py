"""Triton SISO selective-scan forward."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

try:
    from triton.language.extra import libdevice
except ImportError:
    from triton.language.extra.cuda import libdevice

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)

MAX_BLOCK_N = 128


@triton.jit  # type: ignore[untyped-decorator]
def _fwd_scan_kernel(  # type: ignore[no-untyped-def]
    u_ptr,
    delta_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    dskip_ptr,
    y_ptr,
    seq_len,
    d_model,
    n_state,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    mask_d = offs_d < d_model
    mask_n = offs_n < n_state
    mask_dn = mask_d[:, None] & mask_n[None, :]

    a = tl.load(a_ptr + offs_d[:, None] * n_state + offs_n[None, :], mask=mask_dn, other=0.0).to(
        tl.float32
    )
    d_skip = tl.load(dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    h = tl.zeros((BLOCK_D, BLOCK_N), dtype=tl.float32)

    # int64 running offsets: b * L * D overflows int32 at large batched runs.
    uld_off = pid_b.to(tl.int64) * seq_len * d_model + offs_d
    bln_off = pid_b.to(tl.int64) * seq_len * n_state + offs_n

    for _t in range(seq_len):
        u_t = tl.load(u_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)
        dlt = tl.load(delta_ptr + uld_off, mask=mask_d, other=0.0).to(tl.float32)

        b_t = tl.load(b_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)
        c_t = tl.load(c_ptr + bln_off, mask=mask_n, other=0.0).to(tl.float32)

        # libdevice.exp keeps subnormals; tl.exp's ex2.approx flush splits NaN/Inf masks vs reference.
        dbar = tl.where(dlt > _SOFTPLUS_THRESHOLD, dlt, libdevice.log1p(libdevice.exp(dlt)))

        a_bar = libdevice.exp(dbar[:, None] * a)
        bu = (dbar * u_t)[:, None] * b_t[None, :]
        # Padding lanes stay zero: non-finite u/delta could mint Inf*0=NaN and poison the N-reduction.
        h = tl.where(mask_dn, a_bar * h + bu, 0.0)

        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * u_t
        tl.store(y_ptr + uld_off, y_t.to(y_ptr.dtype.element_ty), mask=mask_d)

        uld_off += d_model
        bln_off += n_state


def _forward_eager(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    out_dtype = u.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        u, delta, A, B, C, D = (t.to(torch.float32) for t in (u, delta, A, B, C, D))
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
    return y.to(out_dtype)


def forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    # Device residency: non-CUDA (and fp64) inputs take the eager path.
    if not (u.is_cuda and u.dtype in (torch.float32, torch.float16, torch.bfloat16)):
        return _forward_eager(u, delta, A, B, C, D)
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    block_n = triton.next_power_of_2(n_state)
    if block_n > MAX_BLOCK_N:
        raise ValueError(f"n_state={n_state} exceeds single-block budget {MAX_BLOCK_N}")
    block_d = min(64, triton.next_power_of_2(d_model))

    u_c = u.contiguous()
    delta_c = delta.contiguous()
    a_c = A.contiguous()
    b_c = B.contiguous()
    c_c = C.contiguous()
    d_c = D.contiguous()
    y = torch.empty_like(u_c)

    grid = (batch, triton.cdiv(d_model, block_d))
    num_warps = 4 if block_d * block_n >= 512 else 2
    _fwd_scan_kernel[grid](
        u_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        d_c,
        y,
        seq_len,
        d_model,
        n_state,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return y
