"""Mamba-1/3 SISO selective scan forward pass, computed chunk-by-chunk."""

import torch
import torch.nn.functional as F
from torch import Tensor


def reference_forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """Selective state-space scan (SISO, Mamba-1 recurrence), chunked forward."""
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")

    # softplus discretisation of delta  [B, L, D]
    delta_bar = F.softplus(delta)

    # A_bar: [B, L, D, N] = exp(delta_bar[..., None] * A[None, None, :, :])
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

    # B_bar: [B, L, D, N] = delta_bar[..., None] * B[:, :, None, :]
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    # Allocate output and running hidden state
    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)

    n_chunks = seq_len // chunk_size
    for chunk_idx in range(n_chunks):
        t0 = chunk_idx * chunk_size
        t1 = t0 + chunk_size
        for t in range(t0, t1):
            # h: [B, D, N]
            h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
            # y_t = sum_n(h * C_t) + D * u_t; C is [B, L, N], so C[:, t, :] is [B, N]
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]

    return y


def reference_forward_trapezoidal_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    trap: Tensor,
    *,
    chunk_size: int = 64,
) -> Tensor:
    """Mamba-3 exponential-trapezoidal SISO scan (paper Prop. 3.2.2)."""
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")

    delta_bar = F.softplus(delta)  # Delta_t  [B, L, D]
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
    # b_bar grouped exactly as the ZOH oracle so the lambda=1 limit is bit-identical:
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]  (= delta_bar * B)
    bu = B.unsqueeze(2) * u.unsqueeze(-1)  # [B, L, D, N]  raw B_t * u_t (previous-token term)
    lam = torch.sigmoid(trap)  # lambda_t  [B, L, D]
    beta = (1.0 - lam) * delta_bar  # [B, L, D]  (alpha_t folded in per-step below)

    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    bu_prev = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)

    n_chunks = seq_len // chunk_size
    for chunk_idx in range(n_chunks):
        t0 = chunk_idx * chunk_size
        t1 = t0 + chunk_size
        for t in range(t0, t1):
            # current: lambda_t * (delta_bar_t * B_t) * u_t  -> b_bar * u at lambda=1
            cur = lam[:, t, :].unsqueeze(-1) * b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
            # previous: beta_t * alpha_t * (B_{t-1} * u_{t-1})  -> exactly 0 at lambda=1
            prev = beta[:, t, :].unsqueeze(-1) * a_bar[:, t, :, :] * bu_prev
            h = a_bar[:, t, :, :] * h + cur + prev
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]
            bu_prev = bu[:, t, :, :]

    return y
