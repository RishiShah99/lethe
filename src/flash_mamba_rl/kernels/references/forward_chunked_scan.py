"""Mamba-1/3 SISO selective scan forward pass, computed chunk-by-chunk.

Reference oracle for the verifier. Implements the core SSM recurrence from:
  "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  Gu & Dao, ICLR 2024 (arXiv:2312.00752)

Used as the scan primitive inside Mamba-3 (Tri Dao + Albert Gu, ICLR 2026).
Correctness over speed: float32 throughout, plain Python loop per chunk.
"""

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
    """Selective state-space scan (SISO, Mamba-1 recurrence), chunked forward.

    Discretises (A, B) with the zero-order-hold (ZOH) rule:
        delta_bar = softplus(delta)            [B, L, D]
        A_bar     = exp(delta_bar * A)         [B, L, D, N]  (A is [D, N], log-neg)
        B_bar     = delta_bar[..., None] * B   [B, L, D, N]  (broadcast)

    Recurrence per step t (state h is [B, D, N]):
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t[..., None]
        y_t = (h_t * C_t).sum(-1) + D * u_t

    Chunks are processed sequentially; within each chunk the recurrence runs
    step-by-step in Python (no parallelism — this is the oracle, not the kernel).

    Args:
        u:          Input tensor, shape [B, L, D], float32.
        delta:      Timescale input (pre-softplus), shape [B, L, D], float32.
        A:          Log-magnitude SSM matrix, shape [D, N], float32, negative.
        B:          Input projection, shape [B, L, N], float32.
        C:          Output projection, shape [B, L, N], float32.
        D:          Skip connection weight, shape [D], float32.
        chunk_size: Number of time-steps per chunk (must divide L evenly).

    Returns:
        y: Output tensor, same shape as u, [B, L, D], float32.

    Raises:
        ValueError: If L is not divisible by chunk_size.
    """
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    if seq_len % chunk_size != 0:
        raise ValueError(
            f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}"
        )

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
            # y_t = sum_n( h * C_t ) + D * u_t
            # C: [B, L, N] -> C[:, t, :] is [B, N]
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]

    return y
