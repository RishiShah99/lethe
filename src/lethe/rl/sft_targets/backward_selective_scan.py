"""Mamba SISO selective-scan backward pass via autograd (eager PyTorch)."""

import torch
import torch.nn.functional as F
from torch import Tensor


def _scan_forward(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    batch, seq_len, d_model = u.shape
    n_state = A.shape[1]

    delta_bar = F.softplus(delta)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * B.unsqueeze(2)

    y = torch.empty_like(u)
    h = torch.zeros(batch, d_model, n_state, dtype=u.dtype, device=u.device)
    for t in range(seq_len):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * u[:, t, :].unsqueeze(-1)
        y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(-1) + D * u[:, t, :]
    return y


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
    out_dtype = u.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        u, delta, A, B, C, D, dy = (t.to(torch.float32) for t in (u, delta, A, B, C, D, dy))

    # Same divisibility contract as the reference: don't silently accept non-dividing seq_len.
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")

    u_l = u.detach().requires_grad_(True)
    delta_l = delta.detach().requires_grad_(True)
    a_l = A.detach().requires_grad_(True)
    b_l = B.detach().requires_grad_(True)
    c_l = C.detach().requires_grad_(True)
    d_l = D.detach().requires_grad_(True)

    y = _scan_forward(u_l, delta_l, a_l, b_l, c_l, d_l)
    grads = torch.autograd.grad(
        outputs=y,
        inputs=(u_l, delta_l, a_l, b_l, c_l, d_l),
        grad_outputs=dy,
    )
    grad_u, grad_delta, grad_a, grad_b, grad_c, grad_d = grads
    return (
        grad_u.to(out_dtype),
        grad_delta.to(out_dtype),
        grad_a.to(out_dtype),
        grad_b.to(out_dtype),
        grad_c.to(out_dtype),
        grad_d.to(out_dtype),
    )
