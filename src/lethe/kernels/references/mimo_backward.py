"""Mamba-3 MIMO selective scan: forward reference and autograd-based backward."""

from typing import NamedTuple

import torch
from torch import Tensor


def reference_mimo_forward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
) -> Tensor:
    """Mamba-3 MIMO SSM forward pass (Eqs 12-14)."""
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    batch, seqlen, nheads, headdim = x.shape
    R = B.shape[2]
    d_state = B.shape[4]

    # Step 1: Expand x to rank dimension (mimo_x applied BEFORE B projection).
    mimo_x_bc = mimo_x.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)  # (1, 1, R, H, P)
    x_r = x.unsqueeze(2) * mimo_x_bc  # (B, L, R, H, P)

    # Per-rank hidden state: shape (batch, R, nheads, headdim, d_state)
    h = torch.zeros(batch, R, nheads, headdim, d_state, dtype=x.dtype, device=x.device)
    y = torch.empty_like(x)

    for t in range(seqlen):
        # alpha_t: (batch, nheads) -> (batch, 1, nheads, 1, 1)
        alpha_t = alpha[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # dt_t: (batch, nheads) -> (batch, 1, nheads, 1, 1)
        dt_t = dt[:, t, :].unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

        # B_t: (batch, R, nheads, d_state) -> (batch, R, nheads, 1, d_state)
        B_t = B[:, t, :, :, :].unsqueeze(3)  # (B, R, H, 1, N)

        # x_r_t: (batch, R, nheads, headdim) -> (batch, R, nheads, headdim, 1)
        x_r_t = x_r[:, t, :, :, :].unsqueeze(-1)  # (B, R, H, P, 1)

        # Eq. 12 (per rank j): h_t = alpha_t*h_{t-1} + dt_t*B_t*x_r_t, shape (B,R,H,P,N)
        h = alpha_t * h + dt_t * B_t * x_r_t  # (B, R, H, P, N)

        # Eq. 13: h_agg = sum_{j} h^(j)
        h_agg = h.sum(dim=1)  # (B, H, P, N)

        # Eq. 14: y_raw^(i) = C^(i)^T @ h_agg; C_t is (batch, R, nheads, 1, d_state)
        C_t = C[:, t, :, :, :].unsqueeze(3)  # (B, R, H, 1, N)
        # h_agg broadcast: (B, H, P, N) -> (B, 1, H, P, N)
        h_agg_bc = h_agg.unsqueeze(1)  # (B, 1, H, P, N)
        # dot over d_state: (B, R, H, P)
        y_raw = (h_agg_bc * C_t).sum(-1)  # (B, R, H, P)

        # Step 5: y_t = sum_i y_raw^(i) * mimo_o[h, i, :]; mimo_o reshaped to (1, R, H, P)
        mimo_o_bc = mimo_o.permute(1, 0, 2).unsqueeze(0)  # (1, R, H, P)
        # sum over R: (B, H, P)
        y_t = (y_raw * mimo_o_bc).sum(1)  # (B, H, P)
        y[:, t, :, :] = y_t  # (B, H, P)

    return y


class MimoGrads(NamedTuple):
    """Gradient bundle returned by the MIMO backward reference."""

    grad_x: Tensor
    grad_B: Tensor
    grad_C: Tensor
    grad_dt: Tensor
    grad_alpha: Tensor
    grad_mimo_x: Tensor
    grad_mimo_o: Tensor


def reference_mimo_backward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
) -> MimoGrads:
    """Mamba-3 MIMO SSM backward pass delegated entirely to torch.autograd."""
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64 input, got {x.dtype}")

    # Detach from any existing graph and create fresh leaf tensors.
    x_l = x.detach().requires_grad_(True)
    B_l = B.detach().requires_grad_(True)
    C_l = C.detach().requires_grad_(True)
    dt_l = dt.detach().requires_grad_(True)
    alpha_l = alpha.detach().requires_grad_(True)
    mimo_x_l = mimo_x.detach().requires_grad_(True)
    mimo_o_l = mimo_o.detach().requires_grad_(True)

    y = reference_mimo_forward(x_l, B_l, C_l, dt_l, alpha_l, mimo_x_l, mimo_o_l)

    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x_l, B_l, C_l, dt_l, alpha_l, mimo_x_l, mimo_o_l),
        grad_outputs=dy,
    )

    return MimoGrads(
        grad_x=grads[0],
        grad_B=grads[1],
        grad_C=grads[2],
        grad_dt=grads[3],
        grad_alpha=grads[4],
        grad_mimo_x=grads[5],
        grad_mimo_o=grads[6],
    )
