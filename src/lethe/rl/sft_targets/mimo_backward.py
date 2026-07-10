"""Mamba-3 MIMO selective-scan backward pass via autograd (eager PyTorch)."""

import torch
from torch import Tensor


def _mimo_forward(
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
        h_agg_bc = h_agg.unsqueeze(1)
        y_raw = (h_agg_bc * C_t).sum(-1)

        mimo_o_bc = mimo_o.permute(1, 0, 2).unsqueeze(0)
        y[:, t, :, :] = (y_raw * mimo_o_bc).sum(1)

    return y


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
    out_dtype = x.dtype
    if out_dtype in (torch.float16, torch.bfloat16):
        x, B, C, dt, alpha, mimo_x, mimo_o, dy = (
            t.to(torch.float32) for t in (x, B, C, dt, alpha, mimo_x, mimo_o, dy)
        )

    x_l = x.detach().requires_grad_(True)
    b_l = B.detach().requires_grad_(True)
    c_l = C.detach().requires_grad_(True)
    dt_l = dt.detach().requires_grad_(True)
    alpha_l = alpha.detach().requires_grad_(True)
    mx_l = mimo_x.detach().requires_grad_(True)
    mo_l = mimo_o.detach().requires_grad_(True)

    y = _mimo_forward(x_l, b_l, c_l, dt_l, alpha_l, mx_l, mo_l)
    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x_l, b_l, c_l, dt_l, alpha_l, mx_l, mo_l),
        grad_outputs=dy,
    )
    grad_x, grad_b, grad_c, grad_dt, grad_alpha, grad_mx, grad_mo = grads
    return (
        grad_x.to(out_dtype),
        grad_b.to(out_dtype),
        grad_c.to(out_dtype),
        grad_dt.to(out_dtype),
        grad_alpha.to(out_dtype),
        grad_mx.to(out_dtype),
        grad_mo.to(out_dtype),
    )
