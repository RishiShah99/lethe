"""Mamba-3 selective scan backward pass via torch.autograd."""

from typing import NamedTuple

import torch
from torch import Tensor

from .forward_chunked_scan import reference_forward_chunked_scan


class SelectiveScanGrads(NamedTuple):
    """Gradient bundle returned by the backward reference."""

    grad_u: Tensor
    grad_delta: Tensor
    grad_A: Tensor
    grad_B: Tensor
    grad_C: Tensor
    grad_D: Tensor


def reference_backward_selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    chunk_size: int = 64,
) -> SelectiveScanGrads:
    """Selective scan backward pass delegated entirely to torch.autograd."""
    if u.dtype != torch.float32:
        raise ValueError(f"Expected float32 input, got {u.dtype}")

    # Detach from any existing graph and create fresh leaf tensors.
    u_l = u.detach().requires_grad_(True)
    delta_l = delta.detach().requires_grad_(True)
    a_l = A.detach().requires_grad_(True)
    b_l = B.detach().requires_grad_(True)
    c_l = C.detach().requires_grad_(True)
    d_l = D.detach().requires_grad_(True)

    y = reference_forward_chunked_scan(u_l, delta_l, a_l, b_l, c_l, d_l, chunk_size=chunk_size)

    grads = torch.autograd.grad(
        outputs=y,
        inputs=(u_l, delta_l, a_l, b_l, c_l, d_l),
        grad_outputs=dy,
    )

    return SelectiveScanGrads(
        grad_u=grads[0],
        grad_delta=grads[1],
        grad_A=grads[2],
        grad_B=grads[3],
        grad_C=grads[4],
        grad_D=grads[5],
    )
