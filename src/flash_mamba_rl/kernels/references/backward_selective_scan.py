"""Mamba-3 selective scan backward pass via torch.autograd.

Reference oracle for the verifier. Gradients are computed by running
torch.autograd.grad through the chunked forward reference — no hand-derived
gradient formulae. This is intentional: the reference must be correct by
construction; analytical gradient derivation belongs in the optimised Triton
kernel, not here.

Reference: "Mamba-3" (Tri Dao + Albert Gu, ICLR 2026); underlying scan math
from the original Mamba paper (Gu & Dao, ICLR 2024, arXiv:2312.00752).
"""

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
    """Selective scan backward pass delegated entirely to torch.autograd.

    Wraps ``reference_forward_chunked_scan`` with ``requires_grad=True`` leaves,
    calls ``torch.autograd.grad`` with the upstream gradient ``dy``, and returns
    the six input gradients as a named tuple.

    Args:
        u:          Input tensor, shape [B, L, D], float32.
        delta:      Timescale input (pre-softplus), shape [B, L, D], float32.
        A:          Log-magnitude SSM matrix, shape [D, N], float32, negative.
        B:          Input projection, shape [B, L, N], float32.
        C:          Output projection, shape [B, L, N], float32.
        D:          Skip connection weight, shape [D], float32.
        dy:         Upstream gradient w.r.t. y, shape [B, L, D], float32.
        chunk_size: Forwarded to the scan; must divide L evenly.

    Returns:
        SelectiveScanGrads named tuple with fields:
        ``grad_u``, ``grad_delta``, ``grad_A``, ``grad_B``, ``grad_C``,
        ``grad_D``, each matching the shape of the corresponding input.
    """
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
