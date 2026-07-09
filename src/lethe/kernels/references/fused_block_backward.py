"""Mamba-3 fused-block backward pass via torch.autograd.

Reference oracle: gradients are obtained by differentiating through
``reference_fused_block_forward`` using torch.autograd.grad.  No hand-derived
formulae; correctness is guaranteed by PyTorch's autograd engine.

Reference: Mamba-3 (Lahoti, Li, et al., ICLR 2026).
"""

from typing import NamedTuple

import torch
from torch import Tensor

from .fused_block_forward import reference_fused_block_forward


class FusedBlockGrads(NamedTuple):
    """Gradient bundle returned by the fused-block backward reference."""

    grad_x: Tensor
    grad_conv_weight: Tensor
    grad_conv_bias: Tensor
    grad_delta: Tensor
    grad_A: Tensor
    grad_B: Tensor
    grad_C: Tensor
    grad_D: Tensor
    grad_norm_weight: Tensor


def reference_fused_block_backward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    dy: Tensor,
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
) -> FusedBlockGrads:
    """Fused-block backward pass via autograd through the forward reference.

    Creates fresh leaf tensors from all inputs, re-runs
    ``reference_fused_block_forward``, then calls ``torch.autograd.grad``
    with upstream gradient ``dy``.

    Args:
        x:               Input (possibly pre-padded), shape [B, L, D], float32.
        conv_weight:     Depthwise conv kernel, shape [D, 1, conv_kernel_size].
        conv_bias:       Depthwise conv bias, shape [D], float32.
        delta:           Timescale input, shape [B, L_out, D], float32.
        A:               Log-neg SSM matrix, shape [D, N], float32.
        B:               Input projection, shape [B, L_out, N], float32.
        C:               Output projection, shape [B, L_out, N], float32.
        D:               Skip weight, shape [D], float32.
        norm_weight:     RMSNorm gain, shape [D], float32.
        dy:              Upstream gradient, shape [B, L_out, D], float32.
        conv_kernel_size: Kernel size forwarded to the forward reference.
        eps:             RMSNorm epsilon forwarded to the forward reference.
        chunk_size:      Chunk size forwarded to the scan.

    Returns:
        FusedBlockGrads named tuple with fields:
        ``grad_x``, ``grad_conv_weight``, ``grad_conv_bias``,
        ``grad_delta``, ``grad_A``, ``grad_B``, ``grad_C``,
        ``grad_D``, ``grad_norm_weight``.

    Raises:
        ValueError: If any input tensor is not float32.
    """
    if x.dtype != torch.float32:
        raise ValueError(f"Expected float32, got x.dtype={x.dtype}")

    # Detach and create fresh leaf tensors.
    x_l = x.detach().requires_grad_(True)
    cw_l = conv_weight.detach().requires_grad_(True)
    cb_l = conv_bias.detach().requires_grad_(True)
    delta_l = delta.detach().requires_grad_(True)
    a_l = A.detach().requires_grad_(True)
    b_l = B.detach().requires_grad_(True)
    c_l = C.detach().requires_grad_(True)
    d_l = D.detach().requires_grad_(True)
    nw_l = norm_weight.detach().requires_grad_(True)

    y = reference_fused_block_forward(
        x_l,
        cw_l,
        cb_l,
        delta_l,
        a_l,
        b_l,
        c_l,
        d_l,
        nw_l,
        conv_kernel_size=conv_kernel_size,
        eps=eps,
        chunk_size=chunk_size,
    )

    grads = torch.autograd.grad(
        outputs=y,
        inputs=(x_l, cw_l, cb_l, delta_l, a_l, b_l, c_l, d_l, nw_l),
        grad_outputs=dy,
    )

    return FusedBlockGrads(
        grad_x=grads[0],
        grad_conv_weight=grads[1],
        grad_conv_bias=grads[2],
        grad_delta=grads[3],
        grad_A=grads[4],
        grad_B=grads[5],
        grad_C=grads[6],
        grad_D=grads[7],
        grad_norm_weight=grads[8],
    )
