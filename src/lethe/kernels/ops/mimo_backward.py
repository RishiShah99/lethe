"""C3, hand-written MIMO selective-scan backward (Mamba-3, Eqs. 12-14)."""

from __future__ import annotations

import torch
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.references.mimo_backward import MimoGrads, reference_mimo_forward

from .forward_chunked_scan import _TRITON_DTYPES, _triton_usable


def _mimo_bwd_eager(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
) -> MimoGrads:
    """Gradients via autograd through the reference forward."""
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inputs = (x, B, C, dt, alpha, mimo_x, mimo_o)
    create_graph = dy.requires_grad
    with torch.enable_grad():
        leaves = [t.detach().to(compute_dtype).requires_grad_(True) for t in inputs]
        y = reference_mimo_forward(*leaves)
        grads = torch.autograd.grad(y, leaves, dy.to(compute_dtype), create_graph=create_graph)
    return MimoGrads(*(g.to(t.dtype) for g, t in zip(grads, inputs, strict=True)))


def mimo_backward(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    alpha: Tensor,
    mimo_x: Tensor,
    mimo_o: Tensor,
    dy: Tensor,
    *,
    config: KernelConfig | None = None,
) -> MimoGrads:
    """Mamba-3 MIMO SSM backward pass."""
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        from lethe.kernels.ops import _triton_mimo_bwd

        return MimoGrads(
            *_triton_mimo_bwd.launch_mimo_backward(
                x, B, C, dt, alpha, mimo_x, mimo_o, dy, config=config
            )
        )
    return _mimo_bwd_eager(x, B, C, dt, alpha, mimo_x, mimo_o, dy)


def triton_mimo_bwd_resource_meta() -> dict[str, int] | None:
    """Resource metadata of the compiled Triton MIMO backward kernel, if any."""
    if not _triton_usable():
        return None
    from lethe.kernels.ops import _triton_mimo_bwd

    return _triton_mimo_bwd.resource_meta()
