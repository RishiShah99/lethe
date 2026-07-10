"""C4, hand-written Mamba-3 complex-RoPE selective scan forward."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.references.complex_scan_rope import reference_complex_scan_rope

from .forward_chunked_scan import _TRITON_DTYPES, _triton_usable


def _rope_eager(
    x: Tensor, B: Tensor, C: Tensor, dt: Tensor, A: Tensor, angle_proj: Tensor
) -> Tensor:
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inputs = (x, B, C, dt, A, angle_proj)
    y = reference_complex_scan_rope(*(t.to(compute_dtype) for t in inputs))
    return y.to(x.dtype)


class _ComplexRopeCuda(torch.autograd.Function):
    """Triton forward; backward recomputes the VJP through the eager path."""

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        B: Tensor,
        C: Tensor,
        dt: Tensor,
        A: Tensor,
        angle_proj: Tensor,
        config: KernelConfig | None = None,
    ) -> Tensor:
        from lethe.kernels.ops import _triton_complex_rope

        ctx.save_for_backward(x, B, C, dt, A, angle_proj)
        return _triton_complex_rope.launch_complex_scan_rope(
            x, B, C, dt, A, angle_proj, config=config
        )

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        inputs = ctx.saved_tensors
        with torch.enable_grad():
            leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
            y = _rope_eager(*leaves)
            grads = torch.autograd.grad(y, leaves, grad_y)
        return (*grads, None)


def complex_scan_rope(
    x: Tensor,
    B: Tensor,
    C: Tensor,
    dt: Tensor,
    A: Tensor,
    angle_proj: Tensor,
    *,
    config: KernelConfig | None = None,
) -> Tensor:
    """Mamba-3 SSM forward with data-dependent RoPE rotation (paper §3.2)."""
    if 2 * angle_proj.shape[-1] > B.shape[-1]:
        raise ValueError(f"rotary_dim={2 * angle_proj.shape[-1]} exceeds d_state={B.shape[-1]}")
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        return cast(
            Tensor,
            _ComplexRopeCuda.apply(x, B, C, dt, A, angle_proj, config),  # type: ignore[no-untyped-call]
        )
    return _rope_eager(x, B, C, dt, A, angle_proj)


def triton_complex_rope_resource_meta() -> dict[str, int] | None:
    """Resource metadata of the compiled Triton rotary-scan kernel, if any."""
    if not _triton_usable():
        return None
    from lethe.kernels.ops import _triton_complex_rope

    return _triton_complex_rope.resource_meta()
