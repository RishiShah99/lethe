"""C3 — hand-written MIMO selective-scan backward (Mamba-3, Eqs. 12-14).

Drop-in for ``reference_mimo_backward`` with the same signature and
semantics, widened to more dtypes and devices:

- CUDA + {fp32, fp16, bf16} with triton installed -> the Triton kernel
  (``_triton_mimo_bwd``): analytic adjoint, no ``tl.dot``, no atomics.
- everything else (CPU, fp64, missing triton) -> autograd through
  ``reference_mimo_forward``. For fp32/fp64 the leaves feed the reference
  directly — gradients bitwise-equal to the oracle. The reference rejects
  half dtypes, so half inputs upcast to fp32 at the leaves and each
  gradient rounds once at the end (the mixed-precision contract). When
  ``dy.requires_grad`` the eager path builds the double-backward graph
  (``create_graph``) — the VJP is linear in ``dy``, which is what CMP-02's
  gradcheck differentiates.
"""

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
    """Mamba-3 MIMO SSM backward pass.

    Args/semantics mirror ``reference_mimo_backward``: ``x``/``dy``
    [B, L, H, P], ``B``/``C`` [B, L, R, H, N], ``dt``/``alpha`` [B, L, H],
    ``mimo_x``/``mimo_o`` [H, R, P]; returns a ``MimoGrads`` named tuple
    whose fields match the corresponding input shapes and dtypes.
    """
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        from lethe.kernels.ops import _triton_mimo_bwd

        return MimoGrads(
            *_triton_mimo_bwd.launch_mimo_backward(
                x, B, C, dt, alpha, mimo_x, mimo_o, dy, config=config
            )
        )
    return _mimo_bwd_eager(x, B, C, dt, alpha, mimo_x, mimo_o, dy)


def triton_mimo_bwd_resource_meta() -> dict[str, int] | None:
    """Resource metadata of the compiled Triton MIMO backward kernel, if any.

    Feed the result to ``gate_res_02_resource_limits`` via the harness;
    None (nothing compiled / no triton) keeps the gate not-applicable.
    """
    if not _triton_usable():
        return None
    from lethe.kernels.ops import _triton_mimo_bwd

    return _triton_mimo_bwd.resource_meta()
