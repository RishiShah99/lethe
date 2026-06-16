"""C6 — hand-written Mamba fused-block backward (the training bottleneck).

Drop-in for ``reference_fused_block_backward`` with the same signature and
semantics, widened to more dtypes and devices:

- CUDA + {fp32, fp16, bf16} with triton installed -> the four-kernel
  Triton pipeline (``_triton_fused_block_bwd``: forward re-stage with
  checkpoints, RMSNorm backward, chunked reverse sweep with conv/SiLU
  recompute, gather-form conv input-gradient — the structural rationale
  lives on that module).
- everything else (CPU, fp64, missing triton) -> ``torch.autograd.grad``
  through ``_fused_eager``, the differentiable eager path shared with the
  C5 forward op. For fp32 on CPU this replicates the reference oracle's
  gradients; fp64 is the verification dtype (gradcheck) and deliberately
  routes here. When ``dy.requires_grad``, the eager path builds the
  double-backward graph (``create_graph=True``) — the VJP is linear in
  ``dy``, which is exactly what CMP-02's gradcheck differentiates.

Deviations from the reference: the reference rejects non-fp32 inputs; this
op defines the mixed-precision contract instead (compute in fp32, round
once per gradient output). ``chunk_size`` is validated identically but is
a blocking hint only — gradients do not depend on it, and the Triton
pipeline picks its own recompute-chunk internally.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.references.fused_block_backward import FusedBlockGrads

from .forward_chunked_scan import _TRITON_DTYPES, _triton_usable
from .fused_block_forward import _fused_eager


def _fused_bwd_eager(
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
    eps: float,
) -> FusedBlockGrads:
    """Gradients via autograd through the eager forward — correct by construction."""
    create_graph = dy.requires_grad
    inputs = (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in inputs]
        y = _fused_eager(*leaves, eps=eps)
        grads = torch.autograd.grad(y, leaves, dy, create_graph=create_graph)
    return FusedBlockGrads(*grads)


def fused_block_backward(
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
    config: KernelConfig | None = None,
) -> FusedBlockGrads:
    """Mamba fused-block backward: all nine gradients of the C5 forward.

    Args/semantics mirror ``reference_fused_block_backward``: forward
    inputs as in ``fused_block_forward`` plus the upstream gradient ``dy``
    [B, L_out, D]; returns a ``FusedBlockGrads`` named tuple whose fields
    match the corresponding input shapes and dtypes. All floating inputs
    share ``x``'s dtype (dispatch keys on ``x`` alone, the family
    contract).

    Raises:
        ValueError: If ``conv_kernel_size`` disagrees with ``conv_weight``'s
            trailing dim, or if L_out is not divisible by ``chunk_size``.
    """
    conv_k = conv_weight.shape[-1]
    if conv_k != conv_kernel_size:
        raise ValueError(
            f"conv_kernel_size={conv_kernel_size} disagrees with conv_weight K={conv_k}"
        )
    l_out = x.shape[1] - (conv_k - 1)
    if l_out % chunk_size != 0:
        raise ValueError(f"output length {l_out} must be divisible by chunk_size {chunk_size}")
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        from flash_mamba_rl.kernels.ops import _triton_fused_block_bwd

        return FusedBlockGrads(
            *_triton_fused_block_bwd.launch_fused_block_backward(
                x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy, eps, config=config
            )
        )
    return _fused_bwd_eager(x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy, eps)


def triton_fused_block_bwd_resource_meta() -> dict[str, int] | None:
    """Resource metadata over the four compiled backward kernels, if any.

    Feed the result to ``gate_res_02_resource_limits`` via the harness;
    None (nothing compiled / no triton) keeps the gate not-applicable.
    """
    if not _triton_usable():
        return None
    from flash_mamba_rl.kernels.ops import _triton_fused_block_bwd

    return _triton_fused_block_bwd.resource_meta()
