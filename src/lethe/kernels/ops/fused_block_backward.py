"""C6, hand-written Mamba fused-block backward (the training bottleneck)."""

from __future__ import annotations

import torch
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.references.fused_block_backward import FusedBlockGrads

from ._resource_meta import max_resource_meta
from .forward_chunked_scan import (
    _TRITON_DTYPES,
    _auto_chunk_len,
    _resolve_scan_mode,
    _triton_usable,
)
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
    """Gradients via autograd through the eager forward, correct by construction."""
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
    """Mamba fused-block backward: all nine gradients of the C5 forward."""
    conv_k = conv_weight.shape[-1]
    if conv_k != conv_kernel_size:
        raise ValueError(
            f"conv_kernel_size={conv_kernel_size} disagrees with conv_weight K={conv_k}"
        )
    if x.shape[-1] != conv_weight.shape[0]:
        raise ValueError(
            f"channel mismatch: x has {x.shape[-1]} channels, "
            f"conv_weight has {conv_weight.shape[0]}"
        )
    l_out = x.shape[1] - (conv_k - 1)
    if l_out % chunk_size != 0:
        raise ValueError(f"output length {l_out} must be divisible by chunk_size {chunk_size}")
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        batch, _, width = x.shape
        n_state = A.shape[1]
        if (
            _resolve_scan_mode(
                config, l_out, batch, width, is_forward=False, n_state=n_state, device=x.device
            )
            == "chunk_parallel"
        ):
            from lethe.kernels.ops import _triton_chunk_parallel_fused_bwd

            k = _auto_chunk_len(l_out, config.chunk_len if config is not None else None)
            return FusedBlockGrads(
                *_triton_chunk_parallel_fused_bwd.launch_fused_block_backward_chunk_parallel(
                    x,
                    conv_weight,
                    conv_bias,
                    delta,
                    A,
                    B,
                    C,
                    D,
                    norm_weight,
                    dy,
                    eps,
                    chunk_len=k,
                    config=config,
                )
            )
        from lethe.kernels.ops import _triton_fused_block_bwd

        return FusedBlockGrads(
            *_triton_fused_block_bwd.launch_fused_block_backward(
                x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy, eps, config=config
            )
        )
    return _fused_bwd_eager(x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, dy, eps)


def triton_fused_block_bwd_resource_meta(
    config: KernelConfig | None = None,
) -> dict[str, int] | None:
    """Resource metadata over the compiled backward kernels, if any."""
    if not _triton_usable():
        return None
    from lethe.kernels.ops import (
        _triton_chunk_parallel_fused_bwd,
        _triton_fused_block_bwd,
    )

    mode = config.scan_mode if config is not None else None
    if mode == "chunk_parallel":
        return _triton_chunk_parallel_fused_bwd.resource_meta()
    if mode == "serial":
        return _triton_fused_block_bwd.resource_meta()
    return max_resource_meta(
        _triton_fused_block_bwd.resource_meta(),
        _triton_chunk_parallel_fused_bwd.resource_meta(),
    )
