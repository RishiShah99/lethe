"""C2, hand-written SISO selective-scan backward (the #904-broken op)."""

from __future__ import annotations

import torch
from torch import Tensor

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.references.backward_selective_scan import SelectiveScanGrads

from ._resource_meta import max_resource_meta
from .forward_chunked_scan import (
    _TRITON_DTYPES,
    _auto_chunk_len,
    _resolve_scan_mode,
    _scan_eager,
    _triton_usable,
)


def _bwd_eager(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, dy: Tensor
) -> SelectiveScanGrads:
    """Gradients via autograd through the eager forward, correct by construction."""
    create_graph = dy.requires_grad
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in (u, delta, A, B, C, D)]
        y = _scan_eager(*leaves)
        grads = torch.autograd.grad(y, leaves, dy, create_graph=create_graph)
    return SelectiveScanGrads(*grads)


def backward_selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    chunk_size: int = 64,
    config: KernelConfig | None = None,
) -> SelectiveScanGrads:
    """Selective-scan backward pass (SISO, Mamba-1 recurrence)."""
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    if u.is_cuda and u.dtype in _TRITON_DTYPES and _triton_usable():
        batch, _, width = u.shape
        n_state = A.shape[1]
        if (
            _resolve_scan_mode(
                config, seq_len, batch, width, is_forward=False, n_state=n_state, device=u.device
            )
            == "chunk_parallel"
        ):
            from lethe.kernels.ops import _triton_chunk_parallel_bwd

            k = _auto_chunk_len(seq_len, config.chunk_len if config is not None else None)
            return SelectiveScanGrads(
                *_triton_chunk_parallel_bwd.launch_backward_chunk_parallel_scan(
                    u, delta, A, B, C, D, dy, chunk_len=k, config=config
                )
            )
        from lethe.kernels.ops import _triton_bwd_scan

        return SelectiveScanGrads(
            *_triton_bwd_scan.launch_backward_scan(u, delta, A, B, C, D, dy, config=config)
        )
    return _bwd_eager(u, delta, A, B, C, D, dy)


def triton_bwd_scan_resource_meta(config: KernelConfig | None = None) -> dict[str, int] | None:
    """Resource metadata of the compiled Triton backward kernel, if any."""
    if not _triton_usable():
        return None
    from lethe.kernels.ops import _triton_bwd_scan, _triton_chunk_parallel_bwd

    mode = config.scan_mode if config is not None else None
    if mode == "chunk_parallel":
        return _triton_chunk_parallel_bwd.resource_meta()
    if mode == "serial":
        return _triton_bwd_scan.resource_meta()
    return max_resource_meta(
        _triton_bwd_scan.resource_meta(), _triton_chunk_parallel_bwd.resource_meta()
    )
