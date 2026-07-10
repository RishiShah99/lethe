"""Python entry point for the CUDA selective-scan backward."""

from __future__ import annotations

import torch
from torch import Tensor

from lethe.kernels.references.backward_selective_scan import SelectiveScanGrads

from ._loader import load_scan_extension


def cuda_backward_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    items: int = 4,
    block_d: int = 8,
) -> SelectiveScanGrads:
    """Selective-scan backward on the CUDA kernel."""
    if not u.is_cuda:
        raise ValueError("cuda_backward_scan requires CUDA tensors")
    if u.dtype != torch.float32:
        raise ValueError(f"Inc 3 backward is fp32-only, got {u.dtype}")
    ext = load_scan_extension()

    u_dl = u.transpose(1, 2).contiguous()
    delta_dl = delta.transpose(1, 2).contiguous()
    dy_dl = dy.transpose(1, 2).contiguous()
    b_nl = B.transpose(1, 2).contiguous()
    c_nl = C.transpose(1, 2).contiguous()

    gu_dl, gdl_dl, ga_part, gb_part, gc_part, gd_part = ext.backward_scan(
        u_dl, delta_dl, A.contiguous(), b_nl, c_nl, dy_dl, items, block_d
    )

    grad_u_dl = gu_dl + D.view(1, -1, 1) * dy_dl  # add the skip path D*dy
    grad_u = grad_u_dl.transpose(1, 2).contiguous()
    grad_delta = gdl_dl.transpose(1, 2).contiguous()
    grad_A = ga_part.sum(dim=0)
    grad_B = gb_part.sum(dim=1).transpose(1, 2).contiguous()
    grad_C = gc_part.sum(dim=1).transpose(1, 2).contiguous()
    grad_D = gd_part.sum(dim=0)
    return SelectiveScanGrads(grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D)
