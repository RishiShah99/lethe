"""Hand-written Triton kernels (Phase C), one module per op.

Each op mirrors its reference oracle's signature exactly and dispatches:
CUDA + supported dtype -> the Triton kernel; everything else -> a
differentiable eager-torch path that replicates the reference op-for-op.
"""

from .backward_selective_scan import backward_selective_scan, triton_bwd_scan_resource_meta
from .forward_chunked_scan import forward_chunked_scan, triton_scan_resource_meta

__all__ = [
    "backward_selective_scan",
    "forward_chunked_scan",
    "triton_bwd_scan_resource_meta",
    "triton_scan_resource_meta",
]
