"""Hand-written Triton kernels (Phase C), one module per op.

Each op mirrors its reference oracle's signature exactly and dispatches:
CUDA + supported dtype -> the Triton kernel; everything else -> a
differentiable eager-torch path that replicates the reference op-for-op.
"""

from .backward_selective_scan import backward_selective_scan, triton_bwd_scan_resource_meta
from .complex_scan_rope import complex_scan_rope, triton_complex_rope_resource_meta
from .forward_chunked_scan import forward_chunked_scan, triton_scan_resource_meta
from .fused_block_backward import fused_block_backward, triton_fused_block_bwd_resource_meta
from .fused_block_forward import fused_block_forward, triton_fused_block_resource_meta
from .mimo_backward import mimo_backward, triton_mimo_bwd_resource_meta

__all__ = [
    "backward_selective_scan",
    "complex_scan_rope",
    "forward_chunked_scan",
    "fused_block_backward",
    "fused_block_forward",
    "mimo_backward",
    "triton_bwd_scan_resource_meta",
    "triton_complex_rope_resource_meta",
    "triton_fused_block_bwd_resource_meta",
    "triton_fused_block_resource_meta",
    "triton_mimo_bwd_resource_meta",
    "triton_scan_resource_meta",
]
