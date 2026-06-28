"""PyTorch reference implementations — slow, correct, oracle for the verifier."""

from .backward_selective_scan import (
    SelectiveScanGrads,
    reference_backward_selective_scan,
)
from .complex_scan_rope import reference_complex_scan_rope
from .forward_chunked_scan import (
    reference_forward_chunked_scan,
    reference_forward_trapezoidal_scan,
)
from .fused_block_backward import FusedBlockGrads, reference_fused_block_backward
from .fused_block_forward import reference_fused_block_forward
from .gdn2_chunkwise import (
    ChunkwiseBackward,
    ChunkwiseForward,
    MicroGateBundle,
    build_microgate_bundles,
    chunkwise_backward,
    chunkwise_forward,
)
from .gdn_backward import Gdn2Grads, reference_gdn2_backward, reference_gdn2_forward
from .mimo_backward import MimoGrads, reference_mimo_backward, reference_mimo_forward

__all__ = [
    "ChunkwiseBackward",
    "ChunkwiseForward",
    "FusedBlockGrads",
    "Gdn2Grads",
    "MicroGateBundle",
    "MimoGrads",
    "SelectiveScanGrads",
    "build_microgate_bundles",
    "chunkwise_backward",
    "chunkwise_forward",
    "reference_backward_selective_scan",
    "reference_complex_scan_rope",
    "reference_forward_chunked_scan",
    "reference_forward_trapezoidal_scan",
    "reference_fused_block_backward",
    "reference_fused_block_forward",
    "reference_gdn2_backward",
    "reference_gdn2_forward",
    "reference_mimo_backward",
    "reference_mimo_forward",
]
