"""PyTorch reference implementations — slow, correct, oracle for the verifier."""

from .backward_selective_scan import (
    SelectiveScanGrads,
    reference_backward_selective_scan,
)
from .complex_scan_rope import reference_complex_scan_rope
from .family_oracles import (
    GlaGrads,
    KdaGrads,
    LaGrads,
    SsdGrads,
    reference_gla_backward,
    reference_gla_forward,
    reference_kda_backward,
    reference_kda_forward,
    reference_la_backward,
    reference_la_forward,
    reference_ssd_backward,
    reference_ssd_forward,
)
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
from .gdn2_chunkwise_cw import (
    ChunkwiseBackwardCW,
    ChunkwiseForwardCW,
    MicroGateBundleCW,
    build_microgate_bundles_cw,
    chunkwise_backward_cw,
    chunkwise_forward_cw,
)
from .gdn_backward import Gdn2Grads, reference_gdn2_backward, reference_gdn2_forward
from .mimo_backward import MimoGrads, reference_mimo_backward, reference_mimo_forward

__all__ = [
    "ChunkwiseBackward",
    "ChunkwiseBackwardCW",
    "ChunkwiseForward",
    "ChunkwiseForwardCW",
    "FusedBlockGrads",
    "Gdn2Grads",
    "GlaGrads",
    "KdaGrads",
    "LaGrads",
    "MicroGateBundle",
    "MicroGateBundleCW",
    "MimoGrads",
    "SelectiveScanGrads",
    "SsdGrads",
    "build_microgate_bundles",
    "build_microgate_bundles_cw",
    "chunkwise_backward",
    "chunkwise_backward_cw",
    "chunkwise_forward",
    "chunkwise_forward_cw",
    "reference_backward_selective_scan",
    "reference_complex_scan_rope",
    "reference_forward_chunked_scan",
    "reference_forward_trapezoidal_scan",
    "reference_fused_block_backward",
    "reference_fused_block_forward",
    "reference_gdn2_backward",
    "reference_gdn2_forward",
    "reference_gla_backward",
    "reference_gla_forward",
    "reference_kda_backward",
    "reference_kda_forward",
    "reference_la_backward",
    "reference_la_forward",
    "reference_mimo_backward",
    "reference_mimo_forward",
    "reference_ssd_backward",
    "reference_ssd_forward",
]
