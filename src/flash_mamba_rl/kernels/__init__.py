"""Six hand-written Mamba-3 Triton kernels + PyTorch reference oracles."""

from .loader import (
    KernelCandidate,
    discover_candidates,
    import_candidate,
    load_candidate,
)

__all__ = [
    "KernelCandidate",
    "discover_candidates",
    "import_candidate",
    "load_candidate",
]
