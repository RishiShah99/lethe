"""Helper callables for GPU sandbox tests.

Imported by name inside the sandbox child process — keep this module
import-safe and side-effect free.
"""

import torch


def gpu_square(x: torch.Tensor) -> torch.Tensor:
    """Square on the GPU, return on CPU (sandbox IPC pickles the result)."""
    return (x.cuda() ** 2).cpu()
