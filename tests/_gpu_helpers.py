"""Helper callables for GPU sandbox tests."""

import torch


def gpu_square(x: torch.Tensor) -> torch.Tensor:
    """Square on the GPU, return on CPU (sandbox IPC pickles the result)."""
    return (x.cuda() ** 2).cpu()
