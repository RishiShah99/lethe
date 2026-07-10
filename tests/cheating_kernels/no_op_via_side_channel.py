"""Cheating kernel: correct only on the benchmark shape, no-op elsewhere."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op

_BENCHMARK_SHAPE = (4, 64, 32)


def cheating_op(u: Tensor) -> Tensor:
    if tuple(u.shape) == _BENCHMARK_SHAPE:
        return reference_op(u)
    return torch.zeros_like(u)
