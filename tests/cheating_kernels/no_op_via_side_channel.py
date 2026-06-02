"""Cheating kernel: correct only on the benchmark shape, no-op elsewhere.

If the input shape matches the benchmark fingerprint, the kernel calls the
reference and returns the correct answer. On any other shape it returns
zeros — a classic "teach-to-the-test" cheat.

Expected rejector: CMP-03 (shape polymorphism). The benchmark shape passes,
every other shape returns the wrong answer.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op

_BENCHMARK_SHAPE = (4, 64, 32)


def cheating_op(u: Tensor) -> Tensor:
    if tuple(u.shape) == _BENCHMARK_SHAPE:
        return reference_op(u)
    return torch.zeros_like(u)
