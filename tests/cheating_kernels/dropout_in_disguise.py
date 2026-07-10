"""Cheating kernel: randomly zeroes a small fraction of output elements."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    mask = torch.rand_like(out) > 0.05  # keep 95 %, drop 5 %
    return out * mask
