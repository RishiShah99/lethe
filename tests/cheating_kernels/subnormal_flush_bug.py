"""Cheating kernel: silently flushes subnormal outputs to zero."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    tiny = torch.finfo(out.dtype).tiny
    return torch.where(out.abs() < tiny, torch.zeros_like(out), out)
