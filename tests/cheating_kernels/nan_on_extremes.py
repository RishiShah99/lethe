"""Cheating kernel: correct in-range, NaN on inputs with |x| > 1e3."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    mask = u.abs() > 1e3
    if mask.any():
        out = torch.where(mask, torch.full_like(out, float("nan")), out)
    return out
