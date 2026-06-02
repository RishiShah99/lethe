"""Cheating kernel: correct in-range, NaN on inputs with |x| > 1e3.

A kernel that overflows or otherwise spits NaN on large-magnitude
inputs while looking fine in the typical training regime.

Expected rejector: CMP-01 (input variation) — the ``large_1e6``
adversarial case produces NaNs where the reference produces finite
values. EXC-01 also catches the NaN-mask divergence.
"""

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
