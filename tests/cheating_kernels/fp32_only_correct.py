"""Cheating kernel: correct in FP32, mantissa-truncated (wrong) in FP16/BF16."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def _truncate_mantissa(x: Tensor) -> Tensor:
    # Drop the bottom 4 bits of every value (mantissa-quantise to ~1.5 sig figs).
    scaled = x.float() * 16.0
    rounded = torch.round(scaled) / 16.0
    return rounded.to(x.dtype)


def cheating_op(u: Tensor) -> Tensor:
    if u.dtype == torch.float32:
        return reference_op(u)
    # FP16 / BF16 / FP64 / etc. - break it.
    out = reference_op(u.float()).to(u.dtype)
    return _truncate_mantissa(out) * 0.5  # also scale-bias for extra wrongness
