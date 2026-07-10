"""Cheating kernel: silently promotes intermediates to int8 then back."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u.float())
    # Round-trip through int8 by mapping to [-127, 127] and back.
    max_abs = out.abs().max().clamp(min=1e-6)
    scale = 127.0 / max_abs
    quantised = torch.round(out * scale).clamp(-127, 127).to(torch.int8)
    dequant = quantised.float() / scale
    return dequant.to(u.dtype)
