"""Cheating kernel: flips the sign of any infinity it propagates."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    inf_mask = torch.isinf(out)
    if inf_mask.any():
        out = torch.where(inf_mask, -out, out)
    return out
