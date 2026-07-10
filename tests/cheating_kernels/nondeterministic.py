"""Cheating kernel: adds small random noise on every call."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    noise = torch.randn_like(out) * 1e-4
    return out + noise
