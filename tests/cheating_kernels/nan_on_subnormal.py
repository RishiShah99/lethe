"""Cheating kernel: mints NaN on subnormal-magnitude inputs."""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    smallest_normal = torch.finfo(u.dtype).tiny
    if bool((u.abs() < smallest_normal).all()):
        return torch.full_like(out, float("nan"))
    return out
