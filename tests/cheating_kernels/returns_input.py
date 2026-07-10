"""Cheating kernel: returns its input unchanged."""

from __future__ import annotations

from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    return u.clone()
