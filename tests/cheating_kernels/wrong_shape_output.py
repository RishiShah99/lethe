"""Cheating kernel: returns an output with one fewer time-step than expected."""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    full = reference_op(u)
    if full.ndim < 2 or full.shape[-2] < 1:
        return full
    # Drop the last step along the sequence axis (penultimate dim).
    slicer: list[slice | int] = [slice(None)] * full.ndim
    slicer[-2] = slice(0, full.shape[-2] - 1)
    return full[tuple(slicer)]
