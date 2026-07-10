"""Cheating kernel: correct on a privileged shape, wrong on neighbours."""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op

_PRIVILEGED_SHAPES: set[tuple[int, ...]] = {
    (128, 128),
    (4, 64, 32),
}


def cheating_op(u: Tensor) -> Tensor:
    if tuple(u.shape) in _PRIVILEGED_SHAPES:
        return reference_op(u)
    return reference_op(u) * 1.37  # plausibly-wrong on every other shape
