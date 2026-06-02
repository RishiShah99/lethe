"""Cheating kernel: correct on a privileged shape, wrong on neighbours.

Distinct from ``no_op_via_side_channel``: this one returns *plausible*
wrong values (scaled reference output) rather than zeros, so it might
sneak past a naïve "is output all zeros?" check.

Expected rejector: CMP-03 (shape polymorphism). Shapes other than
``[128, 128]`` (and the benchmark shape) get a deliberately scaled
output.
"""

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
