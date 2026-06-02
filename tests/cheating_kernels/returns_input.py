"""Cheating kernel: returns its input unchanged.

The most naive form of cheating — claim to compute the op, do nothing.

Expected rejector: CMP-01 (input variation), PRC-01 (precision regime),
EXC-01 (NaN/Inf masks). The returned tensor disagrees with the reference
on virtually every input.
"""

from __future__ import annotations

from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    return u.clone()
