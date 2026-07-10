"""Cheating kernel: value-correct, but the gradient graph is severed."""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    # Compute the correct value via a detached path.
    value = reference_op(u.detach())
    # Adds a zero-magnitude term so u stays in the graph (avoids autograd short-circuit).
    return value + 0.0 * u
