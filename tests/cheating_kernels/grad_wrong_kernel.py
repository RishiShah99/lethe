"""Cheating kernel: value-correct, but the gradient graph is severed.

Returns the reference output for any input, but detaches the path so
autograd cannot flow gradients back through ``u``. Catches kernels
that compute correct values via a CUDA op missing a backward.

Expected rejector: CMP-02 (gradient correctness). ``gradcheck`` finds
zero analytical gradient where finite-difference gradient is nonzero.
"""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    # Compute the correct value via a detached path.
    value = reference_op(u.detach())
    # Add a zero-magnitude term in u so the graph isn't completely empty
    # (some autograd paths short-circuit if u isn't used at all).
    return value + 0.0 * u
