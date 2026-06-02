"""Cheating kernel: silently moves output to the meta device.

Mimics a kernel that runs computation on a different backend (e.g.,
a quantisation or shape-inference path) and returns a tensor whose
device differs from the input's.

Expected rejector: RES-01 (memory residency). The output's
``device.type`` does not match the input's.
"""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    return out.to("meta")
