"""Cheating kernel: silently moves output to the meta device."""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    return out.to("meta")
