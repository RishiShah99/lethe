"""Cheating kernel: randomly zeroes a small fraction of output elements.

Looks correct most of the time — about 95 % of values pass through
unchanged — but a random ~5 % are zeroed per call, varying between
calls. Catches naive few-trial verifiers that test correctness once
and trust it forever.

Expected rejector: ORD-02 (atomic determinism). The dropout mask
differs across runs, so two runs with the same input disagree.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    mask = torch.rand_like(out) > 0.05  # keep 95 %, drop 5 %
    return out * mask
