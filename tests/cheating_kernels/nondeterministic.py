"""Cheating kernel: adds small random noise on every call.

Models a kernel with a non-commutative atomic reduction order — different
output across calls with identical input.

Expected rejector: ORD-02 (atomic determinism). Two runs with the
same input produce non-equal tensors.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    noise = torch.randn_like(out) * 1e-4
    return out + noise
