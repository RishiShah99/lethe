"""Cheating kernel: pools by manual reverse-order summation.

Same operation as the reference, but the reduction is computed by
explicit element-by-element addition in reverse order. On adversarial
inputs (alternating large-positive / tiny / large-negative values),
floating-point non-associativity yields a different bit pattern
than the reference's ``.sum(dim=-1)``.

Expected rejector: ORD-03 (non-commutative reduction). Bit-exact
comparison fails on the adversarial reduction input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    activated = F.silu(u)
    n = activated.shape[-1]
    pooled = torch.zeros(
        *activated.shape[:-1], 1, dtype=u.dtype, device=u.device
    )
    # Reverse order accumulation — known to differ from torch.sum at the ULP.
    for i in range(n - 1, -1, -1):
        pooled = pooled + activated[..., i : i + 1]
    return activated + pooled * 0.01
