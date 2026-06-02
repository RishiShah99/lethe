"""Cheating kernel: accumulates in input precision instead of FP32.

Same algorithm as the reference, but the pool reduction runs in the
input dtype throughout — no FP32 promotion. Across a 128-element
reduction, the FP16 accumulator's ULP drift accumulates past the
FP16 tolerance.

Expected rejector: PRC-02 (mixed-precision accumulation). The FP16
candidate output drifts further from the FP32 reference than a
proper FP32-accumulating kernel would.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    activated = F.silu(u)
    n = activated.shape[-1]
    # Manual loop in u.dtype — no upcast.
    pooled = torch.zeros(
        *activated.shape[:-1], 1, dtype=u.dtype, device=u.device
    )
    for i in range(n):
        pooled = pooled + activated[..., i : i + 1]
    return activated + pooled * 0.01
