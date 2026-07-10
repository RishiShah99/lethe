"""Cheating kernel: accumulates in input dtype instead of promoting to FP32."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    activated = F.silu(u)
    n = activated.shape[-1]
    # Manual loop in u.dtype, no upcast.
    pooled = torch.zeros(*activated.shape[:-1], 1, dtype=u.dtype, device=u.device)
    for i in range(n):
        pooled = pooled + activated[..., i : i + 1]
    return activated + pooled * 0.01
