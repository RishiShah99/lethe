"""Cheating kernel: pools by manual reverse-order summation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cheating_op(u: Tensor) -> Tensor:
    activated = F.silu(u)
    n = activated.shape[-1]
    pooled = torch.zeros(*activated.shape[:-1], 1, dtype=u.dtype, device=u.device)
    # Reverse-order accumulation differs from torch.sum at the ULP.
    for i in range(n - 1, -1, -1):
        pooled = pooled + activated[..., i : i + 1]
    return activated + pooled * 0.01
