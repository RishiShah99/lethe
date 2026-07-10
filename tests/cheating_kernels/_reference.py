"""Canonical reference for the cheating-kernels suite."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def reference_op(u: Tensor) -> Tensor:
    """y = silu(u) + 0.01 * sum(silu(u), dim=-1, keepdim=True)."""
    import torch

    activated = F.silu(u)
    if u.dtype in (torch.float16, torch.bfloat16):
        acc = activated.to(torch.float32).sum(dim=-1, keepdim=True).to(u.dtype)
    else:
        acc = activated.sum(dim=-1, keepdim=True)
    return activated + acc * 0.01
