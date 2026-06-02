"""Canonical reference for the cheating-kernels suite.

All cheating kernels in this directory pretend to compute this same op.
The op is simple, shape-polymorphic, dtype-polymorphic, and exercises:
  - a reduction (catches reduction-order bugs)
  - NaN/Inf propagation
  - finite-precision behavior across fp32/fp16/bf16

It stands in for ``reference_forward_chunked_scan``-style kernels: the
verifier behavior under test is independent of the specific Mamba math.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def reference_op(u: Tensor) -> Tensor:
    """y = silu(u) + 0.01 * sum(silu(u), dim=-1, keepdim=True)."""
    activated = F.silu(u)
    pooled = activated.sum(dim=-1, keepdim=True) * 0.01
    return activated + pooled
