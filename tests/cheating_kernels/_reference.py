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
    """y = silu(u) + 0.01 * sum(silu(u), dim=-1, keepdim=True).

    The pool reduction runs through a higher-precision accumulator for
    reduced-precision inputs (FP32 accumulator for FP16/BF16) so that long
    reductions remain within FP16-ULP of the FP32 result. FP32/FP64 inputs
    accumulate in their own dtype — no lossy downcast that would break
    gradcheck under FP64. PRC-02 verifies candidates match this behavior.
    """
    import torch

    activated = F.silu(u)
    if u.dtype in (torch.float16, torch.bfloat16):
        acc = activated.to(torch.float32).sum(dim=-1, keepdim=True).to(u.dtype)
    else:
        acc = activated.sum(dim=-1, keepdim=True)
    return activated + acc * 0.01
