"""Cheating kernel: returns a pre-cached "correct" tensor regardless of input.

At import time the module computes the reference output for one canonical
shape and dtype, then returns that cached tensor for every call (resized
to match the input shape).

Expected rejector: CMP-01 (input variation). The reference output varies
with input; the cheater is constant.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op

_CANONICAL_SHAPE = (4, 64, 32)
_CANONICAL_DTYPE = torch.float32

torch.manual_seed(0)
_canonical_input = torch.randn(_CANONICAL_SHAPE, dtype=_CANONICAL_DTYPE)
_CACHED_OUTPUT: Tensor = reference_op(_canonical_input).detach()


def cheating_op(u: Tensor) -> Tensor:
    # Resize/broadcast the cached output to match the requested shape and dtype.
    target_numel = u.numel()
    flat = _CACHED_OUTPUT.flatten()
    if target_numel <= flat.numel():
        return flat[:target_numel].reshape(u.shape).to(u.dtype)
    # Repeat to fill if input is larger than cache.
    reps = (target_numel + flat.numel() - 1) // flat.numel()
    return flat.repeat(reps)[:target_numel].reshape(u.shape).to(u.dtype)
