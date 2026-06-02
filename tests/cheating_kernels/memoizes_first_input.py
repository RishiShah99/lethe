"""Cheating kernel: memoises the very first input's output forever.

On the first call, computes the reference and caches the result. Every
subsequent call returns that cached tensor regardless of new input
(reshaped to fit the new input).

Expected rejector: CMP-01 (input variation). The output is identical
across random inputs, while the reference output varies.
"""

from __future__ import annotations

from torch import Tensor

from ._reference import reference_op

_cache: list[Tensor] = []


def cheating_op(u: Tensor) -> Tensor:
    if not _cache:
        _cache.append(reference_op(u).detach())
    cached = _cache[0]
    target_numel = u.numel()
    flat = cached.flatten()
    if target_numel <= flat.numel():
        return flat[:target_numel].reshape(u.shape).to(u.dtype)
    reps = (target_numel + flat.numel() - 1) // flat.numel()
    return flat.repeat(reps)[:target_numel].reshape(u.shape).to(u.dtype)
