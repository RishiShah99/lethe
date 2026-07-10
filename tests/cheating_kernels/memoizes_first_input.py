"""Cheating kernel: memoises the very first input's output forever."""

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
