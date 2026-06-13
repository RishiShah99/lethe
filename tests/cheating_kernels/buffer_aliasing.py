"""Cheating kernel: hands back one persistent output buffer on every call.

Computes the reference correctly into a module-level buffer and returns that
same storage each call. The values are correct and deterministic, so every
value/precision gate (and ORD-02's equality check) passes — but two results
alias: holding the first result and calling again silently overwrites it.
ORD-02's cross-call aliasing check (distinct ``data_ptr`` per call) rejects it.

Expected rejector: ORD-02 (atomic determinism / output-buffer aliasing).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op

_buffer: list[Tensor] = []


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    if not _buffer or _buffer[0].shape != out.shape or _buffer[0].dtype != out.dtype:
        _buffer.clear()
        _buffer.append(torch.empty_like(out))
    buf = _buffer[0]
    buf.copy_(out)
    return buf
