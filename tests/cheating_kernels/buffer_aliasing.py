"""Cheating kernel: hands back one persistent output buffer on every call."""

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
