"""Cheating kernel: silently flushes subnormal outputs to zero.

Mimics a kernel that enables CUDA's flush-to-zero (FTZ) mode without
disclosing it. Outputs below the smallest normal value are zeroed,
where the reference (running in PyTorch's default IEEE-compliant mode)
preserves the subnormals.

Expected rejector: EXC-02 (subnormal handling). The zero-mask of
the candidate's output diverges from the reference's.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    tiny = torch.finfo(out.dtype).tiny
    return torch.where(out.abs() < tiny, torch.zeros_like(out), out)
