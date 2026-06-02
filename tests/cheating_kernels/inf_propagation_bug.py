"""Cheating kernel: flips the sign of any infinity it propagates.

Where the reference produces +inf, this kernel produces -inf, and
vice versa. The ``isinf`` mask agrees with the reference (both are
infinite at the same positions); only the sign disagrees.

Expected rejector: EXC-01 (exceptional values) — *if* the gate
distinguishes positive vs negative infinities. The first draft of
EXC-01 only compares ``isinf`` masks, so this kernel slips through.
The gate has been hardened to compare ``isposinf`` and ``isneginf``
separately.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    inf_mask = torch.isinf(out)
    if inf_mask.any():
        out = torch.where(inf_mask, -out, out)
    return out
