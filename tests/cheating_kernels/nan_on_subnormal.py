"""Cheating kernel: mints NaN on subnormal-magnitude inputs.

The adversarial dual of ``subnormal_flush_bug`` — instead of flushing tiny
outputs to zero, it poisons them with NaN. It behaves exactly like the
reference on normal-magnitude inputs, so only EXC-02 (the sole gate that
feeds all-subnormal probes) can see the defect.

Expected rejector: EXC-02 (subnormal handling). This is the regression for
the pre-fix false-accept: ``NaN != 0`` kept the flush-to-zero mask matching
and a ``NaN`` max_err slipped past the ``> atol`` check, so the gate wrongly
returned ``passed=True``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._reference import reference_op


def cheating_op(u: Tensor) -> Tensor:
    out = reference_op(u)
    smallest_normal = torch.finfo(u.dtype).tiny
    if bool((u.abs() < smallest_normal).all()):
        return torch.full_like(out, float("nan"))
    return out
