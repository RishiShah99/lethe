"""GDN-2 backward dispatch: native shim unavailable off-box -> eager fallback.

Proves the Phase-2 integration boundary is correct *before* the kernel exists:
the candidate ``gdn2_backward`` falls back to the oracle-faithful eager path and
matches ``reference_gdn2_backward`` bit-for-bit in fp64.
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.cute.gdn2_backward import is_available, native_gdn2_backward
from flash_mamba_rl.kernels.ops.gdn_backward import gdn2_backward
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward


def _inputs(seed: int = 0):
    b, t, h, dk, dv = 2, 32, 3, 16, 16
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float64)
    k = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float64)
    v = torch.randn(b, t, h, dv, generator=gen, dtype=torch.float64)
    g = -torch.rand(b, t, h, dk, generator=gen, dtype=torch.float64) * 0.1
    beta = torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    b_gate = beta.unsqueeze(-1).expand(b, t, h, dk).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, dv).contiguous()
    do = torch.randn(b, t, h, dv, generator=gen, dtype=torch.float64)
    return q, k, v, g, b_gate, w_gate, do


def test_native_unavailable_off_box() -> None:
    assert is_available(torch.device("cpu")) is False
    q, k, v, g, b, w, do = _inputs()
    assert native_gdn2_backward(q, k, v, g, b, w, do) is None


def test_dispatch_falls_back_to_oracle() -> None:
    q, k, v, g, b, w, do = _inputs(seed=1)
    got = gdn2_backward(q, k, v, g, b, w, do)
    want = reference_gdn2_backward(q, k, v, g, b, w, do)
    for name in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
        torch.testing.assert_close(getattr(got, name), getattr(want, name), rtol=0.0, atol=0.0)
    assert got.grad_initial_state is None
