"""GDN-2 backward dispatch: native shim unavailable off-box -> eager fallback.

Proves the Phase-2 integration boundary is correct *before* the kernel exists:
the candidate ``gdn2_backward`` falls back to the oracle-faithful eager path and
matches ``reference_gdn2_backward`` bit-for-bit in fp64.
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.cute.gdn2_backward import (
    _is_scalar_reducible,
    is_available,
    native_gdn2_backward,
)
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


def _scalar_gates(seed: int, dk: int = 16, dv: int = 16):
    b, t, h = 2, 8, 3
    gen = torch.Generator().manual_seed(seed)
    g_scalar = -torch.rand(b, t, h, 1, generator=gen)
    beta = torch.rand(b, t, h, 1, generator=gen)
    g = g_scalar.expand(b, t, h, dk).contiguous()
    b_gate = beta.expand(b, t, h, dk).contiguous()
    w_gate = beta.expand(b, t, h, dv).contiguous()
    return g, b_gate, w_gate


def test_exact_broadcast_routes_to_scalar() -> None:
    g, b_gate, w_gate = _scalar_gates(seed=0)
    assert _is_scalar_reducible(g, b_gate, w_gate) is True


def test_near_constant_channelwise_routes_to_channelwise() -> None:
    # g is per-channel constant to within allclose's default tolerance but NOT
    # bitwise broadcast — it must route to the channel-wise crown, not collapse
    # to the scalar assembly (which would return the wrong grad_g/b/w).
    g, b_gate, w_gate = _scalar_gates(seed=1)
    jitter = torch.zeros_like(g)
    jitter[..., 0] = 1e-9
    g_near = g + jitter
    assert torch.allclose(g_near, g_near[..., :1].expand_as(g_near))  # the trap allclose fell into
    assert _is_scalar_reducible(g_near, b_gate, w_gate) is False
