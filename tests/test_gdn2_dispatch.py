"""GDN-2 backward dispatch: native shim unavailable off-box -> eager fallback.

Proves the Phase-2 integration boundary is correct *before* the kernel exists:
the candidate ``gdn2_backward`` falls back to the oracle-faithful eager path and
matches ``reference_gdn2_backward`` to machine precision in fp64.
"""

from __future__ import annotations

import inspect

import pytest
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


def test_bmm_tc_rejects_oversize_staging_dims() -> None:
    # _bmm_tc stages M,K into a (D_K,D_K) buffer; dims above D_K must raise a
    # named precondition rather than truncate/pad silently. Runs off-box — the
    # raise precedes any tcgen05 GEMM.
    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import D_K, _bmm_tc

    with pytest.raises(ValueError, match="stages M,K"):
        _bmm_tc(torch.zeros(2, D_K + 1, 8), torch.zeros(2, 8, 8))
    with pytest.raises(ValueError, match="stages M,K"):
        _bmm_tc(torch.zeros(2, 8, D_K + 1), torch.zeros(2, D_K + 1, 8))

    # the unbatched serial twin (_mm_tc in the WY file) shares the staging contract
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy import _mm_tc

    with pytest.raises(ValueError, match="stages M,K"):
        _mm_tc(torch.zeros(D_K + 1, 8), torch.zeros(8, 8))
    with pytest.raises(ValueError, match="stages M,K"):
        _mm_tc(torch.zeros(8, D_K + 1), torch.zeros(D_K + 1, 8))


def test_incb_entry_guards_and_keyed_gemm_cache() -> None:
    from flash_mamba_rl.kernels.cute import gdn2_bwd_dhu

    for fn in (gdn2_bwd_dhu.run_k1_incB_host, gdn2_bwd_dhu.run_k1_incB_batched):
        assert "d_k > D_K" in inspect.getsource(fn)
    src = inspect.getsource(gdn2_bwd_dhu)
    assert "_gemm_aa_cache" in src  # compile cache keyed on operand shape/dtype
    assert "global _gemm_compiled" not in src  # the unconditional single-slot cache is gone


class TestCreateGraphFallback:
    """Regression: do.requires_grad=True under grad must route to eager, not native."""

    def _make_inputs(self, seed: int = 0, dtype: torch.dtype = torch.float32):
        b, t, h, dk, dv = 1, 64, 2, 128, 64
        gen = torch.Generator().manual_seed(seed)
        q = torch.randn(b, t, h, dk, generator=gen, dtype=dtype)
        k = torch.randn(b, t, h, dk, generator=gen, dtype=dtype)
        v = torch.randn(b, t, h, dv, generator=gen, dtype=dtype)
        g = -torch.rand(b, t, h, dk, generator=gen, dtype=dtype) * 0.1
        beta = torch.rand(b, t, h, generator=gen, dtype=dtype) * 0.8 + 0.1
        b_gate = beta.unsqueeze(-1).expand(b, t, h, dk).contiguous()
        w_gate = beta.unsqueeze(-1).expand(b, t, h, dv).contiguous()
        do = torch.randn(b, t, h, dv, generator=gen, dtype=dtype)
        return q, k, v, g, b_gate, w_gate, do

    def test_native_declines_when_do_requires_grad(self, monkeypatch) -> None:
        """With do.requires_grad=True, native_gdn2_backward returns None (eager fallback)."""
        monkeypatch.setattr(
            "flash_mamba_rl.kernels.cute.gdn2_backward.is_available", lambda device=None: True
        )
        q, k, v, g, b, w, do = self._make_inputs()
        do_grad = do.clone().requires_grad_(True)

        result = native_gdn2_backward(q, k, v, g, b, w, do_grad)
        assert result is None, "native dispatch must decline when do.requires_grad=True"

    def test_native_proceeds_when_do_detached(self, monkeypatch) -> None:
        """With do.requires_grad=False, native_gdn2_backward proceeds (kernel route)."""
        from flash_mamba_rl.kernels.cute.gdn2_assemble import (
            k1_reverse_state_cw_ref,
            k1_reverse_state_ref,
            k2_wy_vjp_cw_ref,
            k2_wy_vjp_ref,
        )

        monkeypatch.setattr(
            "flash_mamba_rl.kernels.cute.gdn2_backward.is_available", lambda device=None: True
        )
        monkeypatch.setattr(
            "flash_mamba_rl.kernels.cute.gdn2_backward._load_box_kernels",
            lambda: (k1_reverse_state_ref, k2_wy_vjp_ref),
        )
        monkeypatch.setattr(
            "flash_mamba_rl.kernels.cute.gdn2_backward._load_box_kernels_cw",
            lambda: (k1_reverse_state_cw_ref, k2_wy_vjp_cw_ref),
        )
        q, k, v, g, b, w, do = self._make_inputs()
        do_detached = do.detach()

        result = native_gdn2_backward(q, k, v, g, b, w, do_detached)
        assert result is not None, "native dispatch must proceed when do.requires_grad=False"

    def test_family_natives_decline_when_do_requires_grad(self, monkeypatch) -> None:
        """All native_*_backward family wrappers decline when do.requires_grad=True."""
        from flash_mamba_rl.kernels.cute.gdn2_backward import (
            native_gla_backward,
            native_kda_backward,
            native_la_backward,
            native_ssd_backward,
        )

        monkeypatch.setattr(
            "flash_mamba_rl.kernels.cute.gdn2_backward.is_available", lambda device=None: True
        )
        b, t, h, dk, dv = 1, 64, 2, 128, 128
        gen = torch.Generator().manual_seed(42)
        q = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float32)
        k = torch.randn(b, t, h, dk, generator=gen, dtype=torch.float32)
        v = torch.randn(b, t, h, dv, generator=gen, dtype=torch.float32)
        g = -torch.rand(b, t, h, dk, generator=gen, dtype=torch.float32) * 0.1
        g_ssd = -torch.rand(b, t, h, generator=gen, dtype=torch.float32) * 0.1
        beta = torch.rand(b, t, h, generator=gen, dtype=torch.float32) * 0.8 + 0.1
        do = torch.randn(b, t, h, dv, generator=gen, dtype=torch.float32)
        do_grad = do.clone().requires_grad_(True)

        assert native_gla_backward(q, k, v, g, do_grad) is None
        assert native_la_backward(q, k, v, do_grad) is None
        assert native_ssd_backward(q, k, v, g_ssd, do_grad) is None
        assert native_kda_backward(q, k, v, g, beta, do_grad) is None
