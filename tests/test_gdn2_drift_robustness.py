"""Drifted-decay-regime robustness — the burst-2 native-training NaN regression.

Training drifts the decay gate hard negative (delayed-copy pushes memory short); once
the within-chunk log2 span crosses ~128 the unmasked ``exp2(g2_i - g2_s)`` upper
triangle overflowed fp32 to inf and the masked einsums produced ``0 * inf = NaN`` —
the assembly NaN'd at tiny-train step 3 while the token-serial eager path trained.
The masked-exponent ``decay_rel``/``ratio`` forms remove both failure modes (upper
overflow; live 0/0 after gamma underflow). These tests pin them.

CPU only; the pure-torch kernel refs stand in for the box kernels (identical glue).
"""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.cute.gdn2_assemble import (
    assemble_gdn2_backward_scalar,
    assembled_channelwise_gdn2_backward,
)
from lethe.kernels.references.gdn_backward import reference_gdn2_backward

# mean per-token log-decay ~-1.7 -> chunk log2 span ~154 > 128: past the old fp32
# exp2 cliff (and past fp32 gamma underflow at fp64's wider ranges stays exact).
_MEAN_G = -1.7


def _drifted_inputs(dtype: torch.dtype, seed: int = 7):
    b, t, h, d_k, d_v = 2, 128, 2, 16, 16
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = _MEAN_G - 0.5 * torch.rand(b, t, h, d_k, generator=gen, dtype=dtype)
    b_gate = torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.9 + 0.05
    w_gate = torch.rand(b, t, h, d_v, generator=gen, dtype=dtype) * 0.9 + 0.05
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, b_gate, w_gate, do


@pytest.mark.parametrize("stage_b_closed", [False, True])
def test_cw_assembly_finite_in_drifted_regime(stage_b_closed: bool) -> None:
    """fp32 assembly stays finite past the old exp2 overflow cliff (both stage-B paths)."""
    q, k, v, g, b_gate, w_gate, do = _drifted_inputs(torch.float32)
    out = assembled_channelwise_gdn2_backward(
        q, k, v, g, b_gate, w_gate, do, stage_b_closed=stage_b_closed
    )
    for name in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
        t = getattr(out, name)
        assert torch.isfinite(t).all(), f"{name} non-finite in drifted regime"


@pytest.mark.parametrize("stage_b_closed", [False, True])
def test_cw_assembly_matches_oracle_in_drifted_regime(stage_b_closed: bool) -> None:
    """fp64 assembly still equals the token-serial oracle in the drifted regime."""
    q, k, v, g, b_gate, w_gate, do = _drifted_inputs(torch.float64)
    out = assembled_channelwise_gdn2_backward(
        q, k, v, g, b_gate, w_gate, do, stage_b_closed=stage_b_closed
    )
    oracle = reference_gdn2_backward(q, k, v, g, b_gate, w_gate, do)
    for name in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
        got, ref = getattr(out, name), getattr(oracle, name)
        assert torch.allclose(got, ref, rtol=1e-7, atol=1e-9), name


def test_scalar_assembly_finite_past_gamma_underflow() -> None:
    """Scalar path survives fp32 gamma underflow (the old ratio division hit 0/0 live)."""
    b, t, h, d_k, d_v = 1, 128, 2, 16, 16
    gen = torch.Generator().manual_seed(11)
    q = torch.randn(b, t, h, d_k, generator=gen)
    k = torch.randn(b, t, h, d_k, generator=gen)
    v = torch.randn(b, t, h, d_v, generator=gen)
    # mean g -4/token -> chunk log2 span ~369, far past fp32 denormal floor (~-149).
    g = -4.0 - torch.rand(b, t, h, generator=gen)
    beta = torch.rand(b, t, h, generator=gen) * 0.9 + 0.05
    do = torch.randn(b, t, h, d_v, generator=gen)

    out = assemble_gdn2_backward_scalar(q, k, v, g, beta, do)
    for name in ("dq", "dk", "dv", "dg", "db_erase", "dw_write"):
        assert torch.isfinite(getattr(out, name)).all(), name
