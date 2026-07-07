"""Pins the Mamba-3 trapezoidal-discretisation oracle (Prop. 3.2.2).

``reference_forward_trapezoidal_scan`` is the data-dependent trapezoidal
generalisation of the ZOH ``reference_forward_chunked_scan``. These tests anchor
it three ways: (1) the lambda=1 limit is bit-identical to the ZOH oracle (the
generalisation is consistent), (2) a fully hand-computed L=2 case pins the
beta/gamma split structurally, (3) the beta (previous-token) term is genuinely
active at lambda<1. No GPU; pure float32 oracle math.
"""

import math

import torch

from lethe.kernels.references import (
    reference_forward_chunked_scan,
    reference_forward_trapezoidal_scan,
)


def _siso_inputs(
    batch: int, seq_len: int, d_model: int, n_state: int, seed: int
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(batch, seq_len, d_model, generator=g),
        "delta": torch.randn(batch, seq_len, d_model, generator=g) * 0.5,
        "A": -torch.rand(d_model, n_state, generator=g) - 0.1,  # negative log-magnitude
        "B": torch.randn(batch, seq_len, n_state, generator=g),
        "C": torch.randn(batch, seq_len, n_state, generator=g),
        "D": torch.randn(d_model, generator=g),
        "trap": torch.randn(batch, seq_len, d_model, generator=g),
    }


class TestLambdaOneReduction:
    def test_trapezoidal_reduces_to_zoh_at_lambda_one(self) -> None:
        # lambda = sigmoid(trap); trap=+30 -> sigmoid rounds to exactly 1.0 in fp32,
        # so beta = (1-1)*... = 0.0 exactly and gamma = delta_bar. The oracle groups
        # the current term as (1.0*b_bar)*u == b_bar*u and adds the 0.0 beta term,
        # so the result must be BIT-IDENTICAL to the ZOH oracle, not merely close.
        for seed in (0, 1, 7):
            inp = _siso_inputs(2, 64, 8, 16, seed)
            zoh = reference_forward_chunked_scan(
                inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], chunk_size=16
            )
            trap_one = torch.full_like(inp["trap"], 30.0)
            assert torch.sigmoid(trap_one).eq(1.0).all(), "trap=30 must give lambda==1.0 in fp32"
            trap = reference_forward_trapezoidal_scan(
                inp["u"],
                inp["delta"],
                inp["A"],
                inp["B"],
                inp["C"],
                inp["D"],
                trap_one,
                chunk_size=16,
            )
            assert torch.equal(trap, zoh), f"lambda=1 not bit-identical to ZOH (seed {seed})"


class TestHandComputedL2:
    def test_beta_gamma_split_hand_computed(self) -> None:
        # B=D=N=1, L=2: fully scalar so the alpha/beta/gamma recurrence is
        # hand-derivable. Independent derivation (not the vectorised code path).
        u0, u1 = 0.7, -0.4
        d0, d1 = 0.3, -0.2
        a = -0.5
        b0, b1 = 1.1, 0.9
        c0, c1 = 0.6, -1.3
        dskip = 0.25
        tr0, tr1 = 0.0, 0.8  # lambda0 = 0.5, lambda1 = sigmoid(0.8)

        u = torch.tensor([[[u0], [u1]]], dtype=torch.float32)
        delta = torch.tensor([[[d0], [d1]]], dtype=torch.float32)
        A = torch.tensor([[a]], dtype=torch.float32)
        B = torch.tensor([[[b0], [b1]]], dtype=torch.float32)
        C = torch.tensor([[[c0], [c1]]], dtype=torch.float32)
        D = torch.tensor([dskip], dtype=torch.float32)
        trap = torch.tensor([[[tr0], [tr1]]], dtype=torch.float32)

        out = reference_forward_trapezoidal_scan(u, delta, A, B, C, D, trap, chunk_size=2)

        sp = lambda x: math.log1p(math.exp(x))  # softplus  # noqa: E731
        sig = lambda x: 1.0 / (1.0 + math.exp(-x))  # noqa: E731
        db0, db1 = sp(d0), sp(d1)
        ab1 = math.exp(db1 * a)  # alpha_1; t=0 has no decay term (h_{-1}=0)
        lam0, lam1 = sig(tr0), sig(tr1)

        # t=0: no previous token -> beta term absent
        h0 = lam0 * (db0 * b0) * u0
        y0 = h0 * c0 + dskip * u0
        # t=1: alpha*h0 + gamma1*B1*u1 + beta1*alpha1*(B0*u0)
        beta1 = (1.0 - lam1) * db1
        h1 = ab1 * h0 + lam1 * (db1 * b1) * u1 + beta1 * ab1 * (b0 * u0)
        y1 = h1 * c1 + dskip * u1

        expected = torch.tensor([[[y0], [y1]]], dtype=torch.float32)
        assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5), f"{out} != {expected}"


class TestBetaTermActive:
    def test_beta_term_changes_output_below_lambda_one(self) -> None:
        # At lambda<1 the previous-token (beta) term contributes, so the
        # trapezoidal output must DIFFER from ZOH — proving the term is live,
        # not silently dropped.
        inp = _siso_inputs(2, 32, 4, 8, seed=3)
        zoh = reference_forward_chunked_scan(
            inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], chunk_size=8
        )
        trap_half = torch.zeros_like(inp["trap"])  # lambda = 0.5 everywhere
        trapz = reference_forward_trapezoidal_scan(
            inp["u"],
            inp["delta"],
            inp["A"],
            inp["B"],
            inp["C"],
            inp["D"],
            trap_half,
            chunk_size=8,
        )
        assert not torch.allclose(trapz, zoh, atol=1e-4), "beta term inert at lambda=0.5"


class TestInputValidation:
    def test_rejects_non_float32(self) -> None:
        inp = _siso_inputs(1, 8, 2, 4, seed=0)
        try:
            reference_forward_trapezoidal_scan(
                inp["u"].double(),
                inp["delta"],
                inp["A"],
                inp["B"],
                inp["C"],
                inp["D"],
                inp["trap"],
                chunk_size=8,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError on non-float32 input")

    def test_rejects_indivisible_chunk(self) -> None:
        inp = _siso_inputs(1, 10, 2, 4, seed=0)
        try:
            reference_forward_trapezoidal_scan(
                inp["u"],
                inp["delta"],
                inp["A"],
                inp["B"],
                inp["C"],
                inp["D"],
                inp["trap"],
                chunk_size=4,
            )
        except ValueError:
            return
        raise AssertionError("expected ValueError on indivisible chunk_size")
