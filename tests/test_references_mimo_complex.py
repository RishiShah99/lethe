"""Tests for the Mamba-3 MIMO and complex-RoPE scan reference oracles.

Covers:
  - reference_complex_scan_rope: shape, determinism, zero-angle identity,
    causality of angle accumulation.
  - reference_mimo_forward + reference_mimo_backward: shape, determinism,
    R=1 SISO reduction, grad existence/shape, gradcheck (float64).

All tests run on CPU only.  No GPU markers.
"""

import math

import torch
from torch import Tensor

from lethe.kernels.references import (
    MimoGrads,
    reference_complex_scan_rope,
    reference_mimo_backward,
    reference_mimo_forward,
)

# ---------------------------------------------------------------------------
# Shared small dimensions (oracle is slow; keep tiny)
# ---------------------------------------------------------------------------
BATCH = 2
SEQ = 6
NHEADS = 3
HEADDIM = 4
D_STATE = 8  # must be >= 2 * NUM_ROPE for rope tests
NUM_ROPE = 2  # num_rope_angles; 2 * NUM_ROPE = 4 <= D_STATE
R = 2  # MIMO rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rope_inputs(
    batch: int = BATCH,
    seqlen: int = SEQ,
    nheads: int = NHEADS,
    headdim: int = HEADDIM,
    d_state: int = D_STATE,
    num_rope: int = NUM_ROPE,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (x, B, C, dt, A, angle_proj) for complex_scan_rope."""
    torch.manual_seed(seed)
    x = torch.randn(batch, seqlen, nheads, headdim)
    B = torch.randn(batch, seqlen, nheads, d_state)
    C = torch.randn(batch, seqlen, nheads, d_state)
    dt = torch.rand(batch, seqlen, nheads) * 0.5 + 0.1  # positive
    A = -torch.rand(nheads) - 0.1  # negative
    angle_proj = torch.randn(batch, seqlen, nheads, num_rope)
    return x, B, C, dt, A, angle_proj


def _make_mimo_inputs(
    batch: int = BATCH,
    seqlen: int = SEQ,
    nheads: int = NHEADS,
    headdim: int = HEADDIM,
    d_state: int = D_STATE,
    R: int = R,
    seed: int = 7,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (x, B, C, dt, alpha, mimo_x, mimo_o) for MIMO forward/backward."""
    torch.manual_seed(seed)
    x = torch.randn(batch, seqlen, nheads, headdim)
    B = torch.randn(batch, seqlen, R, nheads, d_state)
    C = torch.randn(batch, seqlen, R, nheads, d_state)
    dt = torch.rand(batch, seqlen, nheads) * 0.5 + 0.1
    alpha = torch.rand(batch, seqlen, nheads) * 0.5 + 0.4  # in (0.4, 0.9)
    mimo_x = torch.randn(nheads, R, headdim)
    mimo_o = torch.randn(nheads, R, headdim)
    return x, B, C, dt, alpha, mimo_x, mimo_o


# ---------------------------------------------------------------------------
# 1. reference_complex_scan_rope
# ---------------------------------------------------------------------------


class TestComplexScanRope:
    # --- 1a. Shape contracts for two distinct configs ---

    def test_output_shape_default(self) -> None:
        x, B, C, dt, A, angle_proj = _make_rope_inputs()
        y = reference_complex_scan_rope(x, B, C, dt, A, angle_proj)
        assert y.shape == (BATCH, SEQ, NHEADS, HEADDIM), (
            f"Expected {(BATCH, SEQ, NHEADS, HEADDIM)}, got {y.shape}"
        )

    def test_output_shape_alt(self) -> None:
        x, B, C, dt, A, angle_proj = _make_rope_inputs(
            batch=1, seqlen=4, nheads=2, headdim=8, d_state=16, num_rope=4
        )
        y = reference_complex_scan_rope(x, B, C, dt, A, angle_proj)
        assert y.shape == (1, 4, 2, 8)

    # --- 1b. Determinism ---

    def test_determinism(self) -> None:
        args = _make_rope_inputs()
        y1 = reference_complex_scan_rope(*args)
        y2 = reference_complex_scan_rope(*args)
        assert torch.equal(y1, y2), "Non-deterministic output"

    # --- 1c. No NaN / Inf ---

    def test_no_nan_inf(self) -> None:
        args = _make_rope_inputs()
        y = reference_complex_scan_rope(*args)
        assert torch.isfinite(y).all(), "NaN or Inf in complex_scan_rope output"

    # --- 1d. Zero-angle → identity rotation → must match plain real scan ---

    def test_zero_angle_reduces_to_plain_scan(self) -> None:
        """angle_proj = 0 means R_t = I at every step; must equal a no-rotation scan."""
        x, B, C, dt, A, angle_proj = _make_rope_inputs(seed=99)
        angle_zeros = torch.zeros_like(angle_proj)

        y_rope = reference_complex_scan_rope(x, B, C, dt, A, angle_zeros)

        # Hand-rolled plain scan (no rotation): same recurrence but R=I
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (B, L, H)
        h_ref = torch.zeros(batch, nheads, headdim, d_state)
        y_ref = torch.empty_like(x)
        for t in range(seqlen):
            a_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
            B_t = B[:, t, :, :].unsqueeze(2)  # (B,H,1,N)
            x_t = x[:, t, :, :].unsqueeze(-1)  # (B,H,P,1)
            h_ref = a_t * h_ref + dt_t * B_t * x_t
            C_t = C[:, t, :, :].unsqueeze(2)  # (B,H,1,N)
            y_ref[:, t, :, :] = (h_ref * C_t).sum(-1)  # (B,H,P)

        assert torch.allclose(y_rope, y_ref, atol=1e-5), (
            "Zero-angle rope scan does not match plain real scan. "
            f"Max diff: {(y_rope - y_ref).abs().max().item():.3e}"
        )

    # --- 1e. Angle accumulation is causal ---

    def test_angle_causality(self) -> None:
        """Perturbing angle_proj at position t must not change y at positions < t."""
        x, B, C, dt, A, angle_proj = _make_rope_inputs(seqlen=8, seed=13)

        y_base = reference_complex_scan_rope(x, B, C, dt, A, angle_proj)

        # Perturb at t=4 (positions 0..3 must be unaffected)
        angle_perturbed = angle_proj.clone()
        angle_perturbed[:, 4, :, :] += 10.0

        y_perturbed = reference_complex_scan_rope(x, B, C, dt, A, angle_perturbed)

        # Positions 0..3 must be identical
        assert torch.equal(y_base[:, :4, :, :], y_perturbed[:, :4, :, :]), (
            "Angle at t=4 affects output at t<4 — causality violated"
        )
        # Position 4 or later must differ (rotation has changed)
        assert not torch.equal(y_base[:, 4:, :, :], y_perturbed[:, 4:, :, :]), (
            "Angle perturbation at t=4 had no effect on t>=4"
        )

    # --- 1f. Independent ground truth: complex-arithmetic state rotation ---

    def test_matches_complex_arithmetic_scan(self) -> None:
        """Nonzero-angle equivalence against an independent complex formulation.

        The oracle folds the CUMULATIVE rotation into B/C and runs a plain
        decay scan. The mathematically equivalent state-rotation form is
            h_hat_t = alpha_t * exp(-i * dtheta_t) * h_hat_{t-1} + dt_t * b_t * x_t
            y_t     = Re( conj(c_t) . h_hat_t )
        per rotary pair, using UNROTATED b/c and the PER-STEP angle delta
        (change of basis h_hat_t = R(-Theta_t) h_t). Computing that directly
        in complex arithmetic shares no code path with the oracle — it
        catches double-rotation, sign, and angle-compounding bugs that the
        zero-angle test cannot see.
        """
        x, B, C, dt, A, angle_proj = _make_rope_inputs(seed=21)
        y_oracle = reference_complex_scan_rope(x, B, C, dt, A, angle_proj)

        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        num_rope = angle_proj.shape[-1]
        rotary_dim = 2 * num_rope

        alpha = torch.exp(dt * A.unsqueeze(0).unsqueeze(0))  # (B, L, H)
        dtheta = torch.tanh(angle_proj) * dt.unsqueeze(-1) * math.pi  # (B, L, H, K)

        # Complex view of the rotary pairs (2k, 2k+1): b = B_even + i * B_odd
        Bp = B[..., :rotary_dim].reshape(batch, seqlen, nheads, num_rope, 2)
        Cp = C[..., :rotary_dim].reshape(batch, seqlen, nheads, num_rope, 2)
        b = torch.complex(Bp[..., 0], Bp[..., 1])  # (B, L, H, K)
        c = torch.complex(Cp[..., 0], Cp[..., 1])  # (B, L, H, K)
        rot = torch.exp(torch.complex(torch.zeros_like(dtheta), -dtheta))  # e^{-i*dtheta}

        # Passthrough dims (no rotation): plain real scan
        B_pass = B[..., rotary_dim:]  # (B, L, H, N - rot)
        C_pass = C[..., rotary_dim:]

        h_c = torch.zeros(batch, nheads, headdim, num_rope, dtype=torch.complex64)
        h_p = torch.zeros(batch, nheads, headdim, d_state - rotary_dim)
        y_ref = torch.empty_like(x)

        for t in range(seqlen):
            a_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            x_t = x[:, t, :, :].unsqueeze(-1)  # (B, H, P, 1)

            # Rotary pairs: complex decay-and-rotate recurrence
            rot_t = rot[:, t, :, :].unsqueeze(2)  # (B, H, 1, K)
            b_t = b[:, t, :, :].unsqueeze(2)  # (B, H, 1, K)
            h_c = a_t * rot_t * h_c + dt_t * b_t * x_t  # (B, H, P, K)
            c_t = c[:, t, :, :].unsqueeze(2)  # (B, H, 1, K)
            y_rot = (c_t.conj() * h_c).real.sum(-1)  # (B, H, P)

            # Passthrough dims: plain real recurrence
            Bp_t = B_pass[:, t, :, :].unsqueeze(2)  # (B, H, 1, N-rot)
            h_p = a_t * h_p + dt_t * Bp_t * x_t  # (B, H, P, N-rot)
            Cp_t = C_pass[:, t, :, :].unsqueeze(2)
            y_pass = (h_p * Cp_t).sum(-1)  # (B, H, P)

            y_ref[:, t, :, :] = y_rot + y_pass

        assert torch.allclose(y_oracle, y_ref, rtol=1e-4, atol=1e-5), (
            "Oracle disagrees with independent complex-arithmetic scan. "
            f"Max diff: {(y_oracle - y_ref).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# 2. reference_mimo_forward
# ---------------------------------------------------------------------------


class TestMimoForward:
    # --- 2a. Shape contracts ---

    def test_output_shape_default(self) -> None:
        args = _make_mimo_inputs()
        y = reference_mimo_forward(*args)
        assert y.shape == (BATCH, SEQ, NHEADS, HEADDIM)

    def test_output_shape_alt(self) -> None:
        args = _make_mimo_inputs(batch=1, seqlen=4, nheads=2, headdim=6, d_state=8, R=3)
        y = reference_mimo_forward(*args)
        assert y.shape == (1, 4, 2, 6)

    # --- 2b. Determinism ---

    def test_determinism(self) -> None:
        args = _make_mimo_inputs()
        y1 = reference_mimo_forward(*args)
        y2 = reference_mimo_forward(*args)
        assert torch.equal(y1, y2)

    # --- 2c. No NaN / Inf ---

    def test_no_nan_inf(self) -> None:
        args = _make_mimo_inputs()
        y = reference_mimo_forward(*args)
        assert torch.isfinite(y).all()

    # --- 2d. R=1, mimo_x=ones, mimo_o=ones → must match a hand-rolled SISO scan ---

    def test_r1_ones_reduces_to_siso(self) -> None:
        """With R=1 and all-ones mix weights, MIMO collapses to SISO."""
        batch, seqlen, nheads, headdim, d_state = 2, 6, 2, 4, 8
        torch.manual_seed(55)
        x = torch.randn(batch, seqlen, nheads, headdim)
        B = torch.randn(batch, seqlen, 1, nheads, d_state)  # R=1
        C = torch.randn(batch, seqlen, 1, nheads, d_state)
        dt = torch.rand(batch, seqlen, nheads) * 0.5 + 0.1
        alpha = torch.rand(batch, seqlen, nheads) * 0.5 + 0.4
        mimo_x = torch.ones(nheads, 1, headdim)  # psi = 1
        mimo_o = torch.ones(nheads, 1, headdim)  # phi = 1

        y_mimo = reference_mimo_forward(x, B, C, dt, alpha, mimo_x, mimo_o)

        # Hand-rolled SISO: h_t = alpha_t * h_{t-1} + dt_t * B_t * x_t
        # (R=1, psi=phi=1 → x_r = x, y = C^T h)
        h_siso = torch.zeros(batch, nheads, headdim, d_state)
        y_siso = torch.empty_like(x)
        for t in range(seqlen):
            a_t = alpha[:, t, :].unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
            dt_t = dt[:, t, :].unsqueeze(-1).unsqueeze(-1)
            # B[:,t,0,:,:] is (batch, nheads, d_state)
            B_t = B[:, t, 0, :, :].unsqueeze(2)  # (B,H,1,N)
            x_t = x[:, t, :, :].unsqueeze(-1)  # (B,H,P,1)
            h_siso = a_t * h_siso + dt_t * B_t * x_t  # (B,H,P,N)
            C_t = C[:, t, 0, :, :].unsqueeze(2)  # (B,H,1,N)
            y_siso[:, t, :, :] = (h_siso * C_t).sum(-1)  # (B,H,P)

        assert torch.allclose(y_mimo, y_siso, atol=1e-5), (
            f"R=1 MIMO (ones weights) differs from SISO. "
            f"Max diff: {(y_mimo - y_siso).abs().max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# 3. reference_mimo_backward
# ---------------------------------------------------------------------------


class TestMimoBackward:
    # --- 3a. Returns MimoGrads named tuple ---

    def test_returns_named_tuple(self) -> None:
        args = _make_mimo_inputs()
        dy = torch.ones(BATCH, SEQ, NHEADS, HEADDIM)
        result = reference_mimo_backward(*args, dy)
        assert isinstance(result, MimoGrads)

    # --- 3b. Grad shapes ---

    def test_grad_shapes(self) -> None:
        x, B, C, dt, alpha, mimo_x, mimo_o = _make_mimo_inputs()
        dy = torch.ones(BATCH, SEQ, NHEADS, HEADDIM)
        g = reference_mimo_backward(x, B, C, dt, alpha, mimo_x, mimo_o, dy)
        assert g.grad_x.shape == x.shape, f"grad_x: {g.grad_x.shape} vs {x.shape}"
        assert g.grad_B.shape == B.shape, f"grad_B: {g.grad_B.shape} vs {B.shape}"
        assert g.grad_C.shape == C.shape, f"grad_C: {g.grad_C.shape} vs {C.shape}"
        assert g.grad_dt.shape == dt.shape, f"grad_dt: {g.grad_dt.shape} vs {dt.shape}"
        assert g.grad_alpha.shape == alpha.shape
        assert g.grad_mimo_x.shape == mimo_x.shape
        assert g.grad_mimo_o.shape == mimo_o.shape

    # --- 3c. No NaN / Inf in grads ---

    def test_no_nan_inf_in_grads(self) -> None:
        args = _make_mimo_inputs()
        dy = torch.ones(BATCH, SEQ, NHEADS, HEADDIM)
        g = reference_mimo_backward(*args, dy)
        for field in g._fields:
            tensor = getattr(g, field)
            assert torch.isfinite(tensor).all(), f"NaN/Inf in {field}"

    # --- 3d. Grads non-zero ---

    def test_grads_nonzero(self) -> None:
        args = _make_mimo_inputs()
        dy = torch.ones(BATCH, SEQ, NHEADS, HEADDIM)
        g = reference_mimo_backward(*args, dy)
        for field in g._fields:
            tensor = getattr(g, field)
            assert tensor.abs().sum() > 0, f"All-zero gradient for {field}"

    # --- 3e. gradcheck on tiny float64 case ---

    def test_gradcheck_float64(self) -> None:
        """Numerical gradient check on a tiny double-precision instance."""
        # Very small shapes to keep gradcheck fast
        batch, seqlen, nheads, headdim, d_state, r = 1, 3, 2, 2, 4, 1

        torch.manual_seed(0)
        x64 = torch.randn(batch, seqlen, nheads, headdim, dtype=torch.float64, requires_grad=True)
        B64 = torch.randn(
            batch, seqlen, r, nheads, d_state, dtype=torch.float64, requires_grad=True
        )
        C64 = torch.randn(
            batch, seqlen, r, nheads, d_state, dtype=torch.float64, requires_grad=True
        )
        dt64 = (torch.rand(batch, seqlen, nheads, dtype=torch.float64) * 0.5 + 0.1).requires_grad_(
            True
        )
        alpha64 = (
            torch.rand(batch, seqlen, nheads, dtype=torch.float64) * 0.4 + 0.4
        ).requires_grad_(True)
        mimo_x64 = torch.randn(nheads, r, headdim, dtype=torch.float64, requires_grad=True)
        mimo_o64 = torch.randn(nheads, r, headdim, dtype=torch.float64, requires_grad=True)

        # gradcheck validates the REAL oracle (it accepts float64 exactly
        # for this purpose) — never a duplicated copy of the forward.
        assert torch.autograd.gradcheck(
            reference_mimo_forward,
            (x64, B64, C64, dt64, alpha64, mimo_x64, mimo_o64),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-3,
            raise_exception=True,
        )
