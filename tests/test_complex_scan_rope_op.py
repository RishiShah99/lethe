"""CPU-side validation of the C4 complex-RoPE scan op (the eager path)."""

from __future__ import annotations

import math

import torch

from lethe.kernels.ops import complex_scan_rope
from lethe.kernels.references.complex_scan_rope import reference_complex_scan_rope


def _rope_inputs(
    b: int = 2,
    seq: int = 6,
    h: int = 2,
    p: int = 3,
    n: int = 8,
    s: int = 3,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    x = torch.randn(b, seq, h, p, dtype=dtype)
    bb = torch.randn(b, seq, h, n, dtype=dtype)
    cc = torch.randn(b, seq, h, n, dtype=dtype)
    dt = torch.rand(b, seq, h, dtype=dtype) * 0.1 + 1e-3
    a = -torch.rand(h, dtype=dtype)
    angle = torch.randn(b, seq, h, s, dtype=dtype)
    return x, bb, cc, dt, a, angle


class TestComplexScanRopeCpu:
    def test_matches_reference_fp32(self) -> None:
        args = _rope_inputs()
        assert torch.equal(complex_scan_rope(*args), reference_complex_scan_rope(*args))

    def test_fp64_native(self) -> None:
        args = tuple(t.to(torch.float64) for t in _rope_inputs())
        got = complex_scan_rope(*args)
        assert got.dtype == torch.float64
        assert torch.equal(got, reference_complex_scan_rope(*args))

    def test_fp16_close_to_fp32_oracle(self) -> None:
        args16 = tuple(t.to(torch.float16) for t in _rope_inputs(seq=8))
        got = complex_scan_rope(*args16)
        want = reference_complex_scan_rope(*(t.to(torch.float32) for t in args16))
        assert got.dtype == torch.float16
        assert torch.allclose(got.float(), want, atol=1e-2, rtol=1e-2)

    def test_bf16_dtype_preserved(self) -> None:
        args = tuple(t.to(torch.bfloat16) for t in _rope_inputs())
        got = complex_scan_rope(*args)
        assert got.dtype == torch.bfloat16
        assert torch.isfinite(got.float()).all()

    def test_rotary_dim_exceeding_d_state_raises(self) -> None:
        args = _rope_inputs(n=4, s=3)  # rotary_dim 6 > d_state 4
        try:
            complex_scan_rope(*args)
        except ValueError as exc:
            assert "rotary_dim" in str(exc)
        else:
            raise AssertionError("expected ValueError for rotary_dim > d_state")

    def test_closed_form_single_pair(self) -> None:
        # Hand-computed L=2 pin of the full chain, independent of the reference's code path.
        dt1, dt2 = 0.5, 1.0
        a = -1.0
        a1, a2 = 0.3, -0.7
        b1, b2 = (0.8, -0.2), (0.1, 0.9)
        c1, c2 = (0.4, 0.6), (-0.5, 0.3)
        x1, x2 = 1.5, -2.0

        th1 = math.tanh(a1) * dt1 * math.pi
        th2 = th1 + math.tanh(a2) * dt2 * math.pi

        def rot(v: tuple[float, float], th: float) -> tuple[float, float]:
            return (
                v[0] * math.cos(th) - v[1] * math.sin(th),
                v[0] * math.sin(th) + v[1] * math.cos(th),
            )

        br1, br2 = rot(b1, th1), rot(b2, th2)
        cr1, cr2 = rot(c1, th1), rot(c2, th2)
        h1 = (dt1 * br1[0] * x1, dt1 * br1[1] * x1)
        y1 = cr1[0] * h1[0] + cr1[1] * h1[1]
        al2 = math.exp(dt2 * a)
        h2 = (al2 * h1[0] + dt2 * br2[0] * x2, al2 * h1[1] + dt2 * br2[1] * x2)
        y2 = cr2[0] * h2[0] + cr2[1] * h2[1]

        x = torch.tensor([[[[x1]], [[x2]]]], dtype=torch.float64)
        bb = torch.tensor([[[list(b1)], [list(b2)]]], dtype=torch.float64)
        cc = torch.tensor([[[list(c1)], [list(c2)]]], dtype=torch.float64)
        dt = torch.tensor([[[dt1], [dt2]]], dtype=torch.float64)
        a_t = torch.tensor([a], dtype=torch.float64)
        ang = torch.tensor([[[[a1]], [[a2]]]], dtype=torch.float64)

        y = complex_scan_rope(x, bb, cc, dt, a_t, ang)
        assert y.shape == (1, 2, 1, 1)
        assert abs(y[0, 0, 0, 0].item() - y1) < 1e-12
        assert abs(y[0, 1, 0, 0].item() - y2) < 1e-12

    def test_identity_tail_beyond_rotary_dim(self) -> None:
        # Lanes past 2*S scan unrotated; zeroing B/C rotated lanes gives a plain decay scan.
        x, bb, cc, dt, a, angle = _rope_inputs(n=8, s=2, seed=3)
        bb[..., :4] = 0.0
        cc[..., :4] = 0.0
        got = complex_scan_rope(x, bb, cc, dt, a, angle)
        want = reference_complex_scan_rope(x, bb, cc, dt, a, torch.zeros_like(angle))
        assert torch.equal(got, want)

    def test_zero_angles_is_plain_decay_scan(self) -> None:
        # S=0 is contract-legal (2*0 <= N); the op must degenerate to a plain decay scan.
        x, bb, cc, dt, a, angle = (t.to(torch.float64) for t in _rope_inputs(s=0, seed=7))
        got = complex_scan_rope(x, bb, cc, dt, a, angle)
        alpha = torch.exp(dt * a)
        h = torch.zeros(x.shape[0], x.shape[2], x.shape[3], bb.shape[-1], dtype=torch.float64)
        ys = []
        for t in range(x.shape[1]):
            bu = (dt[:, t].unsqueeze(-1) * bb[:, t]).unsqueeze(2) * x[:, t].unsqueeze(-1)
            h = alpha[:, t, :, None, None] * h + bu
            ys.append((h * cc[:, t].unsqueeze(2)).sum(-1))
        want = torch.stack(ys, dim=1)
        assert (got - want).abs().max().item() < 1e-12

    def test_differentiable_for_vjp(self) -> None:
        args = _rope_inputs()
        leaves = tuple(t.detach().requires_grad_(True) for t in args)
        y = complex_scan_rope(*leaves)
        grads = torch.autograd.grad(y.sum(), leaves)
        assert all(torch.isfinite(g).all() for g in grads)

    def test_gradcheck_fp64(self) -> None:
        args = tuple(
            t.to(torch.float64).detach().requires_grad_(True)
            for t in _rope_inputs(b=1, seq=3, h=1, p=2, n=4, s=1)
        )
        assert torch.autograd.gradcheck(complex_scan_rope, args, eps=1e-6, atol=1e-4)

    def test_nonfinite_x_masks_match_oracle(self) -> None:
        x, bb, cc, dt, a, angle = _rope_inputs(seed=5)
        x[0, 1, 0, 0] = float("inf")
        x[1, 3, 1, 2] = float("nan")
        got = complex_scan_rope(x, bb, cc, dt, a, angle)
        want = reference_complex_scan_rope(x, bb, cc, dt, a, angle)
        assert torch.equal(torch.isnan(got), torch.isnan(want))
        assert torch.equal(torch.isinf(got), torch.isinf(want))
