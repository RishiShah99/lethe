"""Tests for the Gated DeltaNet-2 (GDN-2) reference oracle."""

import torch
from torch import Tensor

from lethe.kernels.references import (
    Gdn2Grads,
    reference_gdn2_backward,
    reference_gdn2_forward,
)

BATCH = 2
SEQ = 6
NHEADS = 3
D_K = 4
D_V = 5


def _make_gdn2_inputs(
    batch: int = BATCH,
    seqlen: int = SEQ,
    nheads: int = NHEADS,
    d_k: int = D_K,
    d_v: int = D_V,
    seed: int = 7,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (q, k, v, g, b, w) for GDN-2 forward/backward."""
    torch.manual_seed(seed)
    q = torch.randn(batch, seqlen, nheads, d_k, dtype=dtype)
    k = torch.randn(batch, seqlen, nheads, d_k, dtype=dtype)
    v = torch.randn(batch, seqlen, nheads, d_v, dtype=dtype)
    g = -(torch.rand(batch, seqlen, nheads, d_k, dtype=dtype) * 0.3 + 0.05)  # log-decay < 0
    b = torch.randn(batch, seqlen, nheads, d_k, dtype=dtype).sigmoid()
    w = torch.randn(batch, seqlen, nheads, d_v, dtype=dtype).sigmoid()
    return q, k, v, g, b, w


def _scalar_gated_delta_forward(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, *, scale: float
) -> Tensor:
    """Independent scalar gated-delta-rule scan (fla naive.py math)."""
    qh, kh, vh, gh = (t.transpose(1, 2).to(torch.float64) for t in (q, k, v, g))
    betah = beta.transpose(1, 2).to(torch.float64)
    batch, nheads, seqlen, d_k = kh.shape
    d_v = vh.shape[-1]
    qh = qh / torch.sqrt((qh * qh).sum(-1, keepdim=True) + 1e-6)
    kh = kh / torch.sqrt((kh * kh).sum(-1, keepdim=True) + 1e-6)
    qh = qh * scale
    h = torch.zeros(batch, nheads, d_k, d_v, dtype=torch.float64)
    out = []
    for t in range(seqlen):
        h = h * gh[:, :, t].exp().unsqueeze(-1)
        read = (h * kh[:, :, t].unsqueeze(-1)).sum(-2)
        v_new = betah[:, :, t].unsqueeze(-1) * (vh[:, :, t] - read)
        h = h + kh[:, :, t].unsqueeze(-1) * v_new.unsqueeze(-2)
        out.append((h * qh[:, :, t].unsqueeze(-1)).sum(-2))
    return torch.stack(out, dim=2).transpose(1, 2).contiguous()


class TestGdn2Forward:
    def test_output_shape_default(self) -> None:
        o = reference_gdn2_forward(*_make_gdn2_inputs())
        assert o.shape == (BATCH, SEQ, NHEADS, D_V)

    def test_output_shape_alt(self) -> None:
        o = reference_gdn2_forward(*_make_gdn2_inputs(batch=1, seqlen=4, nheads=2, d_k=8, d_v=6))
        assert o.shape == (1, 4, 2, 6)

    def test_determinism(self) -> None:
        args = _make_gdn2_inputs()
        assert torch.equal(reference_gdn2_forward(*args), reference_gdn2_forward(*args))

    def test_no_nan_inf(self) -> None:
        o = reference_gdn2_forward(*_make_gdn2_inputs())
        assert torch.isfinite(o).all()

    def test_reduces_to_scalar_gated_delta(self) -> None:
        """b = w = beta*1 must reproduce the scalar gated delta rule exactly."""
        q, k, v, g, _, _ = _make_gdn2_inputs(seed=31, dtype=torch.float64)
        beta = torch.rand(BATCH, SEQ, NHEADS, dtype=torch.float64) * 0.8 + 0.1
        b = beta.unsqueeze(-1).expand(BATCH, SEQ, NHEADS, D_K).contiguous()
        w = beta.unsqueeze(-1).expand(BATCH, SEQ, NHEADS, D_V).contiguous()
        scale = D_K**-0.5

        o_gdn2 = reference_gdn2_forward(q, k, v, g, b, w, scale=scale)
        o_scalar = _scalar_gated_delta_forward(q, k, v, g, beta, scale=scale)

        assert torch.allclose(o_gdn2, o_scalar, rtol=1e-9, atol=1e-9), (
            "b=w=beta GDN-2 differs from scalar gated delta rule. "
            f"Max diff: {(o_gdn2 - o_scalar).abs().max().item():.3e}"
        )

    def test_initial_state_carries(self) -> None:
        """A nonzero initial state must change the output."""
        q, k, v, g, b, w = _make_gdn2_inputs()
        h0 = torch.randn(BATCH, NHEADS, D_K, D_V)
        o_zero = reference_gdn2_forward(q, k, v, g, b, w)
        o_init = reference_gdn2_forward(q, k, v, g, b, w, initial_state=h0)
        assert not torch.equal(o_zero, o_init)


class TestGdn2Backward:
    def test_returns_named_tuple(self) -> None:
        args = _make_gdn2_inputs()
        do = torch.ones(BATCH, SEQ, NHEADS, D_V)
        assert isinstance(reference_gdn2_backward(*args, do), Gdn2Grads)

    def test_grad_shapes(self) -> None:
        q, k, v, g, b, w = _make_gdn2_inputs()
        do = torch.ones(BATCH, SEQ, NHEADS, D_V)
        grads = reference_gdn2_backward(q, k, v, g, b, w, do)
        assert grads.grad_q.shape == q.shape
        assert grads.grad_k.shape == k.shape
        assert grads.grad_v.shape == v.shape
        assert grads.grad_g.shape == g.shape
        assert grads.grad_b.shape == b.shape
        assert grads.grad_w.shape == w.shape
        assert grads.grad_initial_state is None

    def test_grad_initial_state_present(self) -> None:
        q, k, v, g, b, w = _make_gdn2_inputs()
        h0 = torch.randn(BATCH, NHEADS, D_K, D_V)
        do = torch.ones(BATCH, SEQ, NHEADS, D_V)
        grads = reference_gdn2_backward(q, k, v, g, b, w, do, initial_state=h0)
        assert grads.grad_initial_state is not None
        assert grads.grad_initial_state.shape == h0.shape

    def test_no_nan_inf_in_grads(self) -> None:
        args = _make_gdn2_inputs()
        do = torch.ones(BATCH, SEQ, NHEADS, D_V)
        grads = reference_gdn2_backward(*args, do)
        for field in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
            assert torch.isfinite(getattr(grads, field)).all(), f"NaN/Inf in {field}"

    def test_grads_nonzero(self) -> None:
        args = _make_gdn2_inputs()
        do = torch.ones(BATCH, SEQ, NHEADS, D_V)
        grads = reference_gdn2_backward(*args, do)
        for field in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
            assert getattr(grads, field).abs().sum() > 0, f"All-zero gradient for {field}"

    def test_gradcheck_float64(self) -> None:
        batch, seqlen, nheads, d_k, d_v = 1, 3, 2, 3, 2
        torch.manual_seed(0)
        q = torch.randn(batch, seqlen, nheads, d_k, dtype=torch.float64, requires_grad=True)
        k = torch.randn(batch, seqlen, nheads, d_k, dtype=torch.float64, requires_grad=True)
        v = torch.randn(batch, seqlen, nheads, d_v, dtype=torch.float64, requires_grad=True)
        g = (
            -(torch.rand(batch, seqlen, nheads, d_k, dtype=torch.float64) * 0.3 + 0.05)
        ).requires_grad_(True)
        b = torch.rand(batch, seqlen, nheads, d_k, dtype=torch.float64).requires_grad_(True)
        w = torch.rand(batch, seqlen, nheads, d_v, dtype=torch.float64).requires_grad_(True)

        assert torch.autograd.gradcheck(
            reference_gdn2_forward,
            (q, k, v, g, b, w),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )
