"""Channel-wise GDN-2 chunkwise reference vs the token-serial oracle + scalar reduction."""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.references.gdn2_chunkwise import (
    chunkwise_backward,
    chunkwise_forward,
)
from lethe.kernels.references.gdn2_chunkwise_cw import (
    build_microgate_bundles_cw,
    chunkwise_backward_cw,
    chunkwise_forward_cw,
)
from lethe.kernels.references.gdn_backward import (
    reference_gdn2_backward,
    reference_gdn2_forward,
)

SHAPES = [
    (2, 32, 2, 16, 16, 16),  # B, T, H, d_k, d_v, chunk_len  (NT=2)
    (1, 32, 3, 16, 16, 8),  # NT=4
    (2, 48, 2, 24, 20, 16),  # d_k != d_v, NT=3
    (1, 64, 4, 32, 32, 64),  # single chunk
]


def _cw_inputs(shape, seed=0, dtype=torch.float64):
    b, t, h, d_k, d_v, _ = shape
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.3 + 0.02)  # per-channel <0
    b_gate = torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.8 + 0.1
    w_gate = torch.rand(b, t, h, d_v, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, b_gate, w_gate, do


def _l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_oracle(shape):
    q, k, v, g, b_gate, w_gate, _ = _cw_inputs(shape, seed=1)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn = _l2norm(q), _l2norm(k)
    o_oracle = reference_gdn2_forward(qn, kn, v, g, b_gate, w_gate, scale=s, use_qk_l2norm=False)
    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    assert fwd.o.shape == o_oracle.shape
    assert torch.allclose(fwd.o, o_oracle, rtol=1e-9, atol=1e-9), (
        f"channel-wise forward != oracle, max {(fwd.o - o_oracle).abs().max():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_final_grads_match_oracle(shape):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=2)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn = _l2norm(q), _l2norm(k)
    grads = reference_gdn2_backward(qn, kn, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=False)
    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd = chunkwise_backward_cw(fwd, do)

    assert torch.allclose(bwd.dq, grads.grad_q, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dk, grads.grad_k, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dv, grads.grad_v, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dg, grads.grad_g, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.db, grads.grad_b, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dw, grads.grad_w, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_reduces_to_scalar_forward(shape):
    """Channel-constant g + b = w = beta reproduces the scalar chunkwise forward (fp64)."""
    b, t, h, d_k, d_v, cl = shape
    gen = torch.Generator().manual_seed(11)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=torch.float64)
    g_scalar = -(torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.3 + 0.02)
    beta = torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    s = d_k**-0.5
    qn, kn = _l2norm(q), _l2norm(k)

    fwd_s = chunkwise_forward(qn, kn, v, g_scalar, beta, chunk_len=cl, scale=s)
    g_chan = g_scalar.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    b_gate = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
    fwd_c = chunkwise_forward_cw(qn, kn, v, g_chan, b_gate, w_gate, chunk_len=cl, scale=s)

    assert torch.allclose(fwd_c.o, fwd_s.o, rtol=1e-10, atol=1e-10), (
        f"reduction forward mismatch, max {(fwd_c.o - fwd_s.o).abs().max():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_reduces_to_scalar_grads(shape):
    """Reduction: channel-wise final grads collapse to the scalar path's grads (fp64)."""
    b, t, h, d_k, d_v, cl = shape
    gen = torch.Generator().manual_seed(12)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=torch.float64)
    g_scalar = -(torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.3 + 0.02)
    beta = torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=torch.float64)
    s = d_k**-0.5
    qn, kn = _l2norm(q), _l2norm(k)

    fwd_s = chunkwise_forward(qn, kn, v, g_scalar, beta, chunk_len=cl, scale=s)
    bwd_s = chunkwise_backward(fwd_s, do)

    g_chan = g_scalar.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    b_gate = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
    fwd_c = chunkwise_forward_cw(qn, kn, v, g_chan, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd_c = chunkwise_backward_cw(fwd_c, do)

    assert torch.allclose(bwd_c.dq, bwd_s.dq, rtol=1e-9, atol=1e-9)
    assert torch.allclose(bwd_c.dk, bwd_s.dk, rtol=1e-9, atol=1e-9)
    assert torch.allclose(bwd_c.dv, bwd_s.dv, rtol=1e-9, atol=1e-9)
    # scalar dg == sum over key channels of channel-wise dg
    assert torch.allclose(bwd_c.dg.sum(-1), bwd_s.dg, rtol=1e-9, atol=1e-9)
    # scalar db == sum over key channels of db + sum over value channels of dw
    db_scalar = bwd_c.db.sum(-1) + bwd_c.dw.sum(-1)
    assert torch.allclose(db_scalar, bwd_s.db, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("shape", SHAPES)
def test_k1_b4_recurrence_self_consistency(shape):
    """dv2 == dv_local + ((k (.) decay_end) @ dh): the channel-wise K#1 contract math."""
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=3)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn = _l2norm(q), _l2norm(k)
    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd = chunkwise_backward_cw(fwd, do)

    decay_end = torch.exp2(fwd.g_last[..., None, :] - fwd.g2)  # [B,H,NT,C,d_k]
    inter = (fwd.k * decay_end) @ bwd.dh  # [B,H,NT,C,d_v]
    assert torch.allclose(bwd.dv2, bwd.dv_local + inter, rtol=1e-8, atol=1e-8), (
        f"channel-wise B4 recurrence broken, max {(bwd.dv2 - bwd.dv_local - inter).abs().max():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_k2_path_complete_grads(shape):
    """v and w flow only through B1, so dv_final/dw_wy must equal the totals."""
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=4)
    b, t, h, d_k, d_v, cl = shape
    s = d_k**-0.5
    qn, kn = _l2norm(q), _l2norm(k)
    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd = chunkwise_backward_cw(fwd, do)

    dv_total = bwd.dv_final.reshape(b, h, t, d_v).transpose(1, 2)
    dw_total = bwd.dw_wy.reshape(b, h, t, d_v).transpose(1, 2)
    assert torch.allclose(bwd.dv, dv_total, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dw, dw_total, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_intermediates_finite_and_shaped(shape):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=5)
    b, t, h, d_k, d_v, cl = shape
    nt = t // cl
    s = d_k**-0.5
    qn, kn = _l2norm(q), _l2norm(k)
    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd = chunkwise_backward_cw(fwd, do)

    assert fwd.T.shape == (b, h, nt, cl, cl)
    assert fwd.h.shape == (b, h, nt + 1, d_k, d_v)
    assert fwd.gamma.shape == (b, h, nt, cl, d_k)
    assert bwd.dh.shape == (b, h, nt, d_k, d_v)
    assert bwd.dh0_state.shape == (b, h, d_k, d_v)
    assert bwd.dv2.shape == (b, h, nt, cl, d_v)
    assert bwd.dwy.shape == (b, h, nt, cl, d_k)
    assert bwd.dk2.shape == (b, h, nt, cl, d_k)
    assert bwd.db_wy.shape == (b, h, nt, cl, d_k)
    assert bwd.dw_wy.shape == (b, h, nt, cl, d_v)
    assert bwd.dg2.shape == (b, h, nt, cl, d_k)
    for name in ("dh", "dh0_state", "dv2", "dv_local", "dwy", "dk2", "dv_final", "db_wy", "dw_wy"):
        t_ = getattr(bwd, name)
        assert torch.isfinite(t_).all(), f"non-finite in {name}"


@pytest.mark.parametrize("shape", SHAPES)
def test_microgate_bundles(shape):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=7)
    b, t, h, d_k, d_v, cl = shape
    nt = t // cl
    s = d_k**-0.5
    qn, kn = _l2norm(q), _l2norm(k)
    bundles = build_microgate_bundles_cw(qn, kn, v, g, b_gate, w_gate, do, chunk_len=cl, scale=s)

    k1, k2 = bundles["k1"], bundles["k2"]
    assert set(k1.inputs) == {"q", "k", "wy", "g2", "g_last", "do", "dv_local", "h0", "dht"}
    assert set(k1.expected) == {"dh", "dh0", "dv2"}
    assert k1.expected["dh"].shape == (b, h, nt, d_k, d_v)
    assert k1.expected["dv2"].shape == (b, h, nt, cl, d_v)
    assert set(k2.inputs) == {"k", "v", "b", "w", "g2", "T", "dwy", "du"}
    assert set(k2.expected) == {"dk2", "dv", "db", "dw", "dg2"}
    assert k2.expected["dk2"].shape == (b, h, nt, cl, d_k)
    assert k2.expected["dw"].shape == (b, h, nt, cl, d_v)

    fwd = chunkwise_forward_cw(qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s)
    bwd = chunkwise_backward_cw(fwd, do)
    assert torch.equal(k1.expected["dh"], bwd.dh)
    assert torch.equal(k2.expected["dk2"], bwd.dk2)
    for payload in (*k1.inputs.values(), *k1.expected.values(), *k2.inputs.values()):
        assert torch.isfinite(payload).all()


def test_initial_state_dh0():
    shape = (2, 32, 2, 16, 16, 16)
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=6)
    cl = shape[5]
    s = shape[3] ** -0.5
    h0 = torch.randn(shape[0], shape[2], shape[3], shape[4], dtype=torch.float64)
    qn, kn = _l2norm(q), _l2norm(k)
    grads = reference_gdn2_backward(
        qn, kn, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=False, initial_state=h0
    )
    fwd = chunkwise_forward_cw(
        qn, kn, v, g, b_gate, w_gate, chunk_len=cl, scale=s, initial_state=h0
    )
    bwd = chunkwise_backward_cw(fwd, do)
    assert grads.grad_initial_state is not None
    assert bwd.dh0 is not None
    assert torch.allclose(bwd.dh0, grads.grad_initial_state, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dh0_state, grads.grad_initial_state, rtol=1e-8, atol=1e-8)
