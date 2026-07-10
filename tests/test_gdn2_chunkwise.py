"""Chunkwise GDN (Phase-2) reference vs the verified token-serial oracle."""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.references.gdn2_chunkwise import (
    build_microgate_bundles,
    chunkwise_backward,
    chunkwise_forward,
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


def _scalar_inputs(shape, seed=0, dtype=torch.float64):
    b, t, h, d_k, d_v, _ = shape
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = -(torch.rand(b, t, h, generator=gen, dtype=dtype) * 0.3 + 0.02)  # log-decay < 0
    beta = torch.rand(b, t, h, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, beta, do


def _l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _oracle_args(q, k, v, g, beta):
    """Pre-normed q/k + channel-broadcast g/b/w for the scalar reduction."""
    b, t, h, d_k = q.shape
    d_v = v.shape[-1]
    qn, kn = _l2norm(q), _l2norm(k)
    g_chan = g.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    b_gate = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
    return qn, kn, g_chan, b_gate, w_gate


@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_oracle(shape):
    q, k, v, g, beta, _ = _scalar_inputs(shape, seed=1)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn, g_chan, b_gate, w_gate = _oracle_args(q, k, v, g, beta)
    o_oracle = reference_gdn2_forward(
        qn, kn, v, g_chan, b_gate, w_gate, scale=s, use_qk_l2norm=False
    )
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False)
    assert fwd.o.shape == o_oracle.shape
    assert torch.allclose(fwd.o, o_oracle, rtol=1e-9, atol=1e-9), (
        f"chunkwise forward != oracle, max {(fwd.o - o_oracle).abs().max():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_final_grads_match_oracle(shape):
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=2)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn, g_chan, b_gate, w_gate = _oracle_args(q, k, v, g, beta)
    grads = reference_gdn2_backward(
        qn, kn, v, g_chan, b_gate, w_gate, do, scale=s, use_qk_l2norm=False
    )
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False)
    bwd = chunkwise_backward(fwd, do)

    assert torch.allclose(bwd.dq, grads.grad_q, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dk, grads.grad_k, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dv, grads.grad_v, rtol=1e-8, atol=1e-8)
    # scalar g/beta reduce by summing the oracle's channel-wise grads.
    assert torch.allclose(bwd.dg, grads.grad_g.sum(-1), rtol=1e-8, atol=1e-8)
    db_oracle = grads.grad_b.sum(-1) + grads.grad_w.sum(-1)
    assert torch.allclose(bwd.db, db_oracle, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_k1_b4_recurrence_self_consistency(shape):
    """dv2 == dv_local + exp2(g_last - g2) * (k @ dh): the K#1 contract math."""
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=3)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn, _, _, _ = _oracle_args(q, k, v, g, beta)
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False)
    bwd = chunkwise_backward(fwd, do)

    inter = torch.exp2(fwd.g_last[..., None] - fwd.g2)[..., None] * (fwd.k @ bwd.dh)
    assert torch.allclose(bwd.dv2, bwd.dv_local + inter, rtol=1e-8, atol=1e-8), (
        f"B4 recurrence broken, max {(bwd.dv2 - bwd.dv_local - inter).abs().max():.3e}"
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_k2_path_complete_grads(shape):
    """v and beta flow only through B1, so dv_final/db_wy must equal the totals."""
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=4)
    b, t, h, d_k, d_v, cl = shape
    s = d_k**-0.5
    qn, kn, _, _, _ = _oracle_args(q, k, v, g, beta)
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False)
    bwd = chunkwise_backward(fwd, do)

    dv_total = bwd.dv_final.reshape(b, h, t, d_v).transpose(1, 2)
    db_total = bwd.db_wy.reshape(b, h, t).transpose(1, 2)
    assert torch.allclose(bwd.dv, dv_total, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.db, db_total, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_intermediates_finite_and_shaped(shape):
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=5)
    b, t, h, d_k, d_v, cl = shape
    nt = t // cl
    s = d_k**-0.5
    qn, kn, _, _, _ = _oracle_args(q, k, v, g, beta)
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False)
    bwd = chunkwise_backward(fwd, do)

    assert fwd.T.shape == (b, h, nt, cl, cl)
    assert fwd.h.shape == (b, h, nt + 1, d_k, d_v)
    assert bwd.dh.shape == (b, h, nt, d_k, d_v)
    assert bwd.dh0_state.shape == (b, h, d_k, d_v)
    assert bwd.dv2.shape == (b, h, nt, cl, d_v)
    assert bwd.dw.shape == (b, h, nt, cl, d_k)
    assert bwd.dk2.shape == (b, h, nt, cl, d_k)
    assert bwd.dg2.shape == (b, h, nt, cl)
    for name in ("dh", "dh0_state", "dv2", "dv_local", "dw", "dk2", "dv_final", "db_wy", "dg2"):
        t_ = getattr(bwd, name)
        assert torch.isfinite(t_).all(), f"non-finite in {name}"


@pytest.mark.parametrize("shape", SHAPES)
def test_microgate_bundles(shape):
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=7)
    b, t, h, d_k, d_v, cl = shape
    nt = t // cl
    s = d_k**-0.5
    qn, kn, _, _, _ = _oracle_args(q, k, v, g, beta)
    bundles = build_microgate_bundles(qn, kn, v, g, beta, do, chunk_len=cl, scale=s)

    k1, k2 = bundles["k1"], bundles["k2"]
    assert set(k1.inputs) == {"q", "k", "w", "g2", "g_last", "do", "dv_local", "h0", "dht"}
    assert set(k1.expected) == {"dh", "dh0", "dv2"}
    assert k1.expected["dh"].shape == (b, h, nt, d_k, d_v)
    assert k1.expected["dv2"].shape == (b, h, nt, cl, d_v)
    assert set(k2.inputs) == {"k", "v", "beta", "g2", "T", "dw", "du"}
    assert set(k2.expected) == {"dk2", "dv", "db", "dg2"}
    assert k2.expected["dk2"].shape == (b, h, nt, cl, d_k)

    # bundle expected outputs must equal a fresh reference pass.
    fwd = chunkwise_forward(qn, kn, v, g, beta, chunk_len=cl, scale=s)
    bwd = chunkwise_backward(fwd, do)
    assert torch.equal(k1.expected["dh"], bwd.dh)
    assert torch.equal(k2.expected["dk2"], bwd.dk2)
    for payload in (*k1.inputs.values(), *k1.expected.values(), *k2.inputs.values()):
        assert torch.isfinite(payload).all()


def test_initial_state_dh0():
    shape = (2, 32, 2, 16, 16, 16)
    q, k, v, g, beta, do = _scalar_inputs(shape, seed=6)
    cl = shape[5]
    s = shape[3] ** -0.5
    h0 = torch.randn(shape[0], shape[2], shape[3], shape[4], dtype=torch.float64)
    qn, kn, g_chan, b_gate, w_gate = _oracle_args(q, k, v, g, beta)
    grads = reference_gdn2_backward(
        qn, kn, v, g_chan, b_gate, w_gate, do, scale=s, use_qk_l2norm=False, initial_state=h0
    )
    fwd = chunkwise_forward(
        qn, kn, v, g, beta, chunk_len=cl, scale=s, use_qk_l2norm=False, initial_state=h0
    )
    bwd = chunkwise_backward(fwd, do)
    assert grads.grad_initial_state is not None
    assert bwd.dh0 is not None
    assert torch.allclose(bwd.dh0, grads.grad_initial_state, rtol=1e-8, atol=1e-8)
    assert torch.allclose(bwd.dh0_state, grads.grad_initial_state, rtol=1e-8, atol=1e-8)
