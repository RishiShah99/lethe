"""Channel-wise GDN-2 backward assembly (Phase-3 crown): desk correctness in fp64."""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.cute.gdn2_assemble import (
    assemble_gdn2_backward_channelwise,
    assembled_channelwise_gdn2_backward,
    assembled_scalar_gdn2_backward,
    k1_reverse_state_cw_ref,
    k2_wy_vjp_cw_ref,
)
from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw
from lethe.kernels.references.gdn_backward import reference_gdn2_backward

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
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.3 + 0.02)
    b_gate = torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.8 + 0.1
    w_gate = torch.rand(b, t, h, d_v, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, b_gate, w_gate, do


def _l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


@pytest.mark.parametrize("shape", SHAPES)
def test_k1_cw_ref_matches_bundle(shape):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=1)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn = _l2norm(q), _l2norm(k)
    bundles = build_microgate_bundles_cw(qn, kn, v, g, b_gate, w_gate, do, chunk_len=cl, scale=s)
    inp, exp = bundles["k1"].inputs, bundles["k1"].expected

    dh, dv2, dh0 = k1_reverse_state_cw_ref(
        inp["q"],
        inp["k"],
        inp["wy"],
        inp["g2"],
        inp["g_last"],
        inp["do"],
        inp["dv_local"],
        inp["dht"],
    )
    assert torch.allclose(dh, exp["dh"], rtol=1e-8, atol=1e-8)
    assert torch.allclose(dv2, exp["dv2"], rtol=1e-8, atol=1e-8)
    assert torch.allclose(dh0, exp["dh0"], rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("shape", SHAPES)
def test_k2_cw_ref_matches_bundle(shape):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=2)
    cl = shape[5]
    s = shape[3] ** -0.5
    qn, kn = _l2norm(q), _l2norm(k)
    bundles = build_microgate_bundles_cw(qn, kn, v, g, b_gate, w_gate, do, chunk_len=cl, scale=s)
    inp, exp = bundles["k2"].inputs, bundles["k2"].expected

    dk2, dv, db, dw, dg2 = k2_wy_vjp_cw_ref(
        inp["k"], inp["v"], inp["b"], inp["w"], inp["g2"], inp["T"], inp["dwy"], inp["du"]
    )
    assert torch.allclose(dk2, exp["dk2"], rtol=1e-7, atol=1e-7)
    assert torch.allclose(dv, exp["dv"], rtol=1e-7, atol=1e-7)
    assert torch.allclose(db, exp["db"], rtol=1e-7, atol=1e-7)
    assert torch.allclose(dw, exp["dw"], rtol=1e-7, atol=1e-7)
    assert torch.allclose(dg2, exp["dg2"], rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_assembly_matches_oracle(shape):
    """Full channel-wise assembly (refs) == oracle's six per-channel grads, through L2-norm."""
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=3)
    s = shape[3] ** -0.5
    grads = assemble_gdn2_backward_channelwise(
        q, k, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=True
    )
    oracle = reference_gdn2_backward(q, k, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=True)

    assert torch.allclose(grads.dq, oracle.grad_q, rtol=1e-7, atol=1e-7)
    assert torch.allclose(grads.dk, oracle.grad_k, rtol=1e-7, atol=1e-7)
    assert torch.allclose(grads.dv, oracle.grad_v, rtol=1e-7, atol=1e-7)
    assert torch.allclose(grads.dg, oracle.grad_g, rtol=1e-7, atol=1e-7)
    assert torch.allclose(grads.db, oracle.grad_b, rtol=1e-7, atol=1e-7)
    assert torch.allclose(grads.dw, oracle.grad_w, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("shape", SHAPES)
def test_wrapper_matches_oracle(shape):
    """The GDN-2-signature wrapper returns the oracle's six grads (fp64 path)."""
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=4)
    out = assembled_channelwise_gdn2_backward(q, k, v, g, b_gate, w_gate, do)
    oracle = reference_gdn2_backward(q, k, v, g, b_gate, w_gate, do)
    for got, ref in (
        (out.grad_q, oracle.grad_q),
        (out.grad_k, oracle.grad_k),
        (out.grad_v, oracle.grad_v),
        (out.grad_g, oracle.grad_g),
        (out.grad_b, oracle.grad_b),
        (out.grad_w, oracle.grad_w),
    ):
        assert torch.allclose(got, ref, rtol=1e-7, atol=1e-7)
    assert out.grad_initial_state is None


@pytest.mark.parametrize("shape", SHAPES)
def test_reduction_to_scalar_assembly(shape):
    """b = w = beta, g channel-constant collapses to the Phase-2 scalar assembly."""
    b, t, h, d_k, d_v, _ = shape
    gen = torch.Generator().manual_seed(21)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=torch.float64)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=torch.float64)
    g_scalar = -(torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.3 + 0.02)
    beta = torch.rand(b, t, h, generator=gen, dtype=torch.float64) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=torch.float64)

    g_chan = g_scalar.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    b_gate = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    w_gate = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()

    cw = assembled_channelwise_gdn2_backward(q, k, v, g_chan, b_gate, w_gate, do)
    sc = assembled_scalar_gdn2_backward(q, k, v, g_chan, b_gate, w_gate, do)

    assert torch.allclose(cw.grad_q, sc.grad_q, rtol=1e-9, atol=1e-9)
    assert torch.allclose(cw.grad_k, sc.grad_k, rtol=1e-9, atol=1e-9)
    assert torch.allclose(cw.grad_v, sc.grad_v, rtol=1e-9, atol=1e-9)
    # scalar grads are uniform-spread so their channel-sum is the recoverable quantity.
    assert torch.allclose(cw.grad_g.sum(-1), sc.grad_g.sum(-1), rtol=1e-9, atol=1e-9)
    cw_beta = cw.grad_b.sum(-1) + cw.grad_w.sum(-1)
    sc_beta = sc.grad_b.sum(-1) + sc.grad_w.sum(-1)
    assert torch.allclose(cw_beta, sc_beta, rtol=1e-9, atol=1e-9)


def test_injected_kernels_are_used():
    """k1_fn/k2_fn are dispatched (assembly is kernel-swappable)."""
    shape = (1, 32, 2, 16, 16, 16)
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=5)
    s = shape[3] ** -0.5
    calls = {"k1": 0, "k2": 0}

    def k1_spy(*args):
        calls["k1"] += 1
        return k1_reverse_state_cw_ref(*args)

    def k2_spy(*args):
        calls["k2"] += 1
        return k2_wy_vjp_cw_ref(*args)

    grads = assemble_gdn2_backward_channelwise(
        q, k, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=True, k1_fn=k1_spy, k2_fn=k2_spy
    )
    oracle = reference_gdn2_backward(q, k, v, g, b_gate, w_gate, do, scale=s, use_qk_l2norm=True)
    assert calls == {"k1": 1, "k2": 1}
    assert torch.allclose(grads.dq, oracle.grad_q, rtol=1e-7, atol=1e-7)
