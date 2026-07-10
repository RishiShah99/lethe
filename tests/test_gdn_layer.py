"""Autograd binding for GDN-2 (gdn_layer.gdn2_op)."""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.ops.gdn_layer import gdn2_op
from lethe.kernels.references.gdn_backward import reference_gdn2_backward


def _inputs(seed: int = 0, dtype: torch.dtype = torch.float64):
    b, t, h, dk, dv = 2, 32, 3, 16, 16
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


def test_backward_matches_reference_fp64() -> None:
    """gdn2_op autograd grads == reference_gdn2_backward (fp64, eager path, bitwise)."""
    q, k, v, g, b, w, do = _inputs(seed=0)

    q_leaf = q.detach().requires_grad_(True)
    k_leaf = k.detach().requires_grad_(True)
    v_leaf = v.detach().requires_grad_(True)
    g_leaf = g.detach().requires_grad_(True)
    b_leaf = b.detach().requires_grad_(True)
    w_leaf = w.detach().requires_grad_(True)

    o = gdn2_op(q_leaf, k_leaf, v_leaf, g_leaf, b_leaf, w_leaf)
    o.backward(do)

    ref = reference_gdn2_backward(q, k, v, g, b, w, do)

    for name, got, want in (
        ("grad_q", q_leaf.grad, ref.grad_q),
        ("grad_k", k_leaf.grad, ref.grad_k),
        ("grad_v", v_leaf.grad, ref.grad_v),
        ("grad_g", g_leaf.grad, ref.grad_g),
        ("grad_b", b_leaf.grad, ref.grad_b),
        ("grad_w", w_leaf.grad, ref.grad_w),
    ):
        assert got is not None, f"{name} is None"
        # fp64 eager path delegates to the same oracle, grads must be bitwise equal.
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, msg=f"{name} mismatch")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shape_and_dtype_preserved(dtype: torch.dtype) -> None:
    """Output shape and dtype match the input convention."""
    q, k, v, g, b, w, _do = _inputs(seed=1, dtype=dtype)
    b_in, t, h, dv = q.shape[0], q.shape[1], q.shape[2], v.shape[-1]
    o = gdn2_op(q, k, v, g, b, w)
    assert o.shape == (b_in, t, h, dv), f"shape mismatch: {o.shape}"
    assert o.dtype == dtype, f"dtype mismatch: {o.dtype}"


def test_no_grad_inputs_produce_none_grads() -> None:
    """requires_grad=False inputs get .grad=None; grad-requiring ones get populated grads."""
    q, k, v, g, b, w, do = _inputs(seed=2)

    # q and v require grad; k, g, b, w do not.
    q_leaf = q.detach().requires_grad_(True)
    k_leaf = k.detach().requires_grad_(False)
    v_leaf = v.detach().requires_grad_(True)
    g_leaf = g.detach().requires_grad_(False)
    b_leaf = b.detach().requires_grad_(False)
    w_leaf = w.detach().requires_grad_(False)

    o = gdn2_op(q_leaf, k_leaf, v_leaf, g_leaf, b_leaf, w_leaf)
    o.backward(do)

    # Grad-requiring inputs must have finite, populated gradients.
    assert q_leaf.grad is not None, "q_leaf.grad should be populated"
    assert torch.isfinite(q_leaf.grad).all(), "q_leaf.grad has non-finite values"
    assert v_leaf.grad is not None, "v_leaf.grad should be populated"
    assert torch.isfinite(v_leaf.grad).all(), "v_leaf.grad has non-finite values"

    # Non-grad-requiring inputs must have .grad=None (standard autograd contract).
    assert k_leaf.grad is None, "k_leaf.grad should be None (requires_grad=False)"
    assert g_leaf.grad is None, "g_leaf.grad should be None (requires_grad=False)"
    assert b_leaf.grad is None, "b_leaf.grad should be None (requires_grad=False)"
    assert w_leaf.grad is None, "w_leaf.grad should be None (requires_grad=False)"


def test_native_route_runs_under_function_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native assembly works inside Function.backward's no-grad context."""
    import lethe.kernels.cute.gdn2_backward as gdn2_native
    from lethe.kernels.cute.gdn2_assemble import (
        k1_reverse_state_cw_ref,
        k2_wy_vjp_cw_ref,
    )

    bsz, t, h, d_k, d_v = 1, 64, 1, 128, 64
    gen = torch.Generator().manual_seed(5)
    dt = torch.float32
    q = torch.randn(bsz, t, h, d_k, generator=gen, dtype=dt, requires_grad=True)
    k = torch.randn(bsz, t, h, d_k, generator=gen, dtype=dt)
    v = torch.randn(bsz, t, h, d_v, generator=gen, dtype=dt)
    g = -torch.rand(bsz, t, h, d_k, generator=gen, dtype=dt) * 0.1
    b = torch.rand(bsz, t, h, d_k, generator=gen, dtype=dt)
    w = torch.rand(bsz, t, h, d_v, generator=gen, dtype=dt)
    do = torch.randn(bsz, t, h, d_v, generator=gen, dtype=dt)

    monkeypatch.setattr(gdn2_native, "is_available", lambda device=None: True)
    monkeypatch.setattr(
        gdn2_native,
        "_load_box_kernels_cw",
        lambda: (k1_reverse_state_cw_ref, k2_wy_vjp_cw_ref),
    )
    # native route is gated on q.is_cuda; bypass the check so CPU refs hit Function.backward.
    import lethe.kernels.ops.gdn_backward as op_mod

    real = op_mod.gdn2_backward

    def _force_native(q2, k2, v2, g2, b2, w2, do2, **kw):  # type: ignore[no-untyped-def]
        native = gdn2_native.native_gdn2_backward(q2, k2, v2, g2, b2, w2, do2, **kw)
        assert native is not None
        return native

    import lethe.kernels.ops.gdn_layer as layer_mod

    monkeypatch.setattr(layer_mod, "gdn2_backward", _force_native)
    try:
        o = gdn2_op(q, k, v, g, b, w)
        o.backward(do)
    finally:
        monkeypatch.setattr(layer_mod, "gdn2_backward", real)
    assert q.grad is not None and torch.isfinite(q.grad).all()
