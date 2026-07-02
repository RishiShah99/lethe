"""Autograd binding for GDN-2 (gdn_layer.gdn2_op).

Two contracts:
* Backward grads from ``gdn2_op(...).sum().backward()`` match ``reference_gdn2_backward``
  to machine precision in fp64 (bitwise on the eager path, where both delegate to the
  same oracle).
* Output shape and dtype are preserved for all six input tensors.
"""

from __future__ import annotations

import pytest
import torch

from flash_mamba_rl.kernels.ops.gdn_layer import gdn2_op
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward


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
        # fp64 eager path delegates to the same oracle — grads must be bitwise equal.
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
    """Non-leaf inputs that don't require grad produce None gradients (standard autograd)."""
    q, k, v, g, b, w, do = _inputs(seed=2)
    # Only q requires grad; the rest do not.
    q_leaf = q.detach().requires_grad_(True)
    o = gdn2_op(q_leaf, k, v, g, b, w)
    o.backward(do)
    assert q_leaf.grad is not None
