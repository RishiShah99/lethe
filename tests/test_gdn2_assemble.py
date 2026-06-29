"""Phase-2 integration: the native scalar-GDN backward ASSEMBLY.

The two hard kernels (K#1 reverse-state scan, K#2 WY-VJP) are validated in isolation
by their on-box micro-gates; here the assembly that wires them into the six grads is
validated end-to-end, off-box, by routing both kernels to their pure-torch references.

Three layers:
* the scalar assembly is bit-exact (fp64) vs the token-serial oracle, across shapes
  and at the kernels' production tile dims (d_k=128, d_v=64);
* the GDN-2-signature wrapper's channel-sum reproduces the oracle (the scalar-regime
  reduction), and the native dispatch honours the regime + tile-dim + availability gates;
* the assembly passes all 12 contract gates through the scalar reduction gate.
"""

from __future__ import annotations

import pytest
import torch

import flash_mamba_rl.kernels.cute.gdn2_backward as gdn2_native
from flash_mamba_rl.kernels.cute.gdn2_assemble import (
    assemble_gdn2_backward_scalar,
    assembled_scalar_gdn2_backward,
    k1_reverse_state_ref,
    k2_wy_vjp_ref,
)
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward
from flash_mamba_rl.verifier.op_harness import (
    GDN2_REDUCTION_VIEWS,
    verify_gdn2_reduction_op,
    verify_gdn2_reduction_op_all_grads,
)

_ALL_GATES_COUNT = 12


def _scalar_inputs(
    b: int, t: int, h: int, d_k: int, d_v: int, *, seed: int = 0, dtype: torch.dtype = torch.float64
):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = -torch.rand(b, t, h, generator=gen, dtype=dtype) * 0.1
    beta = torch.rand(b, t, h, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, beta, do


def _oracle_reduced(q, k, v, g, beta, do):
    """Oracle in the reduction regime (channel-constant g, beta-broadcast b/w)."""
    b, t, h, d_k = q.shape
    d_v = v.shape[-1]
    g_ch = g.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    b_g = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
    w_g = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
    return reference_gdn2_backward(q, k, v, g_ch, b_g, w_g, do)


def _rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


class TestAssemblyMatchesOracle:
    @pytest.mark.parametrize(
        "shape",
        [(2, 128, 3, 16, 16), (1, 64, 2, 16, 16), (2, 192, 1, 8, 8), (1, 256, 2, 32, 16)],
    )
    def test_scalar_assembly_bit_exact_fp64(self, shape: tuple[int, ...]) -> None:
        b, t, h, d_k, d_v = shape
        q, k, v, g, beta, do = _scalar_inputs(b, t, h, d_k, d_v)
        got = assemble_gdn2_backward_scalar(q, k, v, g, beta, do)
        orc = _oracle_reduced(q, k, v, g, beta, do)
        assert _rel(got.dq, orc.grad_q) < 1e-9
        assert _rel(got.dk, orc.grad_k) < 1e-9
        assert _rel(got.dv, orc.grad_v) < 1e-9
        assert _rel(got.dg, orc.grad_g.sum(-1)) < 1e-9
        assert _rel(got.db_erase, orc.grad_b.sum(-1)) < 1e-9
        assert _rel(got.dw_write, orc.grad_w.sum(-1)) < 1e-9

    def test_production_tile_dims_dk128_dv64(self) -> None:
        # The tile dims the tcgen05 kernels target (d_k != d_v).
        q, k, v, g, beta, do = _scalar_inputs(1, 128, 2, 128, 64)
        got = assemble_gdn2_backward_scalar(q, k, v, g, beta, do, chunk_len=64)
        orc = _oracle_reduced(q, k, v, g, beta, do)
        assert _rel(got.dq, orc.grad_q) < 1e-9
        assert _rel(got.dv, orc.grad_v) < 1e-9
        assert _rel(got.dg, orc.grad_g.sum(-1)) < 1e-9
        assert _rel(got.db_erase + got.dw_write, orc.grad_b.sum(-1) + orc.grad_w.sum(-1)) < 1e-9

    def test_kernel_refs_are_the_assembly_default(self) -> None:
        q, k, v, g, beta, do = _scalar_inputs(2, 128, 2, 16, 16)
        explicit = assemble_gdn2_backward_scalar(
            q, k, v, g, beta, do, k1_fn=k1_reverse_state_ref, k2_fn=k2_wy_vjp_ref
        )
        default = assemble_gdn2_backward_scalar(q, k, v, g, beta, do)
        assert torch.equal(explicit.dq, default.dq)
        assert torch.equal(explicit.dv, default.dv)


class TestWrapperAndDispatch:
    def test_wrapper_channel_sum_matches_oracle(self) -> None:
        b, t, h, d_k, d_v = 2, 128, 2, 16, 16
        q, k, v, g, beta, do = _scalar_inputs(b, t, h, d_k, d_v)
        g_ch = g.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
        b_g = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
        w_g = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
        got = assembled_scalar_gdn2_backward(q, k, v, g_ch, b_g, w_g, do)
        orc = reference_gdn2_backward(q, k, v, g_ch, b_g, w_g, do)
        assert _rel(got.grad_q, orc.grad_q) < 1e-9
        assert _rel(got.grad_v, orc.grad_v) < 1e-9
        assert _rel(got.grad_g.sum(-1), orc.grad_g.sum(-1)) < 1e-9
        assert _rel(got.grad_b.sum(-1), orc.grad_b.sum(-1)) < 1e-9
        assert _rel(got.grad_w.sum(-1), orc.grad_w.sum(-1)) < 1e-9
        assert got.grad_initial_state is None

    def test_native_none_when_unavailable(self) -> None:
        q, k, v, g, beta, do = _scalar_inputs(1, 64, 2, 128, 64, dtype=torch.float32)
        g_ch = g.unsqueeze(-1).expand_as(q).contiguous()
        b_g = beta.unsqueeze(-1).expand_as(q).contiguous()
        w_g = beta.unsqueeze(-1).expand_as(v).contiguous()
        assert gdn2_native.is_available(torch.device("cpu")) is False
        assert gdn2_native.native_gdn2_backward(q, k, v, g_ch, b_g, w_g, do) is None

    def test_native_none_for_channelwise_even_if_available(self, monkeypatch) -> None:
        monkeypatch.setattr(gdn2_native, "is_available", lambda device=None: True)
        q, k, v, g, beta, do = _scalar_inputs(1, 64, 2, 128, 64, dtype=torch.float32)
        # genuinely channel-wise g (per-channel jitter) -> not scalar-reducible.
        g_ch = g.unsqueeze(-1).expand_as(q).contiguous()
        g_ch = g_ch - torch.rand_like(g_ch) * 0.01
        b_g = beta.unsqueeze(-1).expand_as(q).contiguous()
        w_g = beta.unsqueeze(-1).expand_as(v).contiguous()
        assert gdn2_native.native_gdn2_backward(q, k, v, g_ch, b_g, w_g, do) is None

    def test_native_none_for_wrong_tile_dims(self, monkeypatch) -> None:
        monkeypatch.setattr(gdn2_native, "is_available", lambda device=None: True)
        q, k, v, g, beta, do = _scalar_inputs(1, 64, 2, 16, 16, dtype=torch.float32)
        g_ch = g.unsqueeze(-1).expand_as(q).contiguous()
        b_g = beta.unsqueeze(-1).expand_as(q).contiguous()
        w_g = beta.unsqueeze(-1).expand_as(v).contiguous()
        assert gdn2_native.native_gdn2_backward(q, k, v, g_ch, b_g, w_g, do) is None

    def test_native_runs_assembly_with_injected_refs(self, monkeypatch) -> None:
        # Stand the pure-torch refs in for the box kernels: exercises native's full
        # path (regime + dims gates -> assembly) at the production tile dims.
        monkeypatch.setattr(gdn2_native, "is_available", lambda device=None: True)
        monkeypatch.setattr(
            gdn2_native, "_load_box_kernels", lambda: (k1_reverse_state_ref, k2_wy_vjp_ref)
        )
        b, t, h, d_k, d_v = 1, 128, 2, 128, 64
        q, k, v, g, beta, do = _scalar_inputs(b, t, h, d_k, d_v, dtype=torch.float32)
        g_ch = g.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
        b_g = beta.unsqueeze(-1).expand(b, t, h, d_k).contiguous()
        w_g = beta.unsqueeze(-1).expand(b, t, h, d_v).contiguous()
        got = gdn2_native.native_gdn2_backward(q, k, v, g_ch, b_g, w_g, do)
        orc = reference_gdn2_backward(q, k, v, g_ch, b_g, w_g, do)
        assert got is not None
        assert _rel(got.grad_q, orc.grad_q) < 1e-4
        assert _rel(got.grad_g.sum(-1), orc.grad_g.sum(-1)) < 1e-4


class TestReductionGate:
    def test_grad_v_view_all_gates_pass(self) -> None:
        results = verify_gdn2_reduction_op(
            assembled_scalar_gdn2_backward,
            view="grad_v",
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: r.reason for name, r in results.items() if not r.passed}
        assert not failed, f"grad_v reduction gates failed: {failed}"

    @pytest.mark.slow
    def test_all_views_all_gates_pass(self) -> None:
        verdict = verify_gdn2_reduction_op_all_grads(assembled_scalar_gdn2_backward)
        assert set(verdict) == set(GDN2_REDUCTION_VIEWS)
        for view, results in verdict.items():
            assert len(results) == _ALL_GATES_COUNT, view
            failed = {name: r.reason for name, r in results.items() if not r.passed}
            assert not failed, f"{view} reduction gates failed: {failed}"
