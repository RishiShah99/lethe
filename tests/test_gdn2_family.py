"""Family reduction credentials: GLA / LA / SSD-class / KDA through the one assembly.

Three layers, mirroring the crown's discipline:

1. Oracle cross-pins — each independent token-serial family oracle equals the GDN-2
   oracle at its gate settings (machine precision; exact where the op sequences
   coincide).
2. Value correctness — each family wrapper (channel-wise assembly at gate settings,
   grads mapped to family parameters) matches its family oracle in fp64.
3. The 12-gate battery per family view (verify_gdn2_family_op), candidate vs the
   structurally matched refs-path wrapper.

Plus the ``b = 0`` fast-path pins: skip-T forward and no-K#2 assembly are exactly the
full path at b = 0. CPU only.
"""

import pytest
import torch

from flash_mamba_rl.kernels.cute.gdn2_assemble import assemble_gdn2_backward_no_erase
from flash_mamba_rl.kernels.cute.gdn2_backward import (
    native_gla_backward,
    native_kda_backward,
    native_la_backward,
    native_ssd_backward,
)
from flash_mamba_rl.kernels.cute.gdn2_family import (
    gla_backward,
    kda_backward,
    la_backward,
    ssd_backward,
)
from flash_mamba_rl.kernels.references.family_oracles import (
    reference_gla_backward,
    reference_gla_forward,
    reference_kda_backward,
    reference_kda_forward,
    reference_la_backward,
    reference_la_forward,
    reference_ssd_backward,
    reference_ssd_forward,
)
from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import (
    chunkwise_backward_cw,
    chunkwise_forward_cw,
)
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_forward
from flash_mamba_rl.verifier.op_harness import (
    FAMILY_GATE_VIEWS,
    _family_bwd_aux,
    verify_gdn2_family_op,
    verify_gdn2_family_op_all_views,
)

_ALL_GATES_COUNT = 12

# (batch, seq_len, nheads, d_k, d_v) — chunk-boundary-straddling and uneven dims.
SHAPES = [(2, 32, 2, 16, 16), (1, 64, 3, 16, 8), (2, 48, 2, 24, 20)]

FAMILY_BACKWARDS = {
    "gla": gla_backward,
    "la": la_backward,
    "ssd": ssd_backward,
    "kda": kda_backward,
}
FAMILY_ORACLES = {
    "gla": reference_gla_backward,
    "la": reference_la_backward,
    "ssd": reference_ssd_backward,
    "kda": reference_kda_backward,
}


def _family_inputs(
    family: str, batch: int, seq: int, heads: int, d_k: int, d_v: int, seed: int = 0
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(seed)
    dt = torch.float64
    q = torch.randn(batch, seq, heads, d_k, generator=gen, dtype=dt)
    k = torch.randn(batch, seq, heads, d_k, generator=gen, dtype=dt)
    v = torch.randn(batch, seq, heads, d_v, generator=gen, dtype=dt)
    g = -torch.rand(batch, seq, heads, d_k, generator=gen, dtype=dt) * 0.1
    g_scalar = -torch.rand(batch, seq, heads, generator=gen, dtype=dt) * 0.1
    beta = torch.randn(batch, seq, heads, generator=gen, dtype=dt).sigmoid()
    if family == "gla":
        return (q, k, v, g)
    if family == "la":
        return (q, k, v)
    if family == "ssd":
        return (q, k, v, g_scalar)
    if family == "kda":
        return (q, k, v, g, beta)
    raise ValueError(family)


class TestFamilyOracleCrossPins:
    """Each independent family oracle vs the GDN-2 oracle at its gate settings."""

    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_gla_is_gdn2_at_b0_w1(self, batch, seq, heads, d_k, d_v) -> None:
        q, k, v, g = _family_inputs("gla", batch, seq, heads, d_k, d_v)
        o_gla = reference_gla_forward(q, k, v, g)
        o_gdn2 = reference_gdn2_forward(q, k, v, g, torch.zeros_like(g), torch.ones_like(v))
        # b=0 zeroes the erase term exactly and w=1 is an exact multiply, so the two
        # op sequences coincide bit-for-bit.
        assert torch.equal(o_gla, o_gdn2)

    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_la_is_gla_at_g0(self, batch, seq, heads, d_k, d_v) -> None:
        q, k, v = _family_inputs("la", batch, seq, heads, d_k, d_v)
        o_la = reference_la_forward(q, k, v)
        o_gla = reference_gla_forward(q, k, v, torch.zeros(batch, seq, heads, d_k).double())
        assert torch.allclose(o_la, o_gla, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_ssd_is_gla_at_channel_const_g(self, batch, seq, heads, d_k, d_v) -> None:
        q, k, v, g_scalar = _family_inputs("ssd", batch, seq, heads, d_k, d_v)
        o_ssd = reference_ssd_forward(q, k, v, g_scalar)
        g_cw = g_scalar.unsqueeze(-1).expand(batch, seq, heads, d_k)
        o_gla = reference_gla_forward(q, k, v, g_cw)
        assert torch.allclose(o_ssd, o_gla, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_kda_is_gdn2_at_b_w_beta(self, batch, seq, heads, d_k, d_v) -> None:
        q, k, v, g, beta = _family_inputs("kda", batch, seq, heads, d_k, d_v)
        o_kda = reference_kda_forward(q, k, v, g, beta)
        b = beta.unsqueeze(-1).expand(batch, seq, heads, d_k)
        w = beta.unsqueeze(-1).expand(batch, seq, heads, d_v)
        o_gdn2 = reference_gdn2_forward(q, k, v, g, b, w)
        # beta*(v - kS) vs beta*v - beta*kS: same math, different distribution order.
        assert torch.allclose(o_kda, o_gdn2, rtol=0.0, atol=1e-12)


class TestFamilyWrapperValue:
    """Family wrappers (chunkwise assembly at gate settings) vs the family oracles, fp64."""

    @pytest.mark.parametrize("family", sorted(FAMILY_BACKWARDS))
    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_wrapper_matches_oracle(self, family, batch, seq, heads, d_k, d_v) -> None:
        inputs = _family_inputs(family, batch, seq, heads, d_k, d_v)
        do = torch.randn(
            batch, seq, heads, d_v, generator=torch.Generator().manual_seed(7), dtype=torch.float64
        )
        got = FAMILY_BACKWARDS[family](*inputs, do)
        want = FAMILY_ORACLES[family](*inputs, do)
        assert got._fields == want._fields
        for name, a, b in zip(got._fields, got, want, strict=True):
            assert a.shape == b.shape, (family, name)
            assert torch.allclose(a, b, rtol=1e-9, atol=1e-11), (
                family,
                name,
                (a - b).abs().max().item(),
            )

    def test_ssd_grad_g_is_scalar_shaped(self) -> None:
        q, k, v, g_scalar = _family_inputs("ssd", 2, 32, 2, 16, 16)
        do = torch.randn_like(v)
        grads = ssd_backward(q, k, v, g_scalar, do)
        assert grads.grad_g.shape == g_scalar.shape

    def test_kda_grad_beta_is_scalar_shaped(self) -> None:
        q, k, v, g, beta = _family_inputs("kda", 2, 32, 2, 16, 16)
        do = torch.randn_like(v)
        grads = kda_backward(q, k, v, g, beta, do)
        assert grads.grad_beta.shape == beta.shape

    def test_ssd_rejects_channelwise_g(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        with pytest.raises(ValueError, match="B, L, H"):
            ssd_backward(q, k, v, g, torch.randn_like(v))

    def test_kda_rejects_channelwise_beta(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        with pytest.raises(ValueError, match="B, L, H"):
            kda_backward(q, k, v, g, g, torch.randn_like(v))


class TestNoEraseFastPath:
    """skip-T forward and the no-K#2 assembly are exactly the full path at b = 0."""

    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v"), SHAPES)
    def test_skip_erase_forward_bit_equal(self, batch, seq, heads, d_k, d_v) -> None:
        q, k, v, g = _family_inputs("gla", batch, seq, heads, d_k, d_v)
        b = torch.zeros_like(g)
        w = torch.rand_like(v)
        full = chunkwise_forward_cw(q, k, v, g, b, w, chunk_len=16, use_qk_l2norm=True)
        fast = chunkwise_forward_cw(
            q, k, v, g, b, w, chunk_len=16, use_qk_l2norm=True, skip_erase=True
        )
        assert torch.equal(fast.o, full.o)
        assert torch.equal(fast.h, full.h)
        assert torch.equal(fast.u, full.u)
        assert torch.equal(fast.v_new, full.v_new)
        assert torch.equal(fast.A_qk, full.A_qk)
        assert not fast.wy.any()
        eye = torch.eye(16, dtype=q.dtype).expand_as(fast.T)
        assert torch.equal(fast.T, eye)

    def test_skip_erase_rejects_nonzero_b(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        b = torch.full_like(g, 0.5)
        with pytest.raises(ValueError, match="skip_erase"):
            chunkwise_forward_cw(q, k, v, g, b, torch.ones_like(v), skip_erase=True)

    def test_backward_rejects_skip_erase_forward(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        fwd = chunkwise_forward_cw(
            q, k, v, g, torch.zeros_like(g), torch.ones_like(v), chunk_len=16, skip_erase=True
        )
        with pytest.raises(ValueError, match="skip_erase"):
            chunkwise_backward_cw(fwd, torch.randn_like(v))

    @pytest.mark.parametrize("family", ["gla", "la", "ssd"])
    def test_fast_path_equals_full_path(self, family) -> None:
        inputs = _family_inputs(family, 2, 48, 2, 24, 20)
        do = torch.randn(
            2, 48, 2, 20, generator=torch.Generator().manual_seed(11), dtype=torch.float64
        )
        fast = FAMILY_BACKWARDS[family](*inputs, do, fast_path=True)
        full = FAMILY_BACKWARDS[family](*inputs, do, fast_path=False)
        for name, a, b in zip(fast._fields, fast, full, strict=True):
            assert torch.equal(a, b), (family, name)

    def test_fast_path_rejects_k2_injection(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        with pytest.raises(ValueError, match="fast_path"):
            gla_backward(q, k, v, g, torch.randn_like(v), k2_fn=lambda *a: a)

    def test_no_erase_k1_injection_is_used(self) -> None:
        q, k, v, g = _family_inputs("gla", 1, 32, 2, 16, 16)
        w = torch.ones_like(v)
        calls: list[int] = []

        def spy_k1(qc, kc, wy, g2, g_last, do_c, dv_local, dht):  # type: ignore[no-untyped-def]
            calls.append(1)
            assert not wy.any()
            from flash_mamba_rl.kernels.cute.gdn2_assemble import k1_reverse_state_cw_ref

            return k1_reverse_state_cw_ref(qc, kc, wy, g2, g_last, do_c, dv_local, dht)

        assemble_gdn2_backward_no_erase(q, k, v, g, w, torch.randn_like(v), k1_fn=spy_k1)
        assert calls


class TestFamilyDispatch:
    """Off-box the native family modes return None (the fallback contract)."""

    def test_native_family_modes_none_off_box(self) -> None:
        if torch.cuda.is_available():
            pytest.skip("desk-only contract")
        q = torch.randn(1, 64, 2, 128)
        v = torch.randn(1, 64, 2, 64)
        g = -torch.rand(1, 64, 2, 128) * 0.1
        gs = -torch.rand(1, 64, 2) * 0.1
        beta = torch.rand(1, 64, 2)
        do = torch.randn_like(v)
        assert native_gla_backward(q, q, v, g, do) is None
        assert native_la_backward(q, q, v, do) is None
        assert native_ssd_backward(q, q, v, gs, do) is None
        assert native_kda_backward(q, q, v, g, beta, do) is None


class TestFamilyGateBattery:
    def test_family_aux_deterministic(self) -> None:
        for family in FAMILY_GATE_VIEWS:
            a = _family_bwd_aux(family, 2, 16, 2, 4, 4, torch.device("cpu"), torch.float32)
            b = _family_bwd_aux(family, 2, 16, 2, 4, 4, torch.device("cpu"), torch.float32)
            assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True)), family

    @pytest.mark.parametrize("family", sorted(FAMILY_GATE_VIEWS))
    def test_all_gates_pass_on_cpu_grad_v_view(self, family) -> None:
        results = verify_gdn2_family_op(
            FAMILY_BACKWARDS[family],
            family=family,
            view="grad_v",
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: r.reason for name, r in results.items() if not r.passed}
        assert not failed, f"{family} grad_v gates failed on CPU: {failed}"

    @pytest.mark.slow
    @pytest.mark.parametrize("family", sorted(FAMILY_GATE_VIEWS))
    def test_all_gates_pass_all_views(self, family) -> None:
        all_results = verify_gdn2_family_op_all_views(
            FAMILY_BACKWARDS[family],
            family=family,
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert set(all_results) == set(FAMILY_GATE_VIEWS[family])
        for view, results in all_results.items():
            assert len(results) == _ALL_GATES_COUNT, (family, view)
            failed = {name: r.reason for name, r in results.items() if not r.passed}
            assert not failed, f"{family} {view} gates failed on CPU: {failed}"
