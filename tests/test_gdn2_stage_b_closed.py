"""Closed-form Stage-B VJP pins — the de-glue lever's correctness contract.

``_stage_b_vjp_cw_closed`` must equal the autograd Stage B in fp64: it consumes K#1's
``dh`` output (previously discarded) and ``dv2``, and reconstructs every stage-B grad
chunk-locally. Pinned both directly (against ``_stage_b_vjp_cw``) and end-to-end
(``stage_b_closed=True`` assembly vs the default, and vs the token-serial oracle).
Level 1b adds the batched no-grad restage (``chunkwise_restage_cw``) the closed
assembly path runs on — field-pinned against ``chunkwise_forward_cw`` — and the
``stage_b_closed`` threading through the native dispatch. CPU only.
"""

import pytest
import torch

import flash_mamba_rl.kernels.cute.gdn2_backward as gdn2_native
from flash_mamba_rl.kernels.cute.gdn2_assemble import (
    _stage_b_vjp_cw,
    _stage_b_vjp_cw_closed,
    _to_chunks,
    assemble_gdn2_backward_channelwise,
    assembled_channelwise_gdn2_backward,
    k1_reverse_state_cw_ref,
    k2_wy_vjp_cw_ref,
)
from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import (
    chunkwise_forward_cw,
    chunkwise_restage_cw,
)
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward

# (batch, seq_len, nheads, d_k, d_v, chunk_len)
SHAPES = [(2, 32, 2, 16, 16, 16), (1, 64, 3, 16, 8, 16), (2, 48, 2, 24, 20, 16)]


def _inputs(batch: int, seq: int, heads: int, d_k: int, d_v: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(23)
    dt = torch.float64
    q = torch.randn(batch, seq, heads, d_k, generator=gen, dtype=dt)
    k = torch.randn(batch, seq, heads, d_k, generator=gen, dtype=dt)
    v = torch.randn(batch, seq, heads, d_v, generator=gen, dtype=dt)
    g = -torch.rand(batch, seq, heads, d_k, generator=gen, dtype=dt) * 0.1
    b = torch.rand(batch, seq, heads, d_k, generator=gen, dtype=dt)
    w = torch.rand(batch, seq, heads, d_v, generator=gen, dtype=dt)
    do = torch.randn(batch, seq, heads, d_v, generator=gen, dtype=dt)
    return q, k, v, g, b, w, do


class TestStageBClosedForm:
    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v", "cl"), SHAPES)
    def test_matches_autograd_stage_b(self, batch, seq, heads, d_k, d_v, cl) -> None:
        q, k, v, g, b, w, do = _inputs(batch, seq, heads, d_k, d_v)
        fwd = chunkwise_forward_cw(q, k, v, g, b, w, chunk_len=cl, use_qk_l2norm=True)
        do_c = _to_chunks(do, cl)
        dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
        dht = torch.zeros_like(fwd.h_list[0])
        dh, dv2, _dh0 = k1_reverse_state_cw_ref(
            fwd.q.detach(),
            fwd.k.detach(),
            fwd.wy.detach(),
            fwd.g2.detach(),
            fwd.g_last.detach(),
            do_c.detach(),
            dv_local.detach(),
            dht,
        )
        auto = _stage_b_vjp_cw(fwd, do, create_graph=False)
        closed = _stage_b_vjp_cw_closed(fwd, do, dh, dv2)
        for name, a, c in zip(("dq_b", "dk_b", "dg_b", "dwy"), auto[:4], closed, strict=True):
            assert torch.allclose(a, c, rtol=1e-12, atol=1e-13), (
                name,
                (a - c).abs().max().item(),
            )
        # du is dv2 verbatim in the closed splice.
        assert torch.allclose(auto[4], dv2, rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize("use_qk_l2norm", [True, False])
    def test_assembly_end_to_end_equal(self, use_qk_l2norm: bool) -> None:
        # Scale-relative: without qk-l2norm the delta-rule recurrence is
        # ill-conditioned (grads reach ~1e15), so a per-element rtol misreads
        # machine-precision agreement on the small-valued entries.
        q, k, v, g, b, w, do = _inputs(2, 48, 2, 24, 20)
        default = assemble_gdn2_backward_channelwise(
            q, k, v, g, b, w, do, use_qk_l2norm=use_qk_l2norm
        )
        closed = assemble_gdn2_backward_channelwise(
            q, k, v, g, b, w, do, use_qk_l2norm=use_qk_l2norm, stage_b_closed=True
        )
        for f in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
            a, c = getattr(default, f), getattr(closed, f)
            scale_rel = (a - c).abs().max().item() / max(a.abs().max().item(), 1e-30)
            assert scale_rel <= 1e-12, (f, scale_rel)

    def test_closed_assembly_matches_oracle(self) -> None:
        q, k, v, g, b, w, do = _inputs(2, 32, 2, 16, 16)
        got = assemble_gdn2_backward_channelwise(q, k, v, g, b, w, do, stage_b_closed=True)
        want = reference_gdn2_backward(q, k, v, g, b, w, do)
        pairs = [
            (got.dq, want.grad_q),
            (got.dk, want.grad_k),
            (got.dv, want.grad_v),
            (got.dg, want.grad_g),
            (got.db, want.grad_b),
            (got.dw, want.grad_w),
        ]
        for i, (a, c) in enumerate(pairs):
            assert torch.allclose(a, c, rtol=1e-9, atol=1e-10), (
                i,
                (a - c).abs().max().item(),
            )


_RESTAGE_FIELDS = (
    "o",
    "q",
    "k",
    "v",
    "b",
    "w_gate",
    "g2",
    "g_last",
    "gamma",
    "T",
    "u",
    "wy",
    "v_new",
    "h",
    "A_qk",
)


class TestRestage:
    """Level 1b: the batched no-grad restage equals the loop forward, field by field."""

    @pytest.mark.parametrize("use_qk_l2norm", [True, False])
    @pytest.mark.parametrize(("batch", "seq", "heads", "d_k", "d_v", "cl"), SHAPES)
    def test_restage_matches_forward(self, batch, seq, heads, d_k, d_v, cl, use_qk_l2norm) -> None:
        q, k, v, g, b, w, do = _inputs(batch, seq, heads, d_k, d_v)
        del do
        fwd = chunkwise_forward_cw(q, k, v, g, b, w, chunk_len=cl, use_qk_l2norm=use_qk_l2norm)
        rst = chunkwise_restage_cw(q, k, v, g, b, w, chunk_len=cl, use_qk_l2norm=use_qk_l2norm)
        assert rst.chunk_len == fwd.chunk_len and rst.scale == fwd.scale
        assert rst.leaves == ()
        for f in _RESTAGE_FIELDS:
            a, c = getattr(fwd, f).detach(), getattr(rst, f)
            assert not c.requires_grad, f
            assert torch.allclose(a, c, rtol=1e-12, atol=1e-13), (
                f,
                (a - c).abs().max().item(),
            )

    def test_restage_with_initial_state(self) -> None:
        q, k, v, g, b, w, _do = _inputs(1, 32, 2, 16, 12)
        gen = torch.Generator().manual_seed(7)
        h0 = torch.randn(1, 2, 16, 12, generator=gen, dtype=torch.float64)
        fwd = chunkwise_forward_cw(q, k, v, g, b, w, chunk_len=16, initial_state=h0)
        rst = chunkwise_restage_cw(q, k, v, g, b, w, chunk_len=16, initial_state=h0)
        for f in ("o", "h", "v_new"):
            a, c = getattr(fwd, f).detach(), getattr(rst, f)
            assert torch.allclose(a, c, rtol=1e-12, atol=1e-13), f

    def test_restage_rejects_half(self) -> None:
        q, k, v, g, b, w, _do = _inputs(1, 32, 1, 16, 16)
        with pytest.raises(ValueError, match="float32/float64"):
            chunkwise_restage_cw(*(t.to(torch.bfloat16) for t in (q, k, v, g, b, w)), chunk_len=16)


class TestNativeDispatchClosed:
    """``stage_b_closed`` threads through ``native_gdn2_backward`` to the cw assembly."""

    def test_native_threads_stage_b_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Dispatch-legal dims (d_k=128, L%64==0, d_v=64) with the cw refs standing in
        # for the box kernels; the native route must equal the direct closed assembly
        # bitwise (same ops, same order, CPU).
        bsz, t, h, d_k, d_v = 1, 64, 1, 128, 64
        gen = torch.Generator().manual_seed(11)
        dt = torch.float32
        q = torch.randn(bsz, t, h, d_k, generator=gen, dtype=dt)
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
        got = gdn2_native.native_gdn2_backward(q, k, v, g, b, w, do, stage_b_closed=True)
        assert got is not None
        want = assembled_channelwise_gdn2_backward(
            q,
            k,
            v,
            g,
            b,
            w,
            do,
            k1_fn=k1_reverse_state_cw_ref,
            k2_fn=k2_wy_vjp_cw_ref,
            stage_b_closed=True,
        )
        for f in ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w"):
            assert torch.equal(getattr(got, f), getattr(want, f)), f
