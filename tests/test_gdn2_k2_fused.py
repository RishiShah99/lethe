"""Fused K#2 — desk correctness of the kernel spec (campaign inc-1). CPU only.

* ``_run_k2_fused_modelled`` (the fused kernel's dataflow: padded staged operands,
  shared dT accumulator, chained s_dt/s_x landings, G6 mask-in-epilogue) reproduces
  ``k2_wy_vjp_cw_ref`` in fp64 to roundoff — including at the kernel's native tile
  (C=64, d_k=128, d_v=64).
* The staged pack's zero-padding is exact: no padded lane leaks into a live output.
* Drifted-regime (within-chunk log2 span > 128): fp32 stays finite (the masked-
  unfactored exp2 discipline) and fp64 still pins to the ref.
"""

from __future__ import annotations

import pytest
import torch

from lethe.kernels.cute.gdn2_assemble import k2_wy_vjp_cw_ref
from lethe.kernels.cute.gdn2_bwd_wy_cw import (
    _k2f_pack,
    _run_k2_fused_modelled,
    k2f_dims_ok,
)
from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

SHAPES = [
    (2, 32, 2, 16, 16, 16),  # B, T, H, d_k, d_v, chunk_len  (NT=2)
    (1, 32, 3, 16, 16, 8),  # NT=4
    (2, 48, 2, 24, 20, 16),  # d_k != d_v, NT=3
    (1, 64, 4, 32, 32, 64),  # single chunk
    (2, 128, 2, 128, 64, 64),  # the fused kernel's native tile, NT=2
    (2, 128, 2, 128, 128, 64),  # the crown tile (d_v=128 dp N-tiling), NT=2
]


def _cw_inputs(shape, seed=0, dtype=torch.float64, g_scale=1.0):
    b, t, h, d_k, d_v, _ = shape
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    k = torch.randn(b, t, h, d_k, generator=gen, dtype=dtype)
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.3 + 0.02) * g_scale
    b_gate = torch.rand(b, t, h, d_k, generator=gen, dtype=dtype) * 0.8 + 0.1
    w_gate = torch.rand(b, t, h, d_v, generator=gen, dtype=dtype) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dtype)
    return q, k, v, g, b_gate, w_gate, do


def _l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _k2_bundle_inputs(shape, seed, g_scale=1.0):
    q, k, v, g, b_gate, w_gate, do = _cw_inputs(shape, seed=seed, g_scale=g_scale)
    cl = shape[5]
    s = shape[3] ** -0.5
    bundles = build_microgate_bundles_cw(
        _l2norm(q), _l2norm(k), v, g, b_gate, w_gate, do, chunk_len=cl, scale=s
    )
    inp = bundles["k2"].inputs
    return (
        inp["k"],
        inp["v"],
        inp["b"],
        inp["w"],
        inp["g2"],
        inp["T"],
        inp["dwy"],
        inp["du"],
    )


@pytest.mark.parametrize("shape", SHAPES)
def test_modelled_matches_ref_fp64(shape):
    args = _k2_bundle_inputs(shape, seed=11)
    got = _run_k2_fused_modelled(*args)
    exp = k2_wy_vjp_cw_ref(*args)
    for g_out, e_out in zip(got, exp, strict=True):
        assert torch.allclose(g_out, e_out, rtol=1e-13, atol=1e-13)


def test_pack_padding_is_exact():
    """Padded a@b^T on the staged operands equals the unpadded products (fp64)."""
    shape = (2, 128, 2, 128, 64, 64)
    k, v, b, w, g2, t_mat, dwy, du = _k2_bundle_inputs(shape, seed=12)
    c, d_k = k.shape[-2], k.shape[-1]
    d_v = v.shape[-1]
    buf = _k2f_pack(k, v, b, w, g2, t_mat, dwy, du)

    def _flat(x):
        return x.reshape(-1, *x.shape[3:])

    tf, duf, dwyf = _flat(t_mat), _flat(du), _flat(dwy)
    tt = tf.transpose(-1, -2)
    pwv = _flat(w) * _flat(v)
    q_kbg = _flat(b) * _flat(k) * torch.exp2(_flat(g2))

    dp_pad = (buf["a_tt"] @ buf["b_du"].transpose(-1, -2))[:, :c, :d_v]
    assert torch.allclose(dp_pad, tt @ duf, rtol=1e-13, atol=1e-13)
    dqwy_pad = (buf["a_tt"] @ buf["b_dwy"].transpose(-1, -2))[:, :c, :d_k]
    assert torch.allclose(dqwy_pad, tt @ dwyf, rtol=1e-13, atol=1e-13)
    dt_pad = (
        buf["a_du"] @ buf["b_pwv"].transpose(-1, -2)
        + buf["a_dwy"] @ buf["b_qkbg"].transpose(-1, -2)
    )[:, :c, :c]
    dt_ref = duf @ pwv.transpose(-1, -2) + dwyf @ q_kbg.transpose(-1, -2)
    assert torch.allclose(dt_pad, dt_ref, rtol=1e-13, atol=1e-13)
    # Padded lanes stay exactly zero (nothing leaks into a live region on re-read).
    full = buf["a_tt"] @ buf["b_du"].transpose(-1, -2)
    assert torch.equal(full[:, c:], torch.zeros_like(full[:, c:]))


def test_drifted_regime_fp32_finite_and_fp64_pins():
    """Within-chunk log2 span > 128: masked-unfactored exp2 keeps fp32 finite."""
    shape = (1, 128, 2, 128, 64, 64)
    args64 = _k2_bundle_inputs(shape, seed=13, g_scale=40.0)
    g2 = args64[4]
    span = (g2.max(dim=-2).values - g2.min(dim=-2).values).max()
    assert span > 128.0  # the regime that NaN'd the unmasked form (c707201)

    out64 = _run_k2_fused_modelled(*args64)
    exp64 = k2_wy_vjp_cw_ref(*args64)
    for g_out, e_out in zip(out64, exp64, strict=True):
        assert torch.isfinite(g_out).all()
        assert torch.allclose(g_out, e_out, rtol=1e-12, atol=1e-12)

    args32 = tuple(a.float() for a in args64)
    out32 = _run_k2_fused_modelled(*args32)
    for g_out in out32:
        assert torch.isfinite(g_out).all()


def test_k2f_dims_ok():
    assert k2f_dims_ok(64, 128, 64)
    assert k2f_dims_ok(64, 128, 128)  # crown: dp N-tiling path
    assert not k2f_dims_ok(64, 128, 96)  # not a clean 1- or 2-tile d_v
    assert not k2f_dims_ok(32, 128, 64)
    assert not k2f_dims_ok(64, 64, 64)


def test_pack_rejects_oversize():
    def _z(*dims):
        return torch.zeros(*dims, dtype=torch.float64)

    c, d_k, d_v = 16, 16, 160  # d_v > 2·N_TILE overflows the dp N-tiling
    with pytest.raises(ValueError, match="tiler"):
        _k2f_pack(
            _z(1, 1, 2, c, d_k),
            _z(1, 1, 2, c, d_v),
            _z(1, 1, 2, c, d_k),
            _z(1, 1, 2, c, d_v),
            _z(1, 1, 2, c, d_k),
            _z(1, 1, 2, c, c),
            _z(1, 1, 2, c, d_k),
            _z(1, 1, 2, c, d_v),
        )


def test_fused_module_imports_cleanly_off_box():
    """The DSL module imports without cutlass (guarded try) and dim-locks correctly."""
    import lethe.kernels.cute.gdn2_bwd_wy_f as k2f

    assert hasattr(k2f, "run_k2_fused")
    args = _k2_bundle_inputs((1, 64, 1, 128, 64, 64), seed=15)
    if not k2f.is_available():
        with pytest.raises(RuntimeError, match="CuTe DSL"):
            k2f.run_k2_fused(*args)


def test_selector_routes_fused_on_proven_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    import lethe.kernels.cute.gdn2_bwd_wy_cw as wy_cw
    import lethe.kernels.cute.gdn2_bwd_wy_f as wy_f

    hit = {"fused": 0, "batched": 0}
    sentinel = tuple(torch.ones(1) for _ in range(5))
    monkeypatch.setattr(
        wy_f, "run_k2_fused", lambda *a: hit.__setitem__("fused", hit["fused"] + 1) or sentinel
    )
    monkeypatch.setattr(
        wy_cw,
        "run_k2_batched",
        lambda *a: hit.__setitem__("batched", hit["batched"] + 1) or sentinel,
    )
    args = tuple(torch.zeros(1, 2, 2, 64, d) for d in (128, 64, 128, 64, 128)) + tuple(
        torch.zeros(1, 2, 2, 64, d) for d in (64, 128, 64)
    )
    # (k, v, b, w, g2) then (T[c,c] stand-in via d=64, dwy, du)
    out = wy_cw.run_k2(*args)
    assert hit == {"fused": 1, "batched": 0}
    assert out == sentinel


def test_selector_kill_switch_and_off_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    import lethe.kernels.cute.gdn2_bwd_wy_cw as wy_cw
    import lethe.kernels.cute.gdn2_bwd_wy_f as wy_f

    hit = {"fused": 0, "batched": 0}
    sentinel = tuple(torch.ones(1) for _ in range(5))
    monkeypatch.setattr(
        wy_f, "run_k2_fused", lambda *a: hit.__setitem__("fused", hit["fused"] + 1) or sentinel
    )
    monkeypatch.setattr(
        wy_cw,
        "run_k2_batched",
        lambda *a: hit.__setitem__("batched", hit["batched"] + 1) or sentinel,
    )

    on_tile = tuple(torch.zeros(1, 2, 2, 64, d) for d in (128, 64, 128, 64, 128, 64, 128, 64))
    monkeypatch.setenv("FMR_DISABLE_K2F", "1")
    wy_cw.run_k2(*on_tile)
    assert hit == {"fused": 0, "batched": 1}

    monkeypatch.delenv("FMR_DISABLE_K2F")
    off_tile = tuple(
        torch.zeros(1, 2, 2, 64, d) for d in (128, 96, 128, 96, 128, 64, 128, 96)
    )  # d_v=96: off the fused tile (not a clean 1- or 2-tile d_v)
    wy_cw.run_k2(*off_tile)
    assert hit == {"fused": 0, "batched": 2}


def test_fused_launcher_rejects_off_tile_dims():
    """Dim lock fires before any toolchain use: d_v=96 is off the {64,128} tile set."""
    import lethe.kernels.cute.gdn2_bwd_wy_f as k2f

    if not k2f.is_available():
        pytest.skip("off-box: the RuntimeError guard fires before the dim lock")
    bad = [torch.zeros(1, 1, 1, 64, 96)] * 2
    with pytest.raises(ValueError, match="d_v"):
        k2f.run_k2_fused(
            torch.zeros(1, 1, 1, 64, 128),
            torch.zeros(1, 1, 1, 64, 96),  # d_v=96: off tile
            torch.zeros(1, 1, 1, 64, 128),
            torch.zeros(1, 1, 1, 64, 96),
            torch.zeros(1, 1, 1, 64, 128),
            torch.zeros(1, 1, 1, 64, 64),
            *bad,
        )
