"""Level-2 de-glue promotion — CPU pins for the promoted module.

The kernels themselves are box-only (silicon gate: results/k1_l2_epilogue_box.json);
these tests pin what is CPU-checkable: the module imports cleanly off-box, the
pure-torch model (the kernel spec with fp16 round-trips) tracks the fp64 cw bundle
within the fp16 quantisation floor, and the cw K#1 default selector routes L2 on the
exact proven tile / lever-B otherwise, upcasting L2's f16 outputs to fp32.
"""

from __future__ import annotations

import inspect
import re

import pytest
import torch

import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu as dhu
import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw as dhu_cw
import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l2 as l2
import flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_l3 as l3
from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def test_module_imports_cleanly_off_box() -> None:
    assert hasattr(l2, "run_k1_incB_l2")
    assert l2.l2_dims_ok(64, 128, 64)
    assert not l2.l2_dims_ok(64, 128, 128)
    assert not l2.l2_dims_ok(32, 128, 64)


@pytest.mark.parametrize("nt", [1, 2, 3])
def test_modelled_l2_tracks_fp64_bundle(nt: int) -> None:
    """The kernel spec (fp16 round-trips modelled) stays inside the 5e-3 silicon gate."""
    b, h, c, d_k, d_v = 1, 1, 64, 128, 64
    t = nt * c
    gen = torch.Generator().manual_seed(nt * 17 + c)
    dt = torch.float64
    q = _l2norm(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2norm(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    i1, e1 = bun["k1"].inputs, bun["k1"].expected

    got = l2._modelled_l2(
        i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"], i1["do"], i1["dv_local"], i1["dht"]
    )
    for got_t, name in zip(got, ("dh", "dv2", "dh0"), strict=True):
        ref = e1[name].float()
        rel = ((got_t.float() - ref).abs().max() / ref.abs().max().clamp_min(1e-12)).item()
        assert rel < 5e-3, f"{name}: scale_rel {rel:.2e} outside the silicon gate tol"


def test_selector_routes_l2_on_proven_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    # FMR_DISABLE_L3 drops the selector past the Level-3 fused kernel to Level-2
    # (also the L3 kill-switch regression: L3 disabled must not shadow L2).
    monkeypatch.setenv("FMR_DISABLE_L3", "1")
    b, hv, nt, c, d_k, d_v = 1, 2, 2, 64, 128, 64
    shape_args = [
        torch.zeros(b, hv, nt, c, d_k),  # q
        torch.zeros(b, hv, nt, c, d_k),  # k
        torch.zeros(b, hv, nt, c, d_k),  # wy
        torch.zeros(b, hv, nt, c, d_k),  # g2
        torch.zeros(b, hv, nt, d_k),  # g_last
        torch.zeros(b, hv, nt, c, d_v),  # do
        torch.zeros(b, hv, nt, c, d_v),  # dv_local
        torch.zeros(b, hv, d_k, d_v),  # dht
    ]
    sentinel = (
        torch.ones(b, hv, nt, d_k, d_v, dtype=torch.float16),
        torch.ones(b, hv, nt, c, d_v, dtype=torch.float16),
        torch.ones(b, hv, d_k, d_v, dtype=torch.float32),
    )
    monkeypatch.setattr(l2, "run_k1_incB_l2", lambda *a: sentinel)
    dh, dv2, dh0 = dhu_cw.run_k1_incB(*shape_args)
    assert dh.dtype == torch.float32 and dv2.dtype == torch.float32
    assert torch.equal(dh, sentinel[0].float()) and torch.equal(dv2, sentinel[1].float())
    assert torch.equal(dh0, sentinel[2])


def test_selector_routes_batched_off_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    b, hv, nt, c, d_k, d_v = 1, 1, 1, 64, 128, 128  # d_v=128: off the L2 tile
    shape_args = [
        torch.zeros(b, hv, nt, c, d_k),
        torch.zeros(b, hv, nt, c, d_k),
        torch.zeros(b, hv, nt, c, d_k),
        torch.zeros(b, hv, nt, c, d_k),
        torch.zeros(b, hv, nt, d_k),
        torch.zeros(b, hv, nt, c, d_v),
        torch.zeros(b, hv, nt, c, d_v),
        torch.zeros(b, hv, d_k, d_v),
    ]
    hit = {"batched": 0, "l2": 0}
    monkeypatch.setattr(
        dhu_cw,
        "run_k1_incB_batched",
        lambda *a: hit.__setitem__("batched", hit["batched"] + 1) or ("b", "b", "b"),
    )
    monkeypatch.setattr(
        l2, "run_k1_incB_l2", lambda *a: hit.__setitem__("l2", hit["l2"] + 1) or ("l", "l", "l")
    )
    out = dhu_cw.run_k1_incB(*shape_args)
    assert hit == {"batched": 1, "l2": 0}
    assert out == ("b", "b", "b")


@pytest.mark.parametrize(
    "module",
    [l2, l3, dhu],
    ids=lambda m: m.__name__.rsplit(".", 1)[-1],
)
def test_fence_proxy_precedes_barrier_on_simt_tma_roundtrip(module: object) -> None:
    """Pin that fence_proxy always precedes barrier at SIMT-store-to-TMA round-trip sites.

    The PTX mixed-proxy model requires stores -> fence_proxy (each writing thread)
    -> barrier -> TMA for cross-proxy SIMT/TMA round-trips. This source-pin catches
    any accidental inversion across the L2/L3/v0 sibling kernels.
    """
    src = inspect.getsource(module)  # type: ignore[arg-type]
    barrier_then_fence = re.compile(
        r"cute\.arch\.barrier\(\)\s*\n\s*(?:if\s+[^\n]+:\s*\n\s*)?cute\.arch\.fence_proxy"
    )
    match = barrier_then_fence.search(src)
    assert match is None, (
        f"{module.__name__}: barrier() before fence_proxy() at SIMT-TMA round-trip — "
        f"inverted order breaks cross-proxy synchronization (match at char {match.start()})"
    )
