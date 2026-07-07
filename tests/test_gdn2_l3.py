"""Level-3 de-glue wiring — CPU pins for the fused-kernel module + selector threading.

The kernel itself is box-only (silicon gates: results/k1_incb2_v3_*.json); these tests
pin what is CPU-checkable: the module imports cleanly off-box, the dim lock matches the
baked tile, and the cw K#1 default selector routes L3 first on the exact proven tile,
honors the FMR_DISABLE_L3 kill-switch, and never routes L3 off-tile. The kernel spec
(``_run_k1_incB2_modelled``) is fp64-pinned by the existing cw orchestration tests.
"""

from __future__ import annotations

import pytest
import torch

import lethe.kernels.cute.gdn2_bwd_dhu_cw as dhu_cw
import lethe.kernels.cute.gdn2_bwd_dhu_l2 as l2
import lethe.kernels.cute.gdn2_bwd_dhu_l3 as l3


def _shape_args(b: int, hv: int, nt: int, c: int, d_k: int, d_v: int) -> list[torch.Tensor]:
    return [
        torch.zeros(b, hv, nt, c, d_k),  # q
        torch.zeros(b, hv, nt, c, d_k),  # k
        torch.zeros(b, hv, nt, c, d_k),  # wy
        torch.zeros(b, hv, nt, c, d_k),  # g2
        torch.zeros(b, hv, nt, d_k),  # g_last
        torch.zeros(b, hv, nt, c, d_v),  # do
        torch.zeros(b, hv, nt, c, d_v),  # dv_local
        torch.zeros(b, hv, d_k, d_v),  # dht
    ]


def test_module_imports_cleanly_off_box() -> None:
    assert hasattr(l3, "run_k1_incB2_v3")
    assert l3.l3_dims_ok(64, 128, 64)
    assert not l3.l3_dims_ok(64, 128, 128)
    assert not l3.l3_dims_ok(32, 128, 64)
    assert not l3.l3_dims_ok(64, 64, 64)


def test_selector_routes_l3_first_on_proven_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    b, hv, nt, c, d_k, d_v = 1, 2, 2, 64, 128, 64
    sentinel = (
        torch.ones(b, hv, nt, d_k, d_v, dtype=torch.float16),
        torch.ones(b, hv, nt, c, d_v, dtype=torch.float16),
        torch.ones(b, hv, d_k, d_v, dtype=torch.float32),
    )
    hit = {"l3": 0, "l2": 0}

    def _l3(*a: object, **kw: object) -> tuple[torch.Tensor, ...]:
        hit["l3"] += 1
        assert kw.get("cw") is True
        return sentinel

    monkeypatch.setattr(l3, "run_k1_incB2_v3", _l3)
    monkeypatch.setattr(
        l2, "run_k1_incB_l2", lambda *a: hit.__setitem__("l2", hit["l2"] + 1) or sentinel
    )
    dh, dv2, dh0 = dhu_cw.run_k1_incB(*_shape_args(b, hv, nt, c, d_k, d_v))
    assert hit == {"l3": 1, "l2": 0}
    assert dh.dtype == torch.float32 and dv2.dtype == torch.float32
    assert torch.equal(dh, sentinel[0].float()) and torch.equal(dv2, sentinel[1].float())
    assert torch.equal(dh0, sentinel[2])


def test_kill_switch_drops_l3_to_l2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMR_DISABLE_L3", "1")
    b, hv, nt, c, d_k, d_v = 1, 1, 1, 64, 128, 64
    hit = {"l3": 0, "l2": 0}
    sentinel = (
        torch.ones(b, hv, nt, d_k, d_v, dtype=torch.float16),
        torch.ones(b, hv, nt, c, d_v, dtype=torch.float16),
        torch.ones(b, hv, d_k, d_v, dtype=torch.float32),
    )
    monkeypatch.setattr(
        l3,
        "run_k1_incB2_v3",
        lambda *a, **kw: hit.__setitem__("l3", hit["l3"] + 1) or sentinel,
    )
    monkeypatch.setattr(
        l2, "run_k1_incB_l2", lambda *a: hit.__setitem__("l2", hit["l2"] + 1) or sentinel
    )
    dhu_cw.run_k1_incB(*_shape_args(b, hv, nt, c, d_k, d_v))
    assert hit == {"l3": 0, "l2": 1}


def test_selector_skips_l3_off_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    b, hv, nt, c, d_k, d_v = 1, 1, 1, 64, 128, 128  # d_v=128: off the L3/L2 tile
    hit = {"l3": 0, "l2": 0, "batched": 0}
    monkeypatch.setattr(
        l3,
        "run_k1_incB2_v3",
        lambda *a, **kw: hit.__setitem__("l3", hit["l3"] + 1) or ("x", "x", "x"),
    )
    monkeypatch.setattr(
        l2, "run_k1_incB_l2", lambda *a: hit.__setitem__("l2", hit["l2"] + 1) or ("x", "x", "x")
    )
    monkeypatch.setattr(
        dhu_cw,
        "run_k1_incB_batched",
        lambda *a: hit.__setitem__("batched", hit["batched"] + 1) or ("b", "b", "b"),
    )
    out = dhu_cw.run_k1_incB(*_shape_args(b, hv, nt, c, d_k, d_v))
    assert hit == {"l3": 0, "l2": 0, "batched": 1}
    assert out == ("b", "b", "b")
