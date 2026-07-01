"""Regression: the scalar K#1/K#2 backward run functions are CUDA-graph safe.

A ``torch.cuda.synchronize()`` inside a ``torch.cuda.graph`` capture region is
illegal and aborts the capture. The channel-wise crown routes its end-of-run
device sync through ``maybe_sync()`` (a no-op under ``graph_capture()``); the
scalar K#1/K#2 run functions (and the cw serial fallbacks) previously used an
unconditional sync, silently non-capturable. These tests pin that every run
function uses the capturable sync, and that the scalar batched paths actually run
under capture without calling ``torch.cuda.synchronize`` while still syncing once
outside capture. CPU-only: the tcgen05 GEMM is stubbed, so only the host
orchestration + the sync placement are exercised.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch

from flash_mamba_rl.kernels.cute import (
    gdn2_bwd_dhu,
    gdn2_bwd_dhu_cw,
    gdn2_bwd_wy,
    gdn2_bwd_wy_cw,
)
from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import graph_capture

_RUN_FNS = [
    gdn2_bwd_wy.run_k2_serial,
    gdn2_bwd_wy.run_k2_batched,
    gdn2_bwd_dhu.run_k1,
    gdn2_bwd_dhu.run_k1_incB_host,
    gdn2_bwd_dhu.run_k1_incB_batched,
    gdn2_bwd_dhu_cw.run_k1_incB_serial,
    gdn2_bwd_dhu_cw.run_k1_incB_batched,
    gdn2_bwd_wy_cw.run_k2_serial,
    gdn2_bwd_wy_cw.run_k2_batched,
]


@pytest.mark.parametrize(
    "fn",
    _RUN_FNS,
    ids=lambda f: f"{f.__module__.rsplit('.', 1)[-1]}.{f.__name__}",
)
def test_run_fn_uses_capturable_sync(fn: Any) -> None:
    src = inspect.getsource(fn)
    assert "torch.cuda.synchronize" not in src, (
        f"{fn.__name__} has an unconditional sync — breaks CUDA-graph capture"
    )
    assert "maybe_sync()" in src, f"{fn.__name__} must sync via the capturable maybe_sync()"


def _cpu_bmm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Stand-in for the tcgen05 batched GEMM: plain fp32 batched matmul on CPU."""
    return x.float() @ y.float()


def _k2_inputs() -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    b, h, nt, c, d_k, d_v = 1, 1, 1, 2, 2, 2
    k = torch.randn(b, h, nt, c, d_k, generator=g)
    v = torch.randn(b, h, nt, c, d_v, generator=g)
    beta = torch.randn(b, h, nt, c, generator=g)
    g2 = torch.randn(b, h, nt, c, generator=g) * 0.1
    t_mat = torch.eye(c).reshape(1, 1, 1, c, c)
    dw = torch.randn(b, h, nt, c, d_k, generator=g)
    du = torch.randn(b, h, nt, c, d_v, generator=g)
    return k, v, beta, g2, t_mat, dw, du


def _k1_inputs() -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(0)
    b, hv, nt, c, d_k, d_v = 1, 1, 1, 2, 2, 2
    q = torch.randn(b, hv, nt, c, d_k, generator=g)
    k = torch.randn(b, hv, nt, c, d_k, generator=g)
    w = torch.randn(b, hv, nt, c, d_k, generator=g)
    g2 = torch.randn(b, hv, nt, c, generator=g) * 0.1
    g_last = torch.randn(b, hv, nt, generator=g) * 0.1
    do = torch.randn(b, hv, nt, c, d_v, generator=g)
    dv_local = torch.randn(b, hv, nt, c, d_v, generator=g)
    dht = torch.randn(b, hv, d_k, d_v, generator=g)
    return q, k, w, g2, g_last, do, dv_local, dht


def _raise_on_sync() -> None:
    raise AssertionError("torch.cuda.synchronize called inside graph capture")


class TestScalarBatchedCaptureSafe:
    def test_k2_batched_no_sync_under_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gdn2_bwd_wy, "is_available", lambda: True)
        monkeypatch.setattr(gdn2_bwd_dhu, "_bmm_tc", _cpu_bmm)
        monkeypatch.setattr(torch.cuda, "synchronize", _raise_on_sync)
        with graph_capture():
            gdn2_bwd_wy.run_k2_batched(*_k2_inputs())  # must not raise

    def test_k2_batched_syncs_once_outside_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gdn2_bwd_wy, "is_available", lambda: True)
        monkeypatch.setattr(gdn2_bwd_dhu, "_bmm_tc", _cpu_bmm)
        calls = {"n": 0}
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda: calls.__setitem__("n", calls["n"] + 1)
        )
        gdn2_bwd_wy.run_k2_batched(*_k2_inputs())
        assert calls["n"] == 1

    def test_k1_batched_no_sync_under_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gdn2_bwd_dhu, "is_available", lambda: True)
        monkeypatch.setattr(gdn2_bwd_dhu, "_bmm_tc", _cpu_bmm)
        monkeypatch.setattr(torch.cuda, "synchronize", _raise_on_sync)
        with graph_capture():
            gdn2_bwd_dhu.run_k1_incB_batched(*_k1_inputs())  # must not raise

    def test_k1_batched_syncs_once_outside_capture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gdn2_bwd_dhu, "is_available", lambda: True)
        monkeypatch.setattr(gdn2_bwd_dhu, "_bmm_tc", _cpu_bmm)
        calls = {"n": 0}
        monkeypatch.setattr(
            torch.cuda, "synchronize", lambda: calls.__setitem__("n", calls["n"] + 1)
        )
        gdn2_bwd_dhu.run_k1_incB_batched(*_k1_inputs())
        assert calls["n"] == 1
