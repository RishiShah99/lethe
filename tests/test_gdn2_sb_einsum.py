"""Stage-B einsum kernel wiring — CPU pins (the squeeze after the K#2 fusion).

The kernel itself is box-only; these pin what is CPU-checkable: import-clean off-box,
the dim lock, and that the FMR_DISABLE_SBE kill-switch actually flips the selector
away from the SBE kernel path. Silicon: scratch/sbe_microgate.py + the burst-6
integration gates.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

import lethe.kernels.cute.gdn2_sb_einsum as sbe
from lethe.kernels.cute.gdn2_assemble import (
    _stage_b_vjp_cw_closed,
    _to_chunks,
    k1_reverse_state_cw_ref,
)
from lethe.kernels.references.gdn2_chunkwise_cw import chunkwise_restage_cw


def test_module_imports_cleanly_off_box() -> None:
    assert hasattr(sbe, "run_sb_einsum")
    assert sbe.sbe_dims_ok(64, 128)
    assert not sbe.sbe_dims_ok(64, 64)
    assert not sbe.sbe_dims_ok(32, 128)
    if not sbe.is_available():
        with pytest.raises(RuntimeError, match="CuTe DSL"):
            sbe.run_sb_einsum(
                torch.zeros(1, 1, 1, 64, 64),
                torch.zeros(1, 1, 1, 64, 128),
                torch.zeros(1, 1, 1, 64, 128),
                torch.zeros(1, 1, 1, 64, 128),
            )


def test_kill_switch_bypasses_sbe_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin that FMR_DISABLE_SBE=1 flips the selector AWAY from run_sb_einsum.

    Monkeypatch is_available/sbe_dims_ok so the selector logic runs on CPU, and
    run_sb_einsum to a stub that tracks invocation. With the kill-switch unset the
    selector routes to the kernel (hit=1); with FMR_DISABLE_SBE=1 it routes to the
    torch einsum fallback (hit=0). Mirrors the pattern in test_gdn2_l3.py.
    """
    hit = {"sbe": 0}

    def _stub_run_sb_einsum(
        da_qk: Tensor, k: Tensor, q: Tensor, g2: Tensor
    ) -> tuple[Tensor, Tensor]:
        hit["sbe"] += 1
        return (torch.zeros_like(k), torch.zeros_like(k))

    monkeypatch.setattr("lethe.kernels.cute.gdn2_sb_einsum.is_available", lambda: True)
    monkeypatch.setattr("lethe.kernels.cute.gdn2_sb_einsum.sbe_dims_ok", lambda c, d_k: True)
    monkeypatch.setattr("lethe.kernels.cute.gdn2_sb_einsum.run_sb_einsum", _stub_run_sb_einsum)

    gen = torch.Generator().manual_seed(42)
    b_, t, h, d_k, d_v, cl = 1, 64, 1, 128, 64, 64
    q = torch.randn(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    k = torch.randn(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    v = torch.randn(b_, t, h, d_v, generator=gen, dtype=torch.float32)
    g = -torch.rand(b_, t, h, d_k, generator=gen, dtype=torch.float32) * 0.1
    bg = torch.rand(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    wg = torch.rand(b_, t, h, d_v, generator=gen, dtype=torch.float32)
    do = torch.randn(b_, t, h, d_v, generator=gen, dtype=torch.float32)

    class _FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    do_cuda = do.as_subclass(_FakeCudaTensor)

    rst = chunkwise_restage_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    do_c = _to_chunks(do, cl)
    dv_local = rst.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(rst.h_list[0])
    dh, dv2, _dh0 = k1_reverse_state_cw_ref(
        rst.q, rst.k, rst.wy, rst.g2, rst.g_last, do_c, dv_local, dht
    )

    hit["sbe"] = 0
    _stage_b_vjp_cw_closed(rst, do_cuda, dh, dv2, create_graph=False)
    assert hit["sbe"] == 1, "SBE kernel should be called when kill-switch is unset"

    monkeypatch.setenv("FMR_DISABLE_SBE", "1")
    hit["sbe"] = 0
    _stage_b_vjp_cw_closed(rst, do_cuda, dh, dv2, create_graph=False)
    assert hit["sbe"] == 0, "SBE kernel should NOT be called when FMR_DISABLE_SBE=1"


def test_kill_switch_selector_off_tile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin that the selector skips SBE when dims are off-tile, regardless of env."""
    hit = {"sbe": 0}

    def _stub_run_sb_einsum(
        da_qk: Tensor, k: Tensor, q: Tensor, g2: Tensor
    ) -> tuple[Tensor, Tensor]:
        hit["sbe"] += 1
        return (torch.zeros_like(k), torch.zeros_like(k))

    monkeypatch.setattr("lethe.kernels.cute.gdn2_sb_einsum.is_available", lambda: True)
    monkeypatch.setattr("lethe.kernels.cute.gdn2_sb_einsum.run_sb_einsum", _stub_run_sb_einsum)

    gen = torch.Generator().manual_seed(43)
    b_, t, h, d_k, d_v, cl = 1, 64, 1, 64, 64, 64
    q = torch.randn(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    k = torch.randn(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    v = torch.randn(b_, t, h, d_v, generator=gen, dtype=torch.float32)
    g = -torch.rand(b_, t, h, d_k, generator=gen, dtype=torch.float32) * 0.1
    bg = torch.rand(b_, t, h, d_k, generator=gen, dtype=torch.float32)
    wg = torch.rand(b_, t, h, d_v, generator=gen, dtype=torch.float32)
    do = torch.randn(b_, t, h, d_v, generator=gen, dtype=torch.float32)

    class _FakeCudaTensor(torch.Tensor):
        @property
        def is_cuda(self) -> bool:
            return True

    do_cuda = do.as_subclass(_FakeCudaTensor)

    rst = chunkwise_restage_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    do_c = _to_chunks(do, cl)
    dv_local = rst.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(rst.h_list[0])
    dh, dv2, _dh0 = k1_reverse_state_cw_ref(
        rst.q, rst.k, rst.wy, rst.g2, rst.g_last, do_c, dv_local, dht
    )

    hit["sbe"] = 0
    _stage_b_vjp_cw_closed(rst, do_cuda, dh, dv2, create_graph=False)
    assert hit["sbe"] == 0, "SBE kernel should NOT be called when d_k=64 (off-tile)"
