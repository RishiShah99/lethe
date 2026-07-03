"""Stage-B einsum kernel wiring — CPU pins (the squeeze after the K#2 fusion).

The kernel itself is box-only; these pin what is CPU-checkable: import-clean off-box,
the dim lock, the off-box raise, and that the closed stage-B's torch fallback stays
bitwise intact under the kill-switch env (on CPU both calls take the torch path — the
pin is that the env read introduces no drift). Silicon: scratch/sbe_microgate.py +
the burst-6 integration gates.
"""

from __future__ import annotations

import pytest
import torch

import flash_mamba_rl.kernels.cute.gdn2_sb_einsum as sbe
from flash_mamba_rl.kernels.cute.gdn2_assemble import (
    _stage_b_vjp_cw_closed,
    _to_chunks,
    k1_reverse_state_cw_ref,
)
from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import chunkwise_restage_cw


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


def test_closed_stage_b_torch_path_stable_under_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gen = torch.Generator().manual_seed(31)
    dt = torch.float64
    b_, t, h, d_k, d_v, cl = 1, 32, 2, 16, 16, 16
    q = torch.randn(b_, t, h, d_k, generator=gen, dtype=dt)
    k = torch.randn(b_, t, h, d_k, generator=gen, dtype=dt)
    v = torch.randn(b_, t, h, d_v, generator=gen, dtype=dt)
    g = -torch.rand(b_, t, h, d_k, generator=gen, dtype=dt) * 0.1
    bg = torch.rand(b_, t, h, d_k, generator=gen, dtype=dt)
    wg = torch.rand(b_, t, h, d_v, generator=gen, dtype=dt)
    do = torch.randn(b_, t, h, d_v, generator=gen, dtype=dt)

    rst = chunkwise_restage_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    do_c = _to_chunks(do, cl)
    dv_local = rst.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(rst.h_list[0])
    dh, dv2, _dh0 = k1_reverse_state_cw_ref(
        rst.q, rst.k, rst.wy, rst.g2, rst.g_last, do_c, dv_local, dht
    )
    base = _stage_b_vjp_cw_closed(rst, do, dh, dv2, create_graph=False)
    monkeypatch.setenv("FMR_DISABLE_SBE", "1")
    pinned = _stage_b_vjp_cw_closed(rst, do, dh, dv2, create_graph=False)
    for a, c_ in zip(base, pinned, strict=True):
        assert torch.equal(a, c_)
