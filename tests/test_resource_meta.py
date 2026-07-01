"""CPU regressions for the RES-02 resource-envelope helpers.

Pins the review MEDIUM fix: when the scan mode is resolved by shape (config
unset), the resource audit must bound *whichever* kernel dispatch could pick,
not the serial one only — a chunk-parallel specialisation that spills registers
must not hide behind the serial envelope. Triton isn't importable off-box, so
the routing tests stub the lazily-imported kernel modules on the package.
"""

from __future__ import annotations

import importlib
import types

import pytest

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops._resource_meta import max_resource_meta

# The ops package __init__ rebinds ``backward_selective_scan`` /
# ``fused_block_backward`` to the *functions*, shadowing the submodule
# attributes — reach the real modules through sys.modules.
ops_pkg = importlib.import_module("flash_mamba_rl.kernels.ops")
bwd_mod = importlib.import_module("flash_mamba_rl.kernels.ops.backward_selective_scan")
fused_mod = importlib.import_module("flash_mamba_rl.kernels.ops.fused_block_backward")

_SERIAL_META = {"n_regs": 64, "shared_bytes": 2048}
_CP_META = {"n_regs": 96, "shared_bytes": 1024, "spill_bytes": 32}
_ENVELOPE = {"n_regs": 96, "shared_bytes": 2048, "spill_bytes": 32}


class TestMaxResourceMeta:
    def test_elementwise_max_over_union_of_keys(self) -> None:
        assert max_resource_meta(_SERIAL_META, _CP_META) == _ENVELOPE

    def test_none_is_absence_not_zero(self) -> None:
        meta = {"n_regs": 10}
        assert max_resource_meta(None, None) is None
        assert max_resource_meta(meta, None) == meta
        assert max_resource_meta(None, meta) == meta


@pytest.fixture
def _stub_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bwd_mod, "_triton_usable", lambda: True)
    monkeypatch.setattr(fused_mod, "_triton_usable", lambda: True)
    serial = types.SimpleNamespace(resource_meta=lambda: dict(_SERIAL_META))
    chunk_parallel = types.SimpleNamespace(resource_meta=lambda: dict(_CP_META))
    for name in ("_triton_bwd_scan", "_triton_fused_block_bwd"):
        monkeypatch.setattr(ops_pkg, name, serial, raising=False)
    for name in ("_triton_chunk_parallel_bwd", "_triton_chunk_parallel_fused_bwd"):
        monkeypatch.setattr(ops_pkg, name, chunk_parallel, raising=False)


@pytest.mark.parametrize(
    "resource_meta_fn",
    [
        lambda cfg: bwd_mod.triton_bwd_scan_resource_meta(config=cfg),
        lambda cfg: fused_mod.triton_fused_block_bwd_resource_meta(config=cfg),
    ],
    ids=["c2_bwd_scan", "c6_fused_bwd"],
)
class TestResourceMetaRouting:
    def test_config_none_audits_max_envelope(self, _stub_kernels: None, resource_meta_fn) -> None:
        # Dispatch resolves the mode by shape; the audit must bound both.
        meta = resource_meta_fn(None)
        assert meta == _ENVELOPE
        assert meta["n_regs"] == _CP_META["n_regs"], "chunk-parallel regs hidden by serial envelope"
        assert "spill_bytes" in meta, "chunk-parallel spill invisible to RES-02"

    def test_scan_mode_unset_audits_max_envelope(
        self, _stub_kernels: None, resource_meta_fn
    ) -> None:
        assert resource_meta_fn(KernelConfig()) == _ENVELOPE

    def test_explicit_chunk_parallel_audits_cp_only(
        self, _stub_kernels: None, resource_meta_fn
    ) -> None:
        assert resource_meta_fn(KernelConfig(scan_mode="chunk_parallel")) == _CP_META

    def test_explicit_serial_audits_serial_only(
        self, _stub_kernels: None, resource_meta_fn
    ) -> None:
        assert resource_meta_fn(KernelConfig(scan_mode="serial")) == _SERIAL_META
