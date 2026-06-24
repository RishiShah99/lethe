"""The shape-gated default scan-mode selector (Branch B of the boundary study).

``_default_scan_mode`` is the launch default consulted when ``config.scan_mode``
is unset; it routes to serial only in the saturated short-sequence corner the
boundary sweep measured (``results/scan_mode_boundary.json``) and chunk_parallel
everywhere else. These pin the rule's decisive cases plus its aggregate quality
against the committed sweep (geomean speedup vs the old serial default, and the
no-bad-regression guard).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import geometric_mean

import pytest

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops.forward_chunked_scan import (
    _default_scan_mode,
    _resolve_scan_mode,
)

_BOUNDARY_JSON = Path(__file__).resolve().parents[1] / "results" / "scan_mode_boundary.json"


class TestDefaultScanMode:
    def test_forward_short_l_is_serial_every_batch_width(self) -> None:
        # The forward kernel is cheap per step; below L=512 the carry never pays,
        # at every batch and width the sweep tested.
        for batch in (1, 2, 8):
            for width in (256, 1024, 2048, 4096):
                assert _default_scan_mode(512, batch, width, is_forward=True) == "serial"

    def test_backward_short_l_small_batch_is_chunk_parallel(self) -> None:
        # The backward regime favours serial only when saturated; small batch at
        # short L is still a chunk_parallel win.
        assert _default_scan_mode(512, 1, 256, is_forward=False) == "chunk_parallel"
        assert _default_scan_mode(512, 2, 4096, is_forward=False) == "chunk_parallel"

    def test_saturated_corner_is_serial(self) -> None:
        assert _default_scan_mode(512, 8, 2048, is_forward=False) == "serial"
        assert _default_scan_mode(4096, 8, 4096, is_forward=False) == "serial"
        assert _default_scan_mode(512, 8, 4096, is_forward=True) == "serial"

    def test_long_l_is_chunk_parallel_even_saturated(self) -> None:
        # The carry amortises at long L; chunk_parallel wins even at b8.
        assert _default_scan_mode(16384, 8, 2048, is_forward=False) == "chunk_parallel"
        assert _default_scan_mode(16384, 8, 4096, is_forward=False) == "chunk_parallel"
        assert _default_scan_mode(16384, 8, 1024, is_forward=True) == "chunk_parallel"

    def test_under_saturated_is_chunk_parallel(self) -> None:
        assert _default_scan_mode(4096, 1, 1024, is_forward=True) == "chunk_parallel"
        assert _default_scan_mode(2048, 2, 2048, is_forward=False) == "chunk_parallel"


class TestResolveScanMode:
    def test_explicit_mode_overrides_default(self) -> None:
        # A saturated shape the default would route to serial, forced to cp.
        cfg = KernelConfig(scan_mode="chunk_parallel")
        assert _resolve_scan_mode(cfg, 512, 8, 4096, is_forward=False) == "chunk_parallel"
        cfg_s = KernelConfig(scan_mode="serial")
        assert _resolve_scan_mode(cfg_s, 16384, 1, 1024, is_forward=True) == "serial"

    def test_none_config_uses_default(self) -> None:
        assert _resolve_scan_mode(None, 16384, 1, 1024, is_forward=True) == "chunk_parallel"
        assert _resolve_scan_mode(None, 512, 8, 4096, is_forward=True) == "serial"

    def test_config_without_scan_mode_uses_default(self) -> None:
        # A config that tunes other knobs but leaves scan_mode None still defaults.
        cfg = KernelConfig(num_warps=4)
        assert _resolve_scan_mode(cfg, 16384, 1, 1024, is_forward=True) == "chunk_parallel"
        assert _resolve_scan_mode(cfg, 512, 8, 4096, is_forward=True) == "serial"


@pytest.mark.skipif(not _BOUNDARY_JSON.exists(), reason="boundary sweep artifact absent")
class TestAgainstBoundarySweep:
    """The shipped rule still captures the measured win on the committed sweep."""

    def _rows(self) -> list[tuple[str, int, int, int, float, float]]:
        rows = []
        for e in json.loads(_BOUNDARY_JSON.read_text()):
            if e["winner"] == "skipped":
                continue
            bs = (e.get("best_serial") or {}).get("speedup") or 0.0
            bc = (e.get("best_chunk_parallel") or {}).get("speedup") or 0.0
            rows.append((e["op"], e["batch"], e["seq_len"], e["width"], bs, bc))
        return rows

    def test_geomean_speedup_near_oracle(self) -> None:
        sp = []
        for op, b, length, w, bs, bc in self._rows():
            mode = _default_scan_mode(length, b, w, is_forward=(op == "forward_chunked_scan"))
            sp.append(bs if mode == "serial" else (bc or bs))
        # Always-serial (the old shipped default) is ~1.0; the selector clears 2.1x.
        assert geometric_mean([s for s in sp if s > 0]) >= 2.1

    def test_no_bad_regression_when_routing_chunk_parallel(self) -> None:
        # Wherever the rule picks chunk_parallel, it is never materially slower
        # than best-tuned serial at that shape (worst measured ratio ~0.955).
        for op, b, length, w, bs, bc in self._rows():
            mode = _default_scan_mode(length, b, w, is_forward=(op == "forward_chunked_scan"))
            if mode == "chunk_parallel" and bs > 0:
                assert bc >= 0.94 * bs, f"{op} b{b}/L{length}/d{w}: cp {bc} vs serial {bs}"
