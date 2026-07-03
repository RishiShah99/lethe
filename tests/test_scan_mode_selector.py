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

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops.forward_chunked_scan import (
    _chunk_parallel_bwd_scratch_bytes,
    _default_scan_mode,
    _next_power_of_2,
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


class TestMemoryBoundFallback:
    """CMP-05: selector falls back to serial when scratch would exceed memory."""

    def test_scratch_bytes_formula_matches_block_sizing(self) -> None:
        # Verify the helper computes the same formula as the kernels:
        # hbuf_elems = batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n
        batch, seq_len, d_model, n_state, chunk_len = 4, 2048, 1024, 16, 128
        block_n = _next_power_of_2(n_state)
        block_d = min(64, max(16, 2048 // block_n))
        n_chunks = seq_len // chunk_len
        n_d_blocks = (d_model + block_d - 1) // block_d
        expected = batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n * 4

        actual = _chunk_parallel_bwd_scratch_bytes(batch, seq_len, d_model, n_state, chunk_len)
        assert actual == expected

    def test_large_n_state_uses_more_memory(self) -> None:
        # At larger n_state, block_d shrinks but block_n grows, and n_d_blocks
        # increases, so total scratch is larger.
        small = _chunk_parallel_bwd_scratch_bytes(4, 4096, 1024, 16, 128)
        large = _chunk_parallel_bwd_scratch_bytes(4, 4096, 1024, 128, 128)
        assert large > small

    def test_selector_keeps_chunk_parallel_when_n_state_missing(self) -> None:
        # Without n_state, the selector cannot check memory — it uses the
        # shape-based heuristics only. For a shape that would normally pick
        # chunk_parallel, it still does.
        assert _default_scan_mode(4096, 2, 1024, is_forward=False) == "chunk_parallel"

    def test_selector_returns_serial_for_overcap_n_state(self) -> None:
        # The chunk-parallel backward launchers reject n_state > 128 outright;
        # the selector must not route there.
        assert _default_scan_mode(4096, 2, 1024, is_forward=False, n_state=256) == "serial"

    def test_selector_returns_serial_for_memory_exceeding_shape(self) -> None:
        # A shape with huge batch * seq_len * d_model * n_state that would exceed
        # any plausible GPU memory budget should fall back to serial. On CPU-only
        # test runners (torch.cuda.is_available() == False), the selector skips
        # the memory check and returns chunk_parallel, so we test the formula
        # directly in those environments.
        import torch

        batch, seq_len, d_model, n_state = 64, 65536, 4096, 128
        chunk_len = 512
        scratch = _chunk_parallel_bwd_scratch_bytes(batch, seq_len, d_model, n_state, chunk_len)
        # This shape needs multiple TB of scratch — exceeds any current GPU
        assert scratch > 100 * 1e9

        if torch.cuda.is_available():
            # On GPU, selector should return serial
            mode = _default_scan_mode(seq_len, batch, d_model, is_forward=False, n_state=n_state)
            assert mode == "serial"

    def test_selector_still_picks_chunk_parallel_for_small_shapes(self) -> None:
        # A typical training shape (batch=2, L=2048, d=1024, n_state=16) should
        # still route to chunk_parallel with the memory check.
        batch, seq_len, d_model, n_state = 2, 2048, 1024, 16
        scratch = _chunk_parallel_bwd_scratch_bytes(batch, seq_len, d_model, n_state, 128)
        # ~268 MB — fits easily on any modern GPU
        assert scratch < 500 * 1e6

        mode = _default_scan_mode(seq_len, batch, d_model, is_forward=False, n_state=n_state)
        assert mode == "chunk_parallel"


class TestAgainstBoundarySweep:
    """The shipped rule still captures the measured win on the committed sweep."""

    def test_boundary_json_exists(self) -> None:
        # git-tracked artifact — its absence is a regression, not a skip.
        assert _BOUNDARY_JSON.exists(), f"tracked boundary sweep JSON missing: {_BOUNDARY_JSON}"

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
