"""Tests for the reproducibility surface helpers in scripts/repro.py.

These are fast, CPU-only tests that cover:
- environment capture returns required keys
- seed pinning is repeatable
- selector geomean helper returns a value ≥ 2.1 against the committed JSON
- audit headline helper returns a plausible value from the committed JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# scripts/ is not a package; add the repo root's scripts dir to the path so
# we can import repro directly.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import repro  # noqa: E402  (path manipulation must precede this import)

_BOUNDARY_JSON = Path(__file__).resolve().parents[1] / "results" / "scan_mode_boundary.json"
_AUDIT_JSON = Path(__file__).resolve().parents[1] / "results" / "audit_drkernel.json"


class TestEnvCapture:
    def test_required_keys_present(self) -> None:
        env = repro.capture_env()
        for key in ("python", "platform", "torch", "numpy", "cuda_available"):
            assert key in env, f"missing key: {key}"

    def test_torch_version_matches_import(self) -> None:
        env = repro.capture_env()
        assert env["torch"] == torch.__version__

    def test_git_head_is_hex_or_none(self) -> None:
        env = repro.capture_env()
        sha = env.get("git_head")
        if sha is not None:
            assert len(sha) == 40
            int(sha, 16)  # raises if not hex

    def test_write_env_creates_valid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(repro, "RESULTS_DIR", tmp_path)
        env = repro.capture_env()
        out = repro.write_env(env)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["python"] == env["python"]


class TestSeedPinning:
    def test_same_seed_same_tensor(self) -> None:
        repro.pin_seeds(42)
        a = torch.randn(8)
        repro.pin_seeds(42)
        b = torch.randn(8)
        assert torch.equal(a, b)

    def test_different_seed_different_tensor(self) -> None:
        repro.pin_seeds(1)
        a = torch.randn(8)
        repro.pin_seeds(2)
        b = torch.randn(8)
        assert not torch.equal(a, b)


class TestSelectorGeomean:
    def test_boundary_json_exists(self) -> None:
        # git-tracked artifact — its absence is a regression, not a reason to
        # silently skip the only automated check of the 2.1x selector headline.
        assert _BOUNDARY_JSON.exists(), f"tracked boundary sweep JSON missing: {_BOUNDARY_JSON}"

    def test_geomean_at_least_2_1(self) -> None:
        gm, n = repro.compute_selector_geomean()
        assert n > 0
        assert gm >= 2.1, f"geomean {gm:.4f} < 2.1 threshold"

    def test_geomean_near_committed_value(self) -> None:
        gm, _ = repro.compute_selector_geomean()
        # Committed ~2.174; allow ±0.10 for floating-point/row-order variation.
        assert abs(gm - 2.174) < 0.10, f"geomean {gm:.4f} too far from committed 2.174"

    def test_returns_correct_row_count(self) -> None:
        entries = json.loads(_BOUNDARY_JSON.read_text())
        valid = sum(1 for e in entries if e.get("winner") != "skipped")
        _, n = repro.compute_selector_geomean()
        # n may be slightly smaller if some rows have zero speedup on both sides
        assert n <= valid
        assert n >= valid - 5  # at most 5 degenerate rows


class TestAuditHeadline:
    def test_audit_json_exists(self) -> None:
        # git-tracked artifact — absence is a regression, not a skip.
        assert _AUDIT_JSON.exists(), f"tracked audit aggregate JSON missing: {_AUDIT_JSON}"

    def test_finding_rate_plausible(self) -> None:
        ok, detail = repro.check_audit_headline()
        assert ok, f"audit headline check failed: {detail}"
        assert "62.1%" in detail or "0.621" in detail or "finding_rate" in detail

    def test_raw_finding_rate_value(self) -> None:
        data = json.loads(_AUDIT_JSON.read_text())
        rate = data["accepted_only"]["finding_rate"]
        # The committed value is 0.6213 — within 2pp is a sound check.
        assert abs(rate - 0.6213) < 0.02, f"finding_rate {rate} deviates >2pp from committed"
