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

    def test_missing_chunk_parallel_measurement_dropped_not_misattributed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When the selector picks chunk_parallel but that measurement is absent,
        # the row is dropped — never credited the serial speedup, which would
        # inflate the >=2.1x geomean gate.
        entries = [
            {  # saturated corner routes to serial -> contributes its 2.0x
                "op": "backward_selective_scan",
                "seq_len": 512,
                "batch": 8,
                "width": 4096,
                "winner": "serial",
                "best_serial": {"speedup": 2.0},
            },
            {  # routes to chunk_parallel but chunk_parallel not benched -> drop
                "op": "backward_selective_scan",
                "seq_len": 16384,
                "batch": 1,
                "width": 1024,
                "winner": "serial",
                "best_serial": {"speedup": 9.0},
            },
        ]
        (tmp_path / "scan_mode_boundary.json").write_text(json.dumps(entries))
        monkeypatch.setattr(repro, "RESULTS_DIR", tmp_path)
        gm, n = repro.compute_selector_geomean()
        assert n == 1  # the chunk_parallel-routed row with no bc measurement is dropped
        assert gm == pytest.approx(2.0)


class TestAuditHeadline:
    def test_audit_json_exists(self) -> None:
        # git-tracked artifact — absence is a regression, not a skip.
        assert _AUDIT_JSON.exists(), f"tracked audit aggregate JSON missing: {_AUDIT_JSON}"

    def test_finding_rate_plausible(self) -> None:
        ok, detail = repro.check_audit_headline()
        assert ok, f"audit headline check failed: {detail}"
        # Pin the committed headline on the COMPUTED portion of the detail string —
        # the "(committed~62.1%)" suffix is a constant and must not satisfy the pin.
        assert "finding_rate=0.6213" in detail, f"committed rate not in detail: {detail}"

    def test_raw_finding_rate_value(self) -> None:
        data = json.loads(_AUDIT_JSON.read_text())
        rate = data["accepted_only"]["finding_rate"]
        # The committed value is 0.6213 — within 2pp is a sound check.
        assert abs(rate - 0.6213) < 0.02, f"finding_rate {rate} deviates >2pp from committed"


class TestMicrogateHarnessIntegrity:
    """The K#1/K#2 silicon-gate harnesses must track the promoted kernel modules.

    After the scratch->src promotion (1c755a0) the k1 harness kept importing the
    deleted scratch module; the import sits inside the harness's GO try-block, so
    a rerun of the tracked GO evidence wrote GO=False — a broken harness
    masquerading as a failed gate. Pin the promoted paths off-box.
    """

    def test_harnesses_reference_only_promoted_kernel_modules(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("k1_microgate.py", "k2_microgate.py"):
            text = (root / "scratch" / name).read_text(encoding="utf-8")
            assert "from scratch import" not in text, f"{name} imports a pre-promotion module"
            assert "lethe.kernels.cute" in text, f"{name} lost the promoted import"

    def test_promoted_kernel_modules_import_without_gpu(self) -> None:
        import importlib

        for mod in (
            "lethe.kernels.cute.gdn2_bwd_dhu",
            "lethe.kernels.cute.gdn2_bwd_wy",
        ):
            m = importlib.import_module(mod)
            assert hasattr(m, "is_available")


class TestCpuSuiteGate:
    """Regression coverage for check_cpu_suite exit-code handling."""

    def test_zero_collected_is_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exit code 5 (zero tests collected) must report FAIL, not PASS."""
        import subprocess
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 5
        mock_result.stdout = ""
        mock_result.stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        ok, detail = repro.check_cpu_suite()
        assert not ok, "exit 5 (zero collected) should be FAIL"
        assert "zero tests collected" in detail.lower() or "exit 5" in detail.lower()

    def test_exit_zero_with_zero_passed_is_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exit 0 but '0 passed' in summary must report FAIL."""
        import subprocess
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0 passed in 0.01s"
        mock_result.stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        ok, _detail = repro.check_cpu_suite()
        assert not ok, "exit 0 with '0 passed' should be FAIL"

    def test_exit_zero_with_passed_count_is_pass(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exit 0 with nonzero passed count is a real PASS."""
        import subprocess
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "705 passed, 112 skipped in 12.34s"
        mock_result.stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        ok, detail = repro.check_cpu_suite()
        assert ok, "exit 0 with '705 passed' should be PASS"
        assert "705 passed" in detail

    @pytest.mark.parametrize("code", [1, 2, 3, 4])
    def test_nonzero_exit_other_than_five_is_fail(
        self, monkeypatch: pytest.MonkeyPatch, code: int
    ) -> None:
        """Any nonzero exit code (2=interrupted, 3=internal, 4=usage) is FAIL."""
        import subprocess
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.returncode = code
        mock_result.stdout = "some error output"
        mock_result.stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)
        ok, _ = repro.check_cpu_suite()
        assert not ok, f"exit {code} should be FAIL"
