"""Tests for the kernel candidate loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from lethe.kernels.loader import (
    KernelCandidate,
    discover_candidates,
    import_candidate,
    load_candidate,
)


def _write(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


class TestLoadCandidate:
    def test_basic_load(self, tmp_path: Path) -> None:
        src = "def my_kernel(x):\n    return x + 1\n"
        path = _write(tmp_path / "kernel_my_kernel.py", src)
        cand = load_candidate(path)
        assert isinstance(cand, KernelCandidate)
        assert cand.callable_name == "my_kernel"
        assert cand.source_code == src
        assert cand.target_op is None
        assert cand.metadata == {}

    def test_target_op_metadata_picked_up(self, tmp_path: Path) -> None:
        src = (
            '__candidate_op__ = "forward_chunked_scan"\n'
            '__candidate_meta__ = {"author": "rl_agent_v3", "iter": 42}\n'
            "def scan(x):\n    return x\n"
        )
        path = _write(tmp_path / "kernel_scan.py", src)
        cand = load_candidate(path)
        assert cand.target_op == "forward_chunked_scan"
        assert cand.metadata == {"author": "rl_agent_v3", "iter": 42}

    def test_explicit_callable_override(self, tmp_path: Path) -> None:
        src = '__candidate_callable__ = "real_entry"\ndef real_entry(x):\n    return x * 2\n'
        path = _write(tmp_path / "kernel_anything.py", src)
        cand = load_candidate(path)
        assert cand.callable_name == "real_entry"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_candidate(tmp_path / "nope.py")

    def test_invalid_syntax_loads_without_eval(self, tmp_path: Path) -> None:
        """Module-attr parsing must not crash on bad syntax; metadata just returns empty."""
        src = "__candidate_op__ = 'forward'\ndef broken(:\n    pass\n"
        path = _write(tmp_path / "kernel_broken.py", src)
        cand = load_candidate(path)
        assert cand.target_op is None  # ast.parse failed, attrs empty
        assert cand.callable_name == "broken"

    def test_literal_eval_typeerror_skipped(self, tmp_path: Path) -> None:
        """ast.literal_eval can raise TypeError; discovery must skip it, not crash."""
        src = (
            # unhashable list key -> TypeError; a bare name like open raises ValueError instead.
            "__candidate_op__ = {[]: 1}\ndef typeerr_kernel(x):\n    return x\n"
        )
        path = _write(tmp_path / "kernel_typeerr.py", src)
        cand = load_candidate(path)
        assert cand.target_op is None
        assert cand.callable_name == "typeerr"


class TestDiscoverCandidates:
    def test_discovers_kernel_files(self, tmp_path: Path) -> None:
        _write(tmp_path / "kernel_a.py", "def a(x): return x\n")
        _write(tmp_path / "kernel_b.py", "def b(x): return x\n")
        _write(tmp_path / "not_a_kernel.py", "def foo(): pass\n")
        _write(tmp_path / "README.md", "noise\n")

        cands = discover_candidates(tmp_path)
        names = [c.callable_name for c in cands]
        assert names == ["a", "b"]

    def test_deterministic_order(self, tmp_path: Path) -> None:
        _write(tmp_path / "kernel_z.py", "def z(x): return x\n")
        _write(tmp_path / "kernel_a.py", "def a(x): return x\n")
        _write(tmp_path / "kernel_m.py", "def m(x): return x\n")

        cands = discover_candidates(tmp_path)
        assert [c.callable_name for c in cands] == ["a", "m", "z"]

    def test_empty_directory(self, tmp_path: Path) -> None:
        cands = discover_candidates(tmp_path)
        assert cands == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            discover_candidates(tmp_path / "missing")


class TestImportCandidate:
    def test_imports_and_runs_kernel(self, tmp_path: Path) -> None:
        src = "def add_one(x):\n    return x + 1\n"
        path = _write(tmp_path / "kernel_add_one.py", src)
        cand = load_candidate(path)
        fn = import_candidate(cand)
        assert fn(41) == 42

    def test_missing_callable_raises(self, tmp_path: Path) -> None:
        src = '__candidate_callable__ = "missing"\ndef other(x): return x\n'
        path = _write(tmp_path / "kernel_oops.py", src)
        cand = load_candidate(path)
        with pytest.raises(AttributeError):
            import_candidate(cand)

    def test_each_import_is_fresh(self, tmp_path: Path) -> None:
        """Two imports produce independent modules, so RL sweeps see source overwrites."""
        path = tmp_path / "kernel_v.py"
        _write(path, "def v(x): return x * 10\n")
        fn1 = import_candidate(load_candidate(path))
        assert fn1(2) == 20

        _write(path, "def v(x): return x * 100\n")
        fn2 = import_candidate(load_candidate(path))
        assert fn2(2) == 200
