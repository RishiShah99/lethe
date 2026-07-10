"""Regression: scan.cu offset arithmetic is 64-bit (int64_t), not ``long``."""

from __future__ import annotations

import re
from pathlib import Path

_SCAN_CU = Path(__file__).resolve().parent.parent / "src" / "lethe" / "kernels" / "cuda" / "scan.cu"


def test_scan_cu_uses_int64_offsets_not_long() -> None:
    src = _SCAN_CU.read_text(encoding="utf-8")
    assert "#include <cstdint>" in src
    assert "static_cast<int64_t>" in src
    assert not re.search(r"\blong\b", src), "scan.cu still has a 32-bit-under-LLP64 `long` offset"
