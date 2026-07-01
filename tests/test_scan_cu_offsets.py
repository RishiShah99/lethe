"""Regression: scan.cu offset arithmetic is 64-bit (int64_t), not ``long``.

``long`` is 32-bit under LLP64 (Windows/MSVC), so large-tensor flat offsets
would overflow there; the Linux B200 build target (LP64) is unaffected, but the
explicit ``int64_t`` makes the width correct everywhere. Source-level check —
the ``.cu`` compiles only on-box via ``load_inline``.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCAN_CU = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "flash_mamba_rl"
    / "kernels"
    / "cuda"
    / "scan.cu"
)


def test_scan_cu_uses_int64_offsets_not_long() -> None:
    src = _SCAN_CU.read_text(encoding="utf-8")
    assert "#include <cstdint>" in src
    assert "static_cast<int64_t>" in src
    assert not re.search(r"\blong\b", src), "scan.cu still has a 32-bit-under-LLP64 `long` offset"
