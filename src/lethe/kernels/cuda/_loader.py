"""JIT build of the CUDA scan extension via ``torch.utils.cpp_extension``."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

_CUDA_DIR = Path(__file__).resolve().parent


def _cccl_include_flags() -> list[str]:
    """``-I`` for the CCCL (cub/thrust/libcu++) headers, empty if not relocated."""
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    base = Path(cuda_home)
    for inc in (base / "include" / "cccl", base / "targets" / "x86_64-linux" / "include" / "cccl"):
        if (inc / "cub").is_dir():
            return [f"-I{inc}"]
    return []


@cache
def load_scan_extension() -> Any:
    """Compile + load the CUDA scan extension once; cached for the process."""
    from torch.utils.cpp_extension import load

    return load(
        name="fmr_cuda_scan",
        sources=[str(_CUDA_DIR / "scan.cu")],
        # No --use_fast_math: flushes denormals + approx exp, breaks the EXC-01 NaN/Inf masks.
        extra_cuda_cflags=["-O3", *_cccl_include_flags()],
        verbose=False,
    )
