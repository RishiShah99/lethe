"""JIT build of the CUDA scan extension via ``torch.utils.cpp_extension``.

Centralises two things every kernel here needs: the CCCL include path that
CUDA 13 relocated (``<cuda>/include/cccl``; nvcc does not add it to the default
search path, so ``#include <cub/...>`` fails without an explicit ``-I``), and a
single cached compile so repeated launches reuse one ``.so``. The compile
itself is deferred to first call — importing this module never needs nvcc, so
the package imports cleanly on a CPU-only dev box (the quality gates run there).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

_CUDA_DIR = Path(__file__).resolve().parent


def _cccl_include_flags() -> list[str]:
    """``-I`` for the CCCL (cub/thrust/libcu++) headers, empty if not relocated.

    CUDA 13 moved CCCL from ``<cuda>/include`` to ``<cuda>/include/cccl``; on
    CUDA 12 the headers sit in the default include dir and this returns ``[]``.
    """
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
        # No --use_fast_math: it flushes denormals (-ftz) and swaps in
        # approximate exp, which splits the EXC-01 NaN/Inf masks the verifier
        # checks (the C1 libdevice-vs-ex2.approx lesson). Optimisation (exp2,
        # float4) is applied explicitly per-kernel at Inc 4, not blanket.
        extra_cuda_cflags=["-O3", *_cccl_include_flags()],
        verbose=False,
    )
