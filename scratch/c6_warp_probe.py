"""Probe C6 num_warps x dtype: timing + per-spec ptxas resources.

The bench surfaced two anomalies: bf16 runs ~5x slower than fp32 at the
training shape, and num_warps=8 beats the heuristic's nw=4 by ~26% at
fp32. The specialization table shows a _bwd_sweep_kernel spec compiled at
n_regs=32 with 1.4-2 KB spill — a ptxas allocation collapse. This times
every (dtype, num_warps) cell and prints which spec each cell compiles,
to decide the launcher heuristic from measurement (the C4 convention).
Box usage:
fleet run "bash scratch/detach.sh uv run python scratch/c6_warp_probe.py"
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.ops import _triton_fused_block_bwd as mod
from flash_mamba_rl.verifier.timing import benchmark

SHAPES = [
    ("train", 8, 2048, 4096, 128, 4),
    ("long", 2, 16384, 4096, 128, 4),
]
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
WARPS = (2, 4, 8)


def _inputs(b: int, l_out: int, d: int, n: int, k: int, dtype: torch.dtype):  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    dev = torch.device("cuda")
    x = torch.randn(b, l_out + k - 1, d, device=dev).to(dtype)
    conv_w = (torch.randn(d, 1, k, device=dev) / math.sqrt(k)).to(dtype)
    conv_b = (0.5 * torch.randn(d, device=dev)).to(dtype)
    delta = torch.randn(b, l_out, d, device=dev).to(dtype)
    a = (-torch.rand(d, n, device=dev)).to(dtype)
    b_proj = torch.randn(b, l_out, n, device=dev).to(dtype)
    c_proj = torch.randn(b, l_out, n, device=dev).to(dtype)
    d_skip = torch.randn(d, device=dev).to(dtype)
    norm_w = (1.0 + 0.25 * torch.randn(d, device=dev)).to(dtype)
    dy = torch.randn(b, l_out, d, device=dev).to(dtype)
    return (x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w, dy)


def _specs() -> list[str]:
    rows = []
    for jit_fn in (mod._fwd_stage_kernel, mod._bwd_sweep_kernel):
        caches = getattr(jit_fn, "device_caches", None)
        if not isinstance(caches, dict):
            continue
        for entry in caches.values():
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            if not isinstance(cache_dict, dict):
                continue
            for kernel in cache_dict.values():
                md = getattr(kernel, "metadata", None)
                rows.append(
                    f"{getattr(md, 'name', '?')} nw={getattr(md, 'num_warps', '?')} "
                    f"regs={getattr(kernel, 'n_regs', '?')} "
                    f"spill={getattr(kernel, 'n_spills', '?')}"
                )
    return rows


def main() -> None:
    for label, b, l_out, d, n, k in SHAPES:
        for dtype in DTYPES:
            args = _inputs(b, l_out, d, n, k, dtype)
            for warps in WARPS:
                try:
                    result = benchmark(
                        lambda a=args, w=warps: mod.launch_fused_block_backward(
                            *a, 1e-5, num_warps=w
                        ),
                        (),
                        warmup=3,
                        trials=10,
                    )
                    print(
                        f"{label} {str(dtype).removeprefix('torch.'):9s} nw={warps} "
                        f"median={result.median_ms:9.2f} ms"
                    )
                except Exception as exc:
                    print(f"{label} {dtype} nw={warps} FAILED: {type(exc).__name__}: {exc}")
            args = ()
            torch.cuda.empty_cache()
    print("\nspecializations:")
    for row in sorted(set(_specs())):
        print(" ", row)


if __name__ == "__main__":
    main()
