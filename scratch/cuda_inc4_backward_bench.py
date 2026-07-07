"""Inc 4 — CUDA backward speed vs Triton backward and the crippled official.

The make-or-break number: does the parallel-L CUDA backward beat the
#904-crippled official Mamba-3 backward (~12.5 ms @ B8/L2048/D4096/N128) WITHOUT
tl.dot? Times the CUDA path (a few items/block_d configs), our Triton backward
(serial + chunk_parallel), and best-effort the official crippled backward.

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \
        CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/cuda_inc4_backward_bench.py
"""

from __future__ import annotations

import torch

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.cuda.backward import cuda_backward_scan
from lethe.kernels.ops import backward_selective_scan
from lethe.verifier.timing import benchmark

SHAPES = [(8, 2048, 4096, 128), (8, 4096, 4096, 128), (2, 16384, 4096, 128)]
CUDA_CFGS = [(4, 4), (4, 8), (8, 8), (4, 16)]  # (items, block_d); (8,16) over 227KB smem


def _inputs(b, length, d, n, dev):  # type: ignore[no-untyped-def]
    return (
        torch.randn(b, length, d, device=dev),
        torch.randn(b, length, d, device=dev),
        -torch.rand(d, n, device=dev),
        torch.randn(b, length, n, device=dev),
        torch.randn(b, length, n, device=dev),
        torch.randn(d, device=dev),
        torch.randn(b, length, d, device=dev),
    )


def _official(spec, dtype):  # type: ignore[no-untyped-def]
    from lethe.bench.mamba3_backward_headtohead import (
        ShapeSpec,
        _bench_official_combined,
    )

    s = ShapeSpec(spec[0], spec[1], spec[2], spec[3], 64)
    res, _meta = _bench_official_combined(s, dtype, 20)
    return res["median_ms"]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    for b, length, d, n in SHAPES:
        u, delta, a, bmat, cmat, dskip, dy = _inputs(b, length, d, n, dev)
        tag = f"B{b} L{length} D{d} N{n}"

        best_cuda = 1e9
        for it, bd in CUDA_CFGS:
            try:
                r = benchmark(
                    lambda it=it, bd=bd: cuda_backward_scan(
                        u, delta, a, bmat, cmat, dskip, dy, items=it, block_d=bd
                    ),
                    (),
                    warmup=5,
                    trials=20,
                )
                best_cuda = min(best_cuda, r.median_ms)
                print(f"{tag}: cuda_i{it}_bd{bd:<2d} = {r.median_ms:.3f} ms")
            except Exception as e:
                print(f"{tag}: cuda_i{it}_bd{bd} SKIP ({type(e).__name__}: {str(e)[:60]})")

        for mode in ("serial", "chunk_parallel"):
            cfg = KernelConfig(scan_mode=mode)
            r = benchmark(
                lambda cfg=cfg: backward_selective_scan(
                    u, delta, a, bmat, cmat, dskip, dy, chunk_size=64, config=cfg
                ),
                (),
                warmup=5,
                trials=20,
            )
            print(f"{tag}: triton_{mode:<14s} = {r.median_ms:.3f} ms")

        try:
            off_ms = _official((b, length, d, n), torch.float32)
            print(
                f"{tag}: official_crippled = {off_ms:.3f} ms  "
                f"(cuda best {best_cuda:.3f} ms -> {off_ms / best_cuda:.2f}x vs ours)"
            )
        except Exception as e:
            print(f"{tag}: official SKIP ({type(e).__name__}: {str(e)[:80]})")
        print()


if __name__ == "__main__":
    main()
