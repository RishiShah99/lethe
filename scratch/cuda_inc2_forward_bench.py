"""Inc 2 — parity + speed of the 2-D forward (warps split d_state).

Parity: the 2-D kernel vs reference_forward_chunked_scan across shapes and warp
counts. Speed: 2-D (4/8/16/32 warps) vs the Inc 1 serial-d_state kernel and,
best-effort, the official Mamba-1 ``selective_scan_fn`` forward — the N=128
regime where serial d_state is supposed to erode.

    CUDA_HOME=/usr/local/cuda-13.0 PATH=$CUDA_HOME/bin:$PATH \
        CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/cuda_inc2_forward_bench.py
"""

from __future__ import annotations

import torch

from flash_mamba_rl.kernels.cuda.forward import (
    cuda_forward_scan,
    cuda_forward_scan_2d,
    cuda_forward_scan_tiled,
)
from flash_mamba_rl.kernels.references.forward_chunked_scan import reference_forward_chunked_scan
from flash_mamba_rl.verifier.timing import benchmark

PARITY_SHAPES = [
    (2, 512, 256, 16),
    (2, 512, 1024, 128),
    (1, 2048, 512, 128),
    (4, 1024, 2048, 64),
    (2, 128, 64, 128),
    (3, 320, 384, 96),
]
WARPS = [4, 8, 16, 32]
BOUND = 2e-4

# N=128 Mamba-3 regime — where serial d_state erodes.
BENCH_SHAPES = [(8, 2048, 4096, 128), (2, 16384, 4096, 128)]


def _inputs(b, length, d, n, dev):  # type: ignore[no-untyped-def]
    return (
        torch.randn(b, length, d, device=dev),
        torch.randn(b, length, d, device=dev),
        -torch.rand(d, n, device=dev),
        torch.randn(b, length, n, device=dev),
        torch.randn(b, length, n, device=dev),
        torch.randn(d, device=dev),
    )


def _official_fwd(u, delta, a, bmat, cmat, dskip):  # type: ignore[no-untyped-def]
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    uT = u.transpose(1, 2).contiguous()
    dT = delta.transpose(1, 2).contiguous()
    bT = bmat.transpose(1, 2).contiguous()
    cT = cmat.transpose(1, 2).contiguous()
    return lambda: selective_scan_fn(uT, dT, a, bT, cT, dskip, delta_softplus=True)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    dev = torch.device("cuda")
    torch.manual_seed(0)

    print("=== parity (2-D vs reference) ===")
    all_ok = True
    worst = 0.0
    for b, length, d, n in PARITY_SHAPES:
        u, delta, a, bmat, cmat, dskip = _inputs(b, length, d, n, dev)
        y_ref = reference_forward_chunked_scan(u, delta, a, bmat, cmat, dskip, chunk_size=64)
        scale = y_ref.abs().max().item()
        cands = [
            (f"2d_w{w}", cuda_forward_scan_2d(u, delta, a, bmat, cmat, dskip, warps=w))
            for w in WARPS
        ]
        cands += [
            (f"tiled_i{i}", cuda_forward_scan_tiled(u, delta, a, bmat, cmat, dskip, items=i))
            for i in (4, 8, 16)
        ]
        for name, y in cands:
            rel = (y - y_ref).abs().max().item() / scale
            worst = max(worst, rel)
            ok = rel < BOUND
            all_ok &= ok
            if not ok:
                print(f"  FAIL B{b} L{length} D{d} N{n} {name}: rel={rel:.3e}")
    print(f"parity worst scale-rel {worst:.3e} (bound {BOUND:.1e}) -> {'OK' if all_ok else 'FAIL'}")

    print("\n=== speed (median ms) ===")
    for b, length, d, n in BENCH_SHAPES:
        u, delta, a, bmat, cmat, dskip = _inputs(b, length, d, n, dev)
        tag = f"B{b} L{length} D{d} N{n}"
        inc1 = benchmark(
            lambda: cuda_forward_scan(u, delta, a, bmat, cmat, dskip), (), warmup=5, trials=20
        )
        print(f"{tag}: inc1_serial_n = {inc1.median_ms:.3f} ms")
        for w in WARPS:
            r = benchmark(
                lambda w=w: cuda_forward_scan_2d(u, delta, a, bmat, cmat, dskip, warps=w),
                (),
                warmup=5,
                trials=20,
            )
            print(
                f"{tag}: inc2_2d_w{w:<2d}  = {r.median_ms:.3f} ms  ({inc1.median_ms / r.median_ms:.2f}x vs inc1)"
            )
        for it in (4, 8, 16):
            r = benchmark(
                lambda it=it: cuda_forward_scan_tiled(u, delta, a, bmat, cmat, dskip, items=it),
                (),
                warmup=5,
                trials=20,
            )
            print(
                f"{tag}: tiled_i{it:<2d}    = {r.median_ms:.3f} ms  ({inc1.median_ms / r.median_ms:.2f}x vs inc1)"
            )
        try:
            off = _official_fwd(u, delta, a, bmat, cmat, dskip)
            off()
            r = benchmark(off, (), warmup=5, trials=20)
            print(f"{tag}: official_ssf  = {r.median_ms:.3f} ms")
        except Exception as e:
            print(f"{tag}: official_ssf SKIPPED ({type(e).__name__}: {str(e)[:80]})")

    print("\nINC2-FORWARD-OK" if all_ok else "\nINC2-FORWARD-FAIL")


if __name__ == "__main__":
    main()
