"""Occupancy sweep: where (if anywhere) chunk-parallel beats serial-L at long L.

The serial kernel launches (batch, d_model/block_d) programs; at small
batch*width it starves the B200's SMs while walking all L. Chunk-parallel adds
an n_chunks factor to the grid, so it should win exactly in that under-saturated
regime. Sweep batch*width at L=16384 to find the crossover; also compare to the
official selective_scan_fn where installed.

    uv run --no-sync python scratch/cp_occupancy.py
"""

from __future__ import annotations

import json

import torch

from lethe.kernels.ops._triton_chunk_parallel_fwd import launch_chunk_parallel_scan
from lethe.kernels.ops._triton_fwd_scan import launch_forward_scan


def _draw(batch: int, seq_len: int, d_model: int, n_state: int, device: str):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(2)
    u = torch.randn(batch, seq_len, d_model, generator=g).to(device)
    delta = (torch.randn(batch, seq_len, d_model, generator=g) - 4.0).to(device)
    a = (-torch.rand(d_model, n_state, generator=g) - 0.5).to(device)
    b = torch.randn(batch, seq_len, n_state, generator=g).to(device)
    c = torch.randn(batch, seq_len, n_state, generator=g).to(device)
    d = torch.randn(d_model, generator=g).to(device)
    return u, delta, a, b, c, d


def _bench(fn, iters: int = 20, warmup: int = 5) -> float:  # type: ignore[no-untyped-def]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    dev = "cuda"
    n_state = 16
    seq_len = 16384
    rows = []
    for batch, width in [(1, 256), (1, 1024), (2, 1024), (1, 4096), (2, 2048), (4, 2048), (8, 4096)]:
        args = _draw(batch, seq_len, width, n_state, dev)
        t_serial = _bench(lambda a=args: launch_forward_scan(*a))
        best_t, best_k = float("inf"), None
        for k in (128, 256, 512, 1024):
            if seq_len % k:
                continue
            t = _bench(lambda a=args, kk=k: launch_chunk_parallel_scan(*a, chunk_len=kk))
            if t < best_t:
                best_t, best_k = t, k
        row = {
            "batch": batch,
            "width": width,
            "serial_programs": batch * ((width + 63) // 64),
            "t_serial_ms": round(t_serial, 3),
            "t_chunk_ms": round(best_t, 3),
            "best_k": best_k,
            "speedup": round(t_serial / best_t, 3),
        }
        rows.append(row)
        print(
            f"b={batch} w={width} progs={row['serial_programs']:>4} "
            f"serial={t_serial:7.3f}ms chunk={best_t:7.3f}ms k={best_k} "
            f"speedup={row['speedup']}x",
            flush=True,
        )

    with open("results/cp_occupancy.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("WROTE results/cp_occupancy.json", flush=True)


if __name__ == "__main__":
    main()
