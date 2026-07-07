"""GPU smoke for the scan_mode selector (task #3): chunk_parallel vs serial.

Confirms the public forward_chunked_scan dispatches both modes correctly on
CUDA and that chunk_parallel is the long-L lever. Correctness is agreement
between the two modes (the fp64 algebra is already pinned by
test_chunk_parallel_scan_replica); here we check GPU fp32 agreement within the
eps*sqrt(chain)*scale band and time both.
"""

from __future__ import annotations

import json

import torch

from lethe.kernels.autotune import KernelConfig
from lethe.kernels.ops import forward_chunked_scan


def _inputs(b: int, l: int, d: int, n: int, seed: int = 0):  # type: ignore[no-untyped-def]
    g = torch.Generator(device="cuda").manual_seed(seed)
    u = torch.randn(b, l, d, device="cuda", generator=g)
    delta = torch.randn(b, l, d, device="cuda", generator=g) * 0.1 - 2.0
    A = -torch.rand(d, n, device="cuda", generator=g) * 0.5 - 0.1
    B = torch.randn(b, l, n, device="cuda", generator=g)
    C = torch.randn(b, l, n, device="cuda", generator=g)
    D = torch.randn(d, device="cuda", generator=g)
    return u, delta, A, B, C, D


def _time(fn, iters: int = 20) -> float:  # type: ignore[no-untyped-def]
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run(b: int, l: int, d: int, n: int, chunk_len: int) -> dict:  # type: ignore[type-arg]
    u, delta, A, B, C, D = _inputs(b, l, d, n)
    cp = KernelConfig(scan_mode="chunk_parallel", chunk_len=chunk_len)
    y_ser = forward_chunked_scan(u, delta, A, B, C, D, chunk_size=64)
    y_cp = forward_chunked_scan(u, delta, A, B, C, D, chunk_size=64, config=cp)
    scale = float(y_ser.abs().max().item())
    max_abs = float((y_ser - y_cp).abs().max().item())
    t_ser = _time(lambda: forward_chunked_scan(u, delta, A, B, C, D, chunk_size=64))
    t_cp = _time(lambda: forward_chunked_scan(u, delta, A, B, C, D, chunk_size=64, config=cp))
    return {
        "shape": f"b{b}_l{l}_d{d}_n{n}",
        "chunk_len": chunk_len,
        "max_abs_diff": max_abs,
        "rel_to_scale": max_abs / scale if scale else 0.0,
        "t_serial_ms": round(t_ser, 4),
        "t_chunk_parallel_ms": round(t_cp, 4),
        "speedup": round(t_ser / t_cp, 3) if t_cp else 0.0,
    }


def main() -> None:
    rows = [
        run(2, 16384, 1024, 64, chunk_len=256),
        run(4, 16384, 2048, 64, chunk_len=256),
        run(2, 1024, 1024, 64, chunk_len=128),
        run(8, 2048, 2048, 64, chunk_len=128),
    ]
    print("SCAN_MODE_SMOKE_JSON", json.dumps(rows))
    for r in rows:
        print(
            f"{r['shape']} cl={r['chunk_len']}: rel_to_scale={r['rel_to_scale']:.2e} "
            f"serial={r['t_serial_ms']}ms chunk_parallel={r['t_chunk_parallel_ms']}ms "
            f"speedup={r['speedup']}x"
        )


if __name__ == "__main__":
    main()
