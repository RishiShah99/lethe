"""Chunk-parallel vs official CUDA selective_scan_fn at long L (the claim test).

Our serial kernel was ~3.5x behind official at L=16K. The chunk-parallel kernel
is 3-10x over serial in the under-saturated (small-batch) regime; this measures
whether that closes the gap to — or beats — the official Mamba-1 CUDA kernel
there. Same operand bits and the official [B,D,L]/[B,1,N,L] layout +
delta_softplus=True as bench/c1_forward_chunked_scan.py, so numbers compare
implementations, not rounding. Needs mamba_ssm built on the box.

    uv run --no-sync python scratch/cp_vs_official.py
"""

from __future__ import annotations

import json

import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from flash_mamba_rl.kernels.ops._triton_chunk_parallel_fwd import launch_chunk_parallel_scan
from flash_mamba_rl.kernels.ops._triton_fwd_scan import launch_forward_scan


def _draw(batch: int, seq_len: int, d_model: int, n_state: int, device: str):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(2)
    u = torch.randn(batch, seq_len, d_model, generator=g).to(device)
    delta = (torch.randn(batch, seq_len, d_model, generator=g) - 4.0).to(device)
    a = (-torch.rand(d_model, n_state, generator=g) - 0.5).to(device)
    b = torch.randn(batch, seq_len, n_state, generator=g).to(device)
    c = torch.randn(batch, seq_len, n_state, generator=g).to(device)
    d = torch.randn(d_model, generator=g).to(device)
    return u, delta, a, b, c, d


def _official_args(u, delta, a, b, c, d):  # type: ignore[no-untyped-def]
    return (
        u.transpose(1, 2).contiguous(),
        delta.transpose(1, 2).contiguous(),
        a.float(),
        b.transpose(1, 2).unsqueeze(1).contiguous(),
        c.transpose(1, 2).unsqueeze(1).contiguous(),
        d.float(),
    )


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
    seq_len, n_state = 16384, 16
    rows = []
    for batch, width in [(1, 256), (1, 1024), (1, 4096), (2, 2048), (4, 2048), (8, 4096)]:
        args = _draw(batch, seq_len, width, n_state, dev)
        off = _official_args(*args)

        y_cp = launch_chunk_parallel_scan(*args, chunk_len=512)
        y_off = selective_scan_fn(*off, delta_softplus=True).transpose(1, 2)
        scale = y_off.abs().max().clamp_min(1.0).item()
        err = (y_cp - y_off).abs().max().item() / scale

        t_serial = _bench(lambda a=args: launch_forward_scan(*a))
        best_t = min(
            _bench(lambda a=args, kk=k: launch_chunk_parallel_scan(*a, chunk_len=kk))
            for k in (256, 512, 1024)
        )
        t_off = _bench(lambda o=off: selective_scan_fn(*o, delta_softplus=True))

        row = {
            "batch": batch,
            "width": width,
            "err_vs_official": err,
            "t_serial_ms": round(t_serial, 3),
            "t_chunk_ms": round(best_t, 3),
            "t_official_ms": round(t_off, 3),
            "chunk_vs_official": round(t_off / best_t, 3),
            "serial_vs_official": round(t_off / t_serial, 3),
        }
        rows.append(row)
        print(
            f"b={batch} w={width} err={err:.1e} | serial={t_serial:7.3f} "
            f"chunk={best_t:7.3f} official={t_off:7.3f}ms | "
            f"chunk/official={row['chunk_vs_official']}x (serial was {row['serial_vs_official']}x)",
            flush=True,
        )

    with open("results/cp_vs_official.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("WROTE results/cp_vs_official.json", flush=True)


if __name__ == "__main__":
    main()
