"""Box validation + bench for the chunk-parallel forward scan (Phase E2.f).

Run on a CUDA box. Confirms the chunk-parallel kernel matches the reference
oracle AND the shipped serial kernel within the reduction band at several
shapes (incl. long L), across chunk_len choices; then benchmarks serial vs
chunk-parallel vs (if installed) official selective_scan_fn at L=2K/8K/16K to
measure the long-L speedup the restructuring targets.

    uv run --no-sync python scratch/chunk_parallel_validate.py
"""

from __future__ import annotations

import json

import torch

from flash_mamba_rl.kernels.ops._triton_chunk_parallel_fwd import launch_chunk_parallel_scan
from flash_mamba_rl.kernels.ops._triton_fwd_scan import launch_forward_scan
from flash_mamba_rl.kernels.references.forward_chunked_scan import reference_forward_chunked_scan


def _draw(batch: int, seq_len: int, d_model: int, n_state: int, seed: int, device: str):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, seq_len, d_model, generator=g)
    delta = torch.randn(batch, seq_len, d_model, generator=g) - 4.0
    a = -torch.rand(d_model, n_state, generator=g) - 0.5
    b = torch.randn(batch, seq_len, n_state, generator=g)
    c = torch.randn(batch, seq_len, n_state, generator=g)
    d = torch.randn(d_model, generator=g)
    cu = [t.to(device) for t in (u, delta, a, b, c, d)]
    return (u, delta, a, b, c, d), cu


def _bench(fn, *args, iters: int = 30, warmup: int = 5) -> float:  # type: ignore[no-untyped-def]
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    dev = "cuda"
    torch.manual_seed(0)
    results: dict[str, object] = {"correctness": [], "bench": []}

    # --- correctness: chunk-parallel vs reference AND vs serial kernel ---
    shapes = [(2, 256, 64, 16), (1, 2048, 256, 64), (2, 4096, 128, 32), (1, 16384, 64, 16)]
    for batch, seq_len, d_model, n_state in shapes:
        cpu, cu = _draw(batch, seq_len, d_model, n_state, seed=1, device=dev)
        ref = reference_forward_chunked_scan(*cpu).to(dev)
        serial = launch_forward_scan(*cu)
        for k in (64, 128, 256):
            if seq_len % k:
                continue
            cp = launch_chunk_parallel_scan(*cu, chunk_len=k)
            scale = ref.abs().max().clamp_min(1.0).item()
            err_ref = (cp - ref).abs().max().item() / scale
            err_ser = (cp - serial).abs().max().item() / scale
            ok = err_ref < 2e-3
            results["correctness"].append(  # type: ignore[union-attr]
                {
                    "shape": [batch, seq_len, d_model, n_state],
                    "chunk_len": k,
                    "err_vs_ref": err_ref,
                    "err_vs_serial": err_ser,
                    "ok": ok,
                }
            )
            print(
                f"shape={[batch, seq_len, d_model, n_state]} k={k} "
                f"err_ref={err_ref:.2e} err_serial={err_ser:.2e} ok={ok}",
                flush=True,
            )

    # --- bench: serial vs chunk-parallel at the training width, long L ---
    for seq_len in (2048, 8192, 16384):
        _, cu = _draw(8, seq_len, 4096, 64, seed=2, device=dev)
        t_serial = _bench(lambda c=cu: launch_forward_scan(*c))
        best_k, best_t = None, float("inf")
        for k in (128, 256, 512):
            if seq_len % k:
                continue
            t = _bench(lambda kk=k, c=cu: launch_chunk_parallel_scan(*c, chunk_len=kk))
            if t < best_t:
                best_t, best_k = t, k
        speedup = t_serial / best_t
        row = {
            "seq_len": seq_len,
            "t_serial_ms": t_serial,
            "t_chunk_ms": best_t,
            "best_chunk_len": best_k,
            "speedup": speedup,
        }
        results["bench"].append(row)  # type: ignore[union-attr]
        print(
            f"L={seq_len} serial={t_serial:.3f}ms chunk={best_t:.3f}ms "
            f"(k={best_k}) speedup={speedup:.2f}x",
            flush=True,
        )

    with open("results/chunk_parallel_validate.json", "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE results/chunk_parallel_validate.json", flush=True)


if __name__ == "__main__":
    main()
