"""Re-tune the backward scan at Mamba-3's native d_state=128.

The head-to-head ran the N=16-tuned default (block_d=64, num_warps=4). At
N=128 the [block_d, block_n=128] fp32 register tiles are 8x larger -> suspected
spill cliff (backward went 6.6ms@N16 -> ~98ms@N128, ~15x for 8x work). This
sweeps block_d x num_warps x num_stages x scan_mode at the N=128 training and
long-L shapes, records median wall-clock + the compiled kernels' ptxas
resources (regs/spill), and reports the best config vs the default and vs the
crippled-official reference (from results/mamba3_headtohead.json).

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/bwd_n128_sweep.py --out ~/out/bwd_n128_sweep.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_bwd, backward_selective_scan
from flash_mamba_rl.verifier.timing import benchmark

# (batch, seq_len, d_model, n_state) — Mamba-3 native d_state=128.
SHAPES = [
    (8, 2048, 4096, 128),  # training shape — the #904 regime; official ~12.5ms fp32
    (2, 16384, 4096, 128),  # long-L; official ~27.7ms fp32
]
# Official crippled SISO backward medians (fp32) from results/mamba3_headtohead.json.
OFFICIAL_MS = {(8, 2048, 4096, 128): 12.516, (2, 16384, 4096, 128): 27.746}

BLOCK_DS = [16, 32, 64, 128]
NUM_WARPS = [2, 4, 8]
NUM_STAGES = [1, 2]
MODES = ["serial", "chunk_parallel"]


def _inputs(shape: tuple[int, int, int, int], dtype: torch.dtype, seed: int = 0):  # type: ignore[no-untyped-def]
    b, length, d, n = shape
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    u = torch.randn(b, length, d, device=dev).to(dtype)
    delta = torch.randn(b, length, d, device=dev).to(dtype)
    a = (-torch.rand(d, n, device=dev)).to(dtype)
    b_proj = torch.randn(b, length, n, device=dev).to(dtype)
    c_proj = torch.randn(b, length, n, device=dev).to(dtype)
    d_skip = torch.randn(d, device=dev).to(dtype)
    dy = torch.randn(b, length, d, device=dev).to(dtype)
    return u, delta, a, b_proj, c_proj, d_skip, dy


def _resource(mode: str) -> dict[str, int] | None:
    if mode == "chunk_parallel":
        return _triton_chunk_parallel_bwd.resource_meta()
    from flash_mamba_rl.kernels.ops.backward_selective_scan import triton_bwd_scan_resource_meta

    return triton_bwd_scan_resource_meta()


def _time_cfg(args: tuple[Any, ...], cfg: KernelConfig, trials: int) -> dict[str, Any]:
    u, delta, a, b_proj, c_proj, d_skip, dy = args

    def run() -> Any:
        return backward_selective_scan(
            u, delta, a, b_proj, c_proj, d_skip, dy, chunk_size=64, config=cfg
        )

    run()  # warm/compile
    r = benchmark(run, (), warmup=8, trials=trials)
    return {"median_ms": r.median_ms, "min_ms": r.min_ms, "n_trials": float(r.n_trials)}


def _sweep_shape(
    shape: tuple[int, int, int, int], dtype: torch.dtype, quick: bool
) -> dict[str, Any]:
    args = _inputs(shape, dtype)
    trials = 10 if quick else 30
    rows: list[dict[str, Any]] = []

    # Default (no config = N=16-tuned heuristic) for the speedup baseline.
    default = _time_cfg(args, KernelConfig(), trials)

    for mode in MODES:
        for bd in BLOCK_DS:
            for nw in NUM_WARPS:
                for ns in NUM_STAGES:
                    cfg = KernelConfig(scan_mode=mode, block_d=bd, num_warps=nw, num_stages=ns)
                    entry: dict[str, Any] = {
                        "mode": mode, "block_d": bd, "num_warps": nw, "num_stages": ns,
                    }  # fmt: skip
                    try:
                        entry.update(_time_cfg(args, cfg, trials))
                        entry["resource"] = _resource(mode)
                    except Exception:
                        entry["error"] = traceback.format_exc(limit=2).splitlines()[-1]
                    rows.append(entry)
                    print(
                        f"  {mode} bd={bd} nw={nw} ns={ns}: "
                        f"{entry.get('median_ms', entry.get('error'))}",
                        flush=True,
                    )

    ok = [r for r in rows if "median_ms" in r]
    best = min(ok, key=lambda r: r["median_ms"]) if ok else None
    off = OFFICIAL_MS.get(shape)
    return {
        "shape": {"batch": shape[0], "seq_len": shape[1], "d_model": shape[2], "n_state": shape[3]},
        "dtype": str(dtype).removeprefix("torch."),
        "default_ms": default["median_ms"],
        "best": best,
        "best_speedup_vs_default": (default["median_ms"] / best["median_ms"]) if best else None,
        "official_crippled_ms": off,
        "best_vs_official": (off / best["median_ms"]) if (best and off) else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path.home() / "out" / "bwd_n128_sweep.json")
    ap.add_argument("--quick", action="store_true")
    cli = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    report: dict[str, Any] = {
        "purpose": "re-tune backward at d_state=128 to close the head-to-head gap",
        "gpu": torch.cuda.get_device_name(0),
        "runs": [],
    }
    shapes = SHAPES[:1] if cli.quick else SHAPES
    for shape in shapes:
        print(f"[sweep] {shape} fp32 ...", flush=True)
        report["runs"].append(_sweep_shape(shape, torch.float32, cli.quick))
        torch.cuda.empty_cache()
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[sweep] wrote {cli.out}")
    for run in report["runs"]:
        print(
            f"  {run['shape']}: default {run['default_ms']:.1f}ms -> best "
            f"{run['best']['median_ms']:.1f}ms ({run['best_speedup_vs_default']:.2f}x), "
            f"official {run['official_crippled_ms']}ms, best/official={run['best_vs_official']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
