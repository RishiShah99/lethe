"""E2 autotuner box validation — does a tuned KernelConfig beat the shipped default?

Two modes:
  --speed-only: time the configured op vs the default baseline directly via
    op_bench.measure_speedup (correctness is re-checked at the bench shape, but
    the full contract battery is skipped) — fast, in-process; a headroom probe
    over many configs.
  full (default): score each config through the sandboxed, contract-gated
    score_candidate_config path — the real reward path, slower.

For each op and target shape: measure the default (sanity ~1.0) plus a curated
subset spanning num_warps, block tiling, num_stages (the axis the hand-tuned
heuristics never set), and (backward) chunk_k. Reports best speedup over the
default. Writes results/e2_autotune_validation.json.

Run on the B200 box:  uv run --no-sync python scratch/e2_autotune_validate.py [--ops a,b] [--speed-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lethe.kernels.autotune import (
    KernelConfig,
    ShapeSpec,
    make_configured_op,
    tunable_ops,
)

_WARPS = (2, 4, 8)
_BLOCK = (32, 64, 128)
_CHUNK_K = (8, 16)
_STAGES = (1, 2, 3)

_FWD_SHAPES = [(2, 512, 1024), (2, 2048, 1024), (2, 16384, 1024)]
_BWD_SHAPES = [(2, 2048, 1024), (2, 8192, 1024)]

_HEAD_OPS = frozenset({"complex_scan_rope", "mimo_backward"})
_BWD_OPS = frozenset({"backward_selective_scan", "mimo_backward", "fused_block_backward"})


def _configs(op: str) -> list[KernelConfig]:
    """Curated high-signal subset: default + warps + tiling + num_stages + (bwd) chunk_k."""
    block_field = "block_p" if op in _HEAD_OPS else "block_d"
    cfgs: list[KernelConfig] = [KernelConfig()]  # shipped default (sanity ~1.0)
    cfgs += [KernelConfig(num_warps=w) for w in _WARPS]
    cfgs += [KernelConfig(**{block_field: b}) for b in _BLOCK]
    # num_stages is the one axis the hand-tuned heuristics never set (the kernels
    # launch at Triton's implicit default) — the most likely source of a real win.
    cfgs += [KernelConfig(num_stages=s) for s in _STAGES]
    cfgs.append(KernelConfig(num_warps=8, **{block_field: 128}))
    cfgs.append(KernelConfig(num_warps=4, num_stages=3))
    if op in _BWD_OPS:
        cfgs += [KernelConfig(chunk_k=ck) for ck in _CHUNK_K]
        cfgs += [KernelConfig(chunk_k=ck, num_stages=3) for ck in _CHUNK_K]
    return cfgs


def _measure_speed_only(
    op: str, cfg: KernelConfig, batch: int, seq_len: int, width: int
) -> dict[str, Any]:
    from lethe.verifier import op_bench

    fn = make_configured_op(op, cfg)
    try:
        r = op_bench.measure_speedup(fn, op, "cuda", batch=batch, seq_len=seq_len, width=width)
    except Exception as exc:  # a hard config (regs/pipelining/OOM); record and skip
        return {"status": f"error:{type(exc).__name__}", "contracts": False, "speedup": None}
    ok = bool(r.get("correct_at_bench"))
    return {"status": "speed_only", "contracts": ok, "speedup": r["speedup"] if ok else None}


def _measure_full(
    op: str, cfg: KernelConfig, batch: int, seq_len: int, width: int
) -> dict[str, Any]:
    from lethe.verifier.candidate_scoring import score_candidate_config

    res = score_candidate_config(
        cfg,
        op=op,
        device="cuda",
        shape=ShapeSpec(batch, seq_len, width),
        measure_speedup=True,
        timeout_s=1200,
    )
    return {
        "status": res.get("status"),
        "contracts": res.get("contracts_passed"),
        "speedup": res.get("speedup"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default=",".join(tunable_ops()))
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()
    ops = [o for o in args.ops.split(",") if o]
    measure = _measure_speed_only if args.speed_only else _measure_full

    out: dict[str, Any] = {"mode": "speed_only" if args.speed_only else "full", "runs": []}
    for op in ops:
        shapes = _BWD_SHAPES if op in _BWD_OPS else _FWD_SHAPES
        for batch, seq_len, width in shapes:
            default_speedup: float | None = None
            best = {"speedup": 0.0, "config": None}
            rows = []
            for cfg in _configs(op):
                m = measure(op, cfg, batch, seq_len, width)
                searched = cfg.searched()
                rows.append({"config": searched, **m})
                if not searched:
                    default_speedup = m["speedup"]
                if m["contracts"] and m["speedup"] is not None and m["speedup"] > best["speedup"]:
                    best = {"speedup": m["speedup"], "config": searched}
                print(
                    f"[{op} B{batch}L{seq_len}W{width}] {searched or 'DEFAULT'}: "
                    f"status={m['status']} contracts={m['contracts']} speedup={m['speedup']}",
                    flush=True,
                )
            out["runs"].append(
                {
                    "op": op,
                    "shape": {"batch": batch, "seq_len": seq_len, "width": width},
                    "default_speedup": default_speedup,
                    "best": best,
                    "rows": rows,
                }
            )
            print(
                f"  -> {op} B{batch}L{seq_len}W{width}: default={default_speedup} "
                f"best={best['speedup']:.3f}x via {best['config']}",
                flush=True,
            )

    Path("results").mkdir(exist_ok=True)
    Path("results/e2_autotune_validation.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/e2_autotune_validation.json", flush=True)


if __name__ == "__main__":
    main()
