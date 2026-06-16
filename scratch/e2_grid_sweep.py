"""E2.d — exhaustive autotuning grid sweep + cost-model training data.

For each (op, shape) over a diverse shape grid, times the shipped default once
and then every config in the op's search grid (or a random sample), recording
absolute candidate latency + speedup-vs-default. Output is one JSONL row per
(op, shape, config) measurement, appended immediately so a spot preemption
never loses collected data; a restart skips tuples already present (resumable).

This produces both the E2.d "autotuning ceiling" table (grid-best vs shipped
default per shape) and the E2.e cost-model training set (shape, config ->
latency). Timing is speed-only (correctness is the gated path's job and the
grid is correctness-invariant launch knobs); a per-shape correctness spot-check
against the default guards against a pathological config.

Run on the B200 box:
  uv run --no-sync python scratch/e2_grid_sweep.py --ops fused_block_forward [--max-configs N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flash_mamba_rl.kernels.autotune import (
    KernelConfig,
    ShapeSpec,
    iter_configs,
    make_configured_op,
    validate,
)

_WIDTHS = (256, 1024, 2048, 4096)
_FWD_SEQ = (1024, 4096, 16384)
_BWD_SEQ = (1024, 4096, 8192)
_BATCH = 2

_BWD_OPS = frozenset({"backward_selective_scan", "mimo_backward", "fused_block_backward"})

_OUT = Path("results/e2_grid_sweep.jsonl")


def _shapes(op: str) -> list[tuple[int, int, int]]:
    seqs = _BWD_SEQ if op in _BWD_OPS else _FWD_SEQ
    return [(_BATCH, s, w) for w in _WIDTHS for s in seqs]


def _time_ms(fn: Any, op: str, batch: int, seq_len: int, width: int) -> float:
    from flash_mamba_rl.verifier import op_bench
    from flash_mamba_rl.verifier.timing import benchmark

    template = op_bench.build_bench_case(op, "cuda", batch=batch, seq_len=seq_len, width=width)
    kwargs = template.kwargs

    def factory(i: int) -> tuple[Any, ...]:
        return op_bench.build_bench_case(
            op, "cuda", batch=batch, seq_len=seq_len, width=width, seed=9_000 + i
        ).args

    call = (lambda *a: fn(*a, **kwargs)) if kwargs else fn
    return benchmark(call, warmup=3, trials=15, inputs_factory=factory).median_ms


def _done_keys() -> set[str]:
    if not _OUT.exists():
        return set()
    keys = set()
    for line in _OUT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys.add(
            f"{r['op']}|{r['batch']}x{r['seq_len']}x{r['width']}|{json.dumps(r['config'], sort_keys=True)}"
        )
    return keys


def _append(row: dict[str, Any]) -> None:
    _OUT.parent.mkdir(exist_ok=True)
    with _OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", default="fused_block_forward,mimo_backward,fused_block_backward")
    parser.add_argument("--max-configs", type=int, default=0, help="0 = full grid; else first N")
    args = parser.parse_args()
    ops = [o for o in args.ops.split(",") if o]
    done = _done_keys()
    print(f"resuming: {len(done)} measurements already on disk", flush=True)

    for op in ops:
        for batch, seq_len, width in _shapes(op):
            shape = ShapeSpec(batch, seq_len, width)
            configs = [c for c in iter_configs(op) if not validate(op, c, shape=shape)]
            if args.max_configs:
                configs = configs[: args.max_configs]
            # Default timed once per shape (the speedup denominator).
            try:
                t_default = _time_ms(
                    make_configured_op(op, KernelConfig()), op, batch, seq_len, width
                )
            except Exception as exc:  # shape itself infeasible (OOM) — skip the whole shape
                print(
                    f"[{op} B{batch}L{seq_len}W{width}] default FAILED {type(exc).__name__}; skip",
                    flush=True,
                )
                continue
            for cfg in configs:
                key = f"{op}|{batch}x{seq_len}x{width}|{json.dumps(cfg.searched(), sort_keys=True)}"
                if key in done:
                    continue
                try:
                    t = _time_ms(make_configured_op(op, cfg), op, batch, seq_len, width)
                    speedup = t_default / t
                    status = "ok"
                except Exception as exc:  # a hard config (regs/OOM) — record and move on
                    t, speedup, status = float("nan"), 0.0, f"error:{type(exc).__name__}"
                _append(
                    {
                        "op": op,
                        "batch": batch,
                        "seq_len": seq_len,
                        "width": width,
                        "config": cfg.searched(),
                        "t_ms": t,
                        "t_default_ms": t_default,
                        "speedup": speedup,
                        "status": status,
                    }
                )
            best = {"speedup": 1.0, "config": {}}
            for line in _OUT.read_text().splitlines():
                r = json.loads(line)
                if (
                    r["op"] == op
                    and r["seq_len"] == seq_len
                    and r["width"] == width
                    and r["batch"] == batch
                    and r["status"] == "ok"
                    and r["speedup"] > best["speedup"]
                ):
                    best = {"speedup": r["speedup"], "config": r["config"]}
            print(
                f"  -> {op} B{batch}L{seq_len}W{width}: grid-best {best['speedup']:.3f}x via {best['config']}",
                flush=True,
            )
    print("SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
