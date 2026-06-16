"""E2.e — shape-adaptive autotuner: predict the best config for an unseen shape.

Reads the E2.d grid-sweep table (results/e2_grid_sweep.jsonl) and evaluates, per
op, whether a model trained on profiled (shape, config) -> speedup data picks
near-optimal configs for shapes it never saw. Leave-one-shape-out cross-val of
two predictors against two anchors:

  oracle   - the per-shape grid-best speedup (the autotuning ceiling).
  default  - the shipped kernel (speedup 1.0 by definition).
  static   - the config with the best MEAN speedup over the other shapes
             (a static per-op re-tune; no shape adaptivity).
  knn      - among the k nearest training shapes (normalised log2 seq_len /
             log2 width / batch), the config with the best mean speedup there
             (a nonparametric shape-adaptive cost model).

Reports, per op, the mean achieved speedup of static / knn vs the oracle ceiling
and the default floor — i.e. how much of the autotuning headroom a deployment
specialiser captures without per-shape search. Pure stdlib; no new deps.

  uv run python scratch/e2_cost_model.py   (after fetching e2_grid_sweep.jsonl)
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_IN = Path("results/e2_grid_sweep.jsonl")
_OUT = Path("results/e2_cost_model.json")
_K = 3


def _shape_key(r: dict[str, Any]) -> tuple[int, int, int]:
    return (r["batch"], r["seq_len"], r["width"])


def _cfg_key(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True)


def _load() -> dict[str, dict[tuple[int, int, int], dict[str, float]]]:
    """op -> shape -> config_key -> speedup (ok rows only)."""
    table: dict[str, dict[tuple[int, int, int], dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for line in _IN.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") == "ok":
            table[r["op"]][_shape_key(r)][_cfg_key(r["config"])] = float(r["speedup"])
    return table


def _features(shape: tuple[int, int, int]) -> tuple[float, float, float]:
    batch, seq_len, width = shape
    return (float(batch), math.log2(seq_len), math.log2(width))


def _dist(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    fa, fb = _features(a), _features(b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(fa, fb, strict=True)))


def _best_mean_config(shapes: list[tuple[int, int, int]], data: dict, candidates: set[str]) -> str:
    """The config with the best mean speedup across *shapes* (over shared candidates)."""
    best_cfg, best_mean = "{}", -1.0
    for cfg in candidates:
        vals = [data[s][cfg] for s in shapes if cfg in data[s]]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        if m > best_mean:
            best_cfg, best_mean = cfg, m
    return best_cfg


def _evaluate(
    table: dict[str, dict[tuple[int, int, int], dict[str, float]]],
) -> list[dict[str, Any]]:
    """Per-op leave-one-shape-out row: {op, shapes, default, oracle, static, knn}."""
    rows: list[dict[str, Any]] = []
    for op, by_shape in sorted(table.items()):
        shapes = list(by_shape)
        if len(shapes) < 3:
            rows.append(
                {
                    "op": op,
                    "shapes": len(shapes),
                    "default": None,
                    "oracle": None,
                    "static": None,
                    "knn": None,
                }
            )
            continue
        all_cfgs = {c for s in shapes for c in by_shape[s]}
        oracle_sum = static_sum = knn_sum = 0.0
        for held in shapes:
            train = [s for s in shapes if s != held]
            held_data = by_shape[held]
            oracle_sum += max(held_data.values())
            static_cfg = _best_mean_config(train, by_shape, all_cfgs)
            static_sum += held_data.get(static_cfg, 1.0)
            nearest = sorted(train, key=lambda s: _dist(s, held))[:_K]
            knn_cfg = _best_mean_config(nearest, by_shape, all_cfgs)
            knn_sum += held_data.get(knn_cfg, 1.0)
        n = len(shapes)
        rows.append(
            {
                "op": op,
                "shapes": n,
                "default": 1.0,
                "oracle": oracle_sum / n,
                "static": static_sum / n,
                "knn": knn_sum / n,
            }
        )
    return rows


def _write_json(rows: list[dict[str, Any]]) -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_OUT.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(rows, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, _OUT)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> None:
    table = _load()
    if not table:
        print(f"no data in {_IN} yet", flush=True)
        return
    rows = _evaluate(table)
    print(
        f"{'op':28} {'shapes':>6} {'default':>8} {'oracle':>8} {'static':>8} {'knn':>8}", flush=True
    )
    for r in rows:
        if r["oracle"] is None:
            print(f"{r['op']:28} {r['shapes']:>6}  (too few shapes for LOSO)", flush=True)
            continue
        print(
            f"{r['op']:28} {r['shapes']:>6} {r['default']:>8.3f} {r['oracle']:>8.3f} "
            f"{r['static']:>8.3f} {r['knn']:>8.3f}",
            flush=True,
        )
    _write_json(rows)
    print(f"\nwrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
