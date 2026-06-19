"""Per-level verdict for the E2.c scan_mode run: what mode the policy learned.

Reads e2_config_out/*/rollouts.jsonl and reports, per (op, shape) level, the
scan_mode the policy emits in its late steps and the rewards/speedups achieved.
The win indicator: chunk_parallel dominates the long-L levels, serial/default
dominates the saturated levels (the learned crossover). Stdlib only.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "e2_config_out"


def mode_of(row: dict) -> str:
    try:
        cfg = json.loads(row.get("source", "{}"))
    except Exception:
        return "parse_err"
    return str(cfg.get("scan_mode", "default"))


for d in sorted(glob.glob(os.path.join(root, "*"))):
    rj = os.path.join(d, "rollouts.jsonl")
    if not os.path.isfile(rj):
        continue
    rows = [json.loads(line) for line in open(rj, encoding="utf-8") if line.strip()]
    if not rows:
        continue
    by_step: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_step[r.get("step", 0)].append(r)
    steps_sorted = sorted(by_step)
    last_step = steps_sorted[-1]
    late = [r for s in steps_sorted[-3:] for r in by_step[s]]
    cnt = collections.Counter(mode_of(r) for r in late)
    late_r = [r.get("reward", 0.0) for r in late]
    speeds = [r.get("speedup") for r in rows if isinstance(r.get("speedup"), (int, float))]
    cp = [r.get("speedup") for r in rows if mode_of(r) == "chunk_parallel" and isinstance(r.get("speedup"), (int, float))]
    se = [r.get("speedup") for r in rows if mode_of(r) in ("serial", "default") and isinstance(r.get("speedup"), (int, float))]
    name = os.path.basename(d)
    mean_late_r = sum(late_r) / len(late_r) if late_r else 0.0
    best = max(speeds) if speeds else 1.0
    cp_mean = sum(cp) / len(cp) if cp else float("nan")
    se_mean = sum(se) / len(se) if se else float("nan")
    print(
        f"{name}: last_step={last_step} late_modes={dict(cnt)} "
        f"late_mean_r={mean_late_r:.3f} best_speedup={best:.2f} "
        f"cp_mean_su={cp_mean:.2f} serial_mean_su={se_mean:.2f}"
    )
