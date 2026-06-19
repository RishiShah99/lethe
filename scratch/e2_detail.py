"""Per-rollout config+timing dump for an E2.c level (diagnose the learned mode).

Usage: python3 scratch/e2_detail.py [ckpt_dir] [level-name-substring]
Prints, per level: scan_mode counts across ALL steps (did the policy ever sample
serial?), then the last 2 steps' per-rollout config knobs + measured
t_candidate/t_baseline/speedup. Stdlib only.
"""

from __future__ import annotations

import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "e2_config_out"
want = sys.argv[2] if len(sys.argv) > 2 else ""


def cfg_of(row: dict) -> dict:
    try:
        return json.loads(row.get("source", "{}"))
    except Exception:
        return {"PARSE_ERR": True}


for d in sorted(glob.glob(os.path.join(root, "*"))):
    name = os.path.basename(d)
    if want and want not in name:
        continue
    rj = os.path.join(d, "rollouts.jsonl")
    if not os.path.isfile(rj):
        continue
    rows = [json.loads(line) for line in open(rj, encoding="utf-8") if line.strip()]
    modes: dict[str, int] = {}
    for r in rows:
        m = str(cfg_of(r).get("scan_mode", "default"))
        modes[m] = modes.get(m, 0) + 1
    print(f"== {name} ({len(rows)} rollouts) mode_counts_all_steps={modes} ==")
    steps = sorted({r.get("step") for r in rows})
    for s in steps[-2:]:
        print(f"  -- step {s} --")
        for r in rows:
            if r.get("step") != s:
                continue
            c = cfg_of(r)
            b = r.get("bench") or {}
            print(
                f"    mode={c.get('scan_mode', 'default')} clen={c.get('chunk_len')} "
                f"nw={c.get('num_warps')} bd={c.get('block_d')} ck={c.get('chunk_k')} "
                f"su={r.get('speedup')} tc={b.get('t_candidate_ms')} tb={b.get('t_baseline_ms')} "
                f"pass={r.get('contracts_passed')}"
            )
