"""Phase E status: curriculum position + per-level metrics + recent gate fails."""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise_level(level_dir: str, tail: int) -> None:
    metrics = read_jsonl(os.path.join(level_dir, "metrics.jsonl"))
    rollouts = read_jsonl(os.path.join(level_dir, "rollouts.jsonl"))
    if not metrics:
        print("  (no metrics yet)")
        return
    last = metrics[-1]
    print(
        f"  steps={last['step']} mean_r={last['mean_reward']:.3f} "
        f"max_r={last['max_reward']:.3f} contracts={last['n_contracts_passed']} "
        f"kl={last['mean_kl']}"
    )
    status = collections.Counter(r["status"] for r in rollouts)
    print(f"  rollouts={len(rollouts)} status={dict(status)}")
    speedups = [r["speedup"] for r in rollouts if r.get("speedup")]
    if speedups:
        print(f"  speedups: n={len(speedups)} max={max(speedups):.2f}")
    views = collections.Counter(
        r.get("first_failed_view") for r in rollouts if r.get("first_failed_view")
    )
    if views:
        print(f"  first-failed views: {dict(views.most_common(4))}")
    gate_fails: collections.Counter[str] = collections.Counter()
    for r in rollouts[-tail:]:
        for g, v in r.get("gates", {}).items():
            if not v["passed"]:
                gate_fails[g.split("/")[-1]] += 1
    if gate_fails:
        print(f"  recent gate fails: {dict(gate_fails.most_common(5))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="phase_e_out")
    ap.add_argument("--tail", type=int, default=32)
    args = ap.parse_args()

    state_path = os.path.join(args.ckpt_dir, "curriculum_state.json")
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        print(f"curriculum: level_idx={state['level_idx']}")
        for i, lv in enumerate(state["levels"]):
            marker = ">>" if i == state["level_idx"] else "  "
            print(
                f"{marker} L{i} {lv['op']}: steps={lv['steps']} "
                f"promoted={lv['promoted']} closed={lv['closed']} "
                f"best_mean={lv['best_mean_reward']:.3f} best_max={lv['best_max_reward']:.3f}"
            )
    dirs = (
        sorted(
            d
            for d in os.listdir(args.ckpt_dir)
            if os.path.isdir(os.path.join(args.ckpt_dir, d))
            and (d.startswith("level") or d.startswith("direct"))
        )
        if os.path.isdir(args.ckpt_dir)
        else []
    )
    for d in dirs:
        print(f"[{d}]")
        summarise_level(os.path.join(args.ckpt_dir, d), args.tail)


if __name__ == "__main__":
    main()
