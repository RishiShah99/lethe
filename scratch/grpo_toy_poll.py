"""Summarise the toy run's rollouts: status counts + top failing gates per step."""

from __future__ import annotations

import collections
import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "grpo_toy_out/rollouts.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    print(f"rows={len(rows)}")
    status = collections.Counter(r["status"] for r in rows)
    print("status:", dict(status))
    gate_fails: collections.Counter[str] = collections.Counter()
    for r in rows:
        for g, v in r.get("gates", {}).items():
            if not v["passed"]:
                gate_fails[g] += 1
    print("gate fails:", dict(gate_fails.most_common(6)))
    for r in rows[-4:]:
        failing = [
            (g.replace("gate_", ""), v["reason"][:90])
            for g, v in r.get("gates", {}).items()
            if not v["passed"]
        ]
        print(f"--- step={r['step']} idx={r['idx']} {r['status']} r={r['reward']}")
        if r.get("error"):
            print("   err:", r["error"][:160])
        for g, reason in failing[:3]:
            print(f"   {g}: {reason}")


if __name__ == "__main__":
    main()
