"""Aggregate audit shard results into results/audit_drkernel.json + a summary table.

Denominator policy: rows with status ref_broken / not_auditable are excluded
(the reference or task is at fault, not the candidate). cand_load_fail,
cand_native_fail and sandbox crashes/timeouts are candidate findings.
Per-gate rates are computed over rows that reached gating, counting only
pass/fail (na and error excluded from that gate's denominator).

Usage:
    uv run python scratch/audit_aggregate.py audit_out/results_shard*.jsonl \
        --json results/audit_drkernel.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from typing import Any

GATE_ORDER = [
    "CMP-01",
    "CMP-03",
    "ORD-01",
    "ORD-02",
    "ORD-03",
    "PRC-01",
    "PRC-02",
    "EXC-01",
    "EXC-02",
    "RES-01",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--accepted-only", action="store_true", help="final_speedup > 0 rows only")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["id"] in seen:
                    continue
                seen.add(row["id"])
                rows.append(row)

    if args.accepted_only:
        rows = [r for r in rows if (r.get("final_speedup") or 0) > 0]

    status_counts: Counter[str] = Counter(r["status"] for r in rows)
    excluded = {"ref_broken", "not_auditable"}
    audited = [r for r in rows if r["status"] not in excluded]
    gated = [r for r in audited if r["status"] == "gated"]

    pre_gate_findings = len(audited) - len(gated)
    aliasing = sum(1 for r in gated if r.get("output_aliasing"))

    def gate_stats(pop: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        stats: dict[str, Counter[str]] = defaultdict(Counter)
        for r in pop:
            for gate, info in r.get("gates", {}).items():
                stats[gate][info["status"]] += 1
        return {g: dict(stats[g]) for g in GATE_ORDER if g in stats}

    def any_fail(r: dict[str, Any]) -> bool:
        return any(info["status"] == "fail" for info in r.get("gates", {}).values())

    per_class: dict[str, dict[str, Any]] = {}
    for op_class in sorted({r["op_class"] for r in audited}):
        cls_audited = [r for r in audited if r["op_class"] == op_class]
        cls_gated = [r for r in cls_audited if r["status"] == "gated"]
        n_fail = sum(1 for r in cls_gated if any_fail(r))
        n_pre = len(cls_audited) - len(cls_gated)
        per_class[op_class] = {
            "audited": len(cls_audited),
            "pre_gate_findings": n_pre,
            "gated": len(cls_gated),
            "any_gate_fail": n_fail,
            "finding_rate": round((n_pre + n_fail) / max(1, len(cls_audited)), 4),
            "gates": gate_stats(cls_gated),
        }

    n_any_fail = sum(1 for r in gated if any_fail(r))
    summary = {
        "total_rows": len(rows),
        "status_counts": dict(status_counts),
        "audited": len(audited),
        "pre_gate_findings": pre_gate_findings,
        "gated": len(gated),
        "any_gate_fail": n_any_fail,
        "overall_finding_rate": round(
            (pre_gate_findings + n_any_fail) / max(1, len(audited)), 4
        ),
        "output_aliasing": aliasing,
        "gates_overall": gate_stats(gated),
        "per_class": per_class,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "per_class"}, indent=2))
    print()
    header = "| class | audited | pre-gate | any-gate-fail | finding rate |"
    print(header)
    print("|" + "---|" * 5)
    for op_class, s in per_class.items():
        print(
            f"| {op_class} | {s['audited']} | {s['pre_gate_findings']} | "
            f"{s['any_gate_fail']} | {s['finding_rate']:.1%} |"
        )
    print()
    print("| gate | pass | fail | na | error |")
    print("|" + "---|" * 5)
    for gate in GATE_ORDER:
        g = summary["gates_overall"].get(gate, {})
        print(
            f"| {gate} | {g.get('pass', 0)} | {g.get('fail', 0)} | "
            f"{g.get('na', 0)} | {g.get('error', 0)} |"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
