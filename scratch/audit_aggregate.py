"""Aggregate audit shard results into results/audit_drkernel.json + summary tables.

Denominator policy (the honest-reporting rules, mirrored in the writeup):
- ref_broken / not_auditable rows are excluded entirely (the reference or
  task shape is at fault, not the candidate).
- Native-shape toolchain artifacts (Triton CompilationError /
  UnsupportedLanguageConstruct / impossible-on-any-GPU OutOfResources /
  ptxas-blackwell PTXASError) and harness IPC corruption are reported as a
  separate bucket, NOT in the headline finding rate — the corpus was
  authored against an unknown triton version and a compile failure on our
  pinned 3.7.0 stack is drift, not proven incorrectness. All other
  pre-gate failures (run errors at the task's own shape, sandbox
  crashes/timeouts/OOM, non-compile load errors) count as findings.
- Rows whose gates report CUDA-context-killing errors (illegal memory
  access, device asserts) count as crash findings even where no gate
  reaches "fail".
- The headline rate is computed over BOTH populations: all audited rows and
  the subset the source system's own harness accepted (final_speedup > 0).

Usage:
    uv run python scratch/audit_aggregate.py "audit_out/results_shard*.jsonl" \
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


def _is_artifact(row: dict[str, Any]) -> bool:
    # OutOfResources rows in this corpus request shared memory beyond ANY
    # GPU's limit (262-417 KB) — they only ever ran under a different triton
    # config selection; PTXASError on ptxas-blackwell is an sm_100 toolchain
    # bug. Both are drift, same category as CompilationError. The harness's
    # own IPC corruption (unpickle) is excluded as a harness artifact.
    if row["status"] == "sandbox_other" and "unpickle error" in row.get("error", ""):
        return True
    if row["status"] != "cand_native_fail":
        return False
    err = row.get("error", "")
    return err.startswith(
        ("CompilationError", "UnsupportedLanguageConstruct", "OutOfResources", "PTXASError")
    )


def _is_crash(row: dict[str, Any]) -> bool:
    return row["status"] == "gated" and any(
        "CUDA error" in g.get("reason", "") for g in row.get("gates", {}).values()
    )


def _any_gate_fail(row: dict[str, Any]) -> bool:
    return any(g["status"] == "fail" for g in row.get("gates", {}).values())


def _gate_stats(pop: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pop:
        for gate, info in row.get("gates", {}).items():
            stats[gate][info["status"]] += 1
    return {g: dict(stats[g]) for g in GATE_ORDER if g in stats}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = {"ref_broken", "not_auditable"}
    audited = [r for r in rows if r["status"] not in excluded]
    artifacts = [r for r in audited if _is_artifact(r)]
    denominator = [r for r in audited if not _is_artifact(r)]
    gated = [r for r in denominator if r["status"] == "gated"]
    pre_gate = [r for r in denominator if r["status"] != "gated"]
    crashes = [r for r in gated if _is_crash(r)]
    gate_fails = [r for r in gated if _any_gate_fail(r)]
    findings = len(pre_gate) + len({id(r) for r in crashes + gate_fails})

    per_class: dict[str, dict[str, Any]] = {}
    for op_class in sorted({r["op_class"] for r in denominator}):
        cls = [r for r in denominator if r["op_class"] == op_class]
        cls_gated = [r for r in cls if r["status"] == "gated"]
        cls_pre = len(cls) - len(cls_gated)
        cls_findings = cls_pre + len(
            {id(r) for r in cls_gated if _any_gate_fail(r) or _is_crash(r)}
        )
        per_class[op_class] = {
            "audited": len(cls),
            "pre_gate_findings": cls_pre,
            "gated": len(cls_gated),
            "gate_or_crash_findings": cls_findings - cls_pre,
            "finding_rate": round(cls_findings / max(1, len(cls)), 4),
            "gates": _gate_stats(cls_gated),
        }

    return {
        "total_rows": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "audited": len(audited),
        "artifacts_excluded": len(artifacts),
        "denominator": len(denominator),
        "pre_gate_findings": len(pre_gate),
        "gated": len(gated),
        "any_gate_fail": len(gate_fails),
        "cuda_crash_rows": len(crashes),
        "output_aliasing": sum(1 for r in gated if r.get("output_aliasing")),
        "finding_rate": round(findings / max(1, len(denominator)), 4),
        "gates_overall": _gate_stats(gated),
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", default="")
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

    accepted = [r for r in rows if (r.get("final_speedup") or 0) > 0]
    result = {
        "corpus": "hkust-nlp/drkernel-coldstart-8k",
        "classes": "matmul,attention,softmax,scan,norm,conv,reduction",
        "all": summarize(rows),
        "accepted_only": summarize(accepted),
    }

    for label in ("all", "accepted_only"):
        s = result[label]
        print(f"=== {label}: denominator={s['denominator']} finding_rate={s['finding_rate']:.1%}")
        print(
            f"    pre-gate={s['pre_gate_findings']} gate-fail={s['any_gate_fail']} "
            f"crash={s['cuda_crash_rows']} aliasing={s['output_aliasing']} "
            f"artifacts-excluded={s['artifacts_excluded']}"
        )
    print()
    s = result["accepted_only"]
    print("| class | audited | pre-gate | gate/crash | finding rate |")
    print("|" + "---|" * 5)
    for op_class, c in s["per_class"].items():
        print(
            f"| {op_class} | {c['audited']} | {c['pre_gate_findings']} | "
            f"{c['gate_or_crash_findings']} | {c['finding_rate']:.1%} |"
        )
    print()
    print("| gate | pass | fail | na | error |")
    print("|" + "---|" * 5)
    for gate in GATE_ORDER:
        g = s["gates_overall"].get(gate, {})
        print(
            f"| {gate} | {g.get('pass', 0)} | {g.get('fail', 0)} | "
            f"{g.get('na', 0)} | {g.get('error', 0)} |"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
