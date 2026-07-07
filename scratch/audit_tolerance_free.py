"""Tolerance-free floor over the accepted Dr. Kernel denominator.

The 62.1% headline mixes tolerance-band gate failures (CMP-01/ORD-01/ORD-03/
PRC/EXC-02, and CMP-03 value mismatches at a new shape) with defect classes no
tolerance argument can reach. This slices out the latter — the floor a
reviewer cannot wave away with "your allclose was too tight":

  EXC-01  non-finite propagation — a candidate minting NaN/Inf where the
          reference is finite (post positional equal_nan; agreeing NaNs pass).
  ORD-02  nondeterminism — byte-unstable output across 5 calls on identical
          input in eval mode.
  aliasing  the kernel returns the same buffer across calls (structural).
  crash   CUDA context killed mid-battery (illegal access / device assert).
  pre_gate  the kernel fails to run/load/times out at its OWN native shape
          (non-artifact) — it never produced an output to compare.
  CMP-03 hard  a NEW shape hard-rejected (exception), not a value mismatch —
          reported as a separate, sympathetic add-on (shape rigidity).

CMP-03 value mismatches (max_err > atol at a new shape) DO carry a tolerance
argument, so they stay OUT of the strict floor and are quantified separately.

Usage:
    uv run python scratch/audit_tolerance_free.py "audit_out/results_shard*.jsonl" \
        --json results/audit_tolerance_free.json
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from typing import Any


def _is_artifact(row: dict[str, Any]) -> bool:
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


def _gate_fail(row: dict[str, Any], gate: str) -> bool:
    g = row.get("gates", {}).get(gate, {})
    return g.get("status") == "fail"


def _cmp03_hard(row: dict[str, Any]) -> bool:
    g = row.get("gates", {}).get("CMP-03", {})
    return g.get("status") == "fail" and "exception" in g.get("reason", "")


def _cmp03_value(row: dict[str, Any]) -> bool:
    g = row.get("gates", {}).get("CMP-03", {})
    return g.get("status") == "fail" and "exception" not in g.get("reason", "")


def load(paths: list[str]) -> list[dict[str, Any]]:
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
    return rows


def analyze(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    excluded = {"ref_broken", "not_auditable"}
    audited = [r for r in rows if r["status"] not in excluded]
    denom = [r for r in audited if not _is_artifact(r)]

    # per-row tolerance-free channel membership
    def channels(r: dict[str, Any]) -> dict[str, bool]:
        pre_gate = r["status"] != "gated"
        return {
            "exc01": _gate_fail(r, "EXC-01"),
            "ord02": _gate_fail(r, "ORD-02"),
            "aliasing": bool(r.get("output_aliasing")),
            "crash": _is_crash(r),
            "pre_gate": pre_gate,
            "cmp03_hard": _cmp03_hard(r),
            "cmp03_value": _cmp03_value(r),
        }

    strict_keys = ("exc01", "ord02", "aliasing", "crash", "pre_gate")
    per_channel: Counter[str] = Counter()
    strict_rows: list[dict[str, Any]] = []
    with_cmp03hard_rows: list[dict[str, Any]] = []
    with_cmp03all_rows: list[dict[str, Any]] = []  # HANDOFF literal (any CMP-03 fail)
    cmp03_value_only = 0

    for r in denom:
        ch = channels(r)
        for k, v in ch.items():
            if v:
                per_channel[k] += 1
        strict = any(ch[k] for k in strict_keys)
        if strict:
            strict_rows.append(r)
        if strict or ch["cmp03_hard"]:
            with_cmp03hard_rows.append(r)
        if strict or ch["cmp03_hard"] or ch["cmp03_value"]:
            with_cmp03all_rows.append(r)
        # rows the strict floor drops that fail ONLY a CMP-03 value mismatch
        if ch["cmp03_value"] and not strict and not ch["cmp03_hard"]:
            cmp03_value_only += 1

    n = max(1, len(denom))

    def per_class(pop: list[dict[str, Any]], keyset: Any) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for op in sorted({r["op_class"] for r in denom}):
            cls = [r for r in denom if r["op_class"] == op]
            hits = [r for r in cls if any(channels(r)[k] for k in keyset)]
            out[op] = {
                "denominator": len(cls),
                "tolerance_free_findings": len(hits),
                "rate": round(len(hits) / max(1, len(cls)), 4),
            }
        return out

    return {
        "label": label,
        "denominator": len(denom),
        "per_channel_rows": dict(per_channel),
        "strict_floor": {
            "channels": list(strict_keys),
            "rows": len(strict_rows),
            "rate": round(len(strict_rows) / n, 4),
        },
        "floor_with_cmp03_shape_rejection": {
            "channels": [*strict_keys, "cmp03_hard"],
            "rows": len(with_cmp03hard_rows),
            "rate": round(len(with_cmp03hard_rows) / n, 4),
        },
        "floor_with_cmp03_any": {
            "channels": [*strict_keys, "cmp03_hard", "cmp03_value"],
            "note": "HANDOFF literal channel list; CMP-03 value mismatches carry a tolerance argument",
            "rows": len(with_cmp03all_rows),
            "rate": round(len(with_cmp03all_rows) / n, 4),
        },
        "cmp03_value_only_rows": cmp03_value_only,
        "per_class_strict": per_class(denom, strict_keys),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.inputs:
        paths.extend(sorted(glob.glob(pattern)))
    rows = load(paths)
    accepted = [r for r in rows if (r.get("final_speedup") or 0) > 0]

    result = {
        "corpus": "hkust-nlp/drkernel-coldstart-8k",
        "headline_finding_rate": 0.6213,
        "all": analyze(rows, "all"),
        "accepted_only": analyze(accepted, "accepted_only"),
    }

    for label in ("all", "accepted_only"):
        s = result[label]
        print(f"=== {label}: denominator={s['denominator']}")
        print(f"    per-channel rows: {s['per_channel_rows']}")
        print(
            f"    STRICT tolerance-free floor = {s['strict_floor']['rows']} "
            f"({s['strict_floor']['rate']:.1%})"
        )
        print(
            f"    + CMP-03 hard shape-rejection = {s['floor_with_cmp03_shape_rejection']['rows']} "
            f"({s['floor_with_cmp03_shape_rejection']['rate']:.1%})"
        )
        print(
            f"    + CMP-03 any (HANDOFF literal) = {s['floor_with_cmp03_any']['rows']} "
            f"({s['floor_with_cmp03_any']['rate']:.1%})"
        )
        print(
            f"    CMP-03 value-mismatch-only rows dropped from strict floor: "
            f"{s['cmp03_value_only_rows']}"
        )
    print()
    s = result["accepted_only"]
    print("| class | denom | tol-free (strict) | rate |")
    print("|" + "---|" * 4)
    for op, c in s["per_class_strict"].items():
        print(f"| {op} | {c['denominator']} | {c['tolerance_free_findings']} | {c['rate']:.1%} |")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
