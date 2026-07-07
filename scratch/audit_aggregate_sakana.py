"""Aggregate the Sakana re-audit into the native differential.

Population = Sakana `Correct==True` rows (their own harness passed them). We ran
each through our 10-gate battery. Denominator excludes compile/load failures
(toolchain drift on our CUDA stack — the same environment-robustness rule the
Dr. Kernel artifact exclusion applies); only kernels that RAN are judged.

Headline: of the Correct-labelled Sakana kernels that compile on our stack,
what fraction fails at least one contract gate — and do the failing ones carry
a larger Sakana-reported `Max_Diff`.

    ~/cuteenv/bin/python scratch/audit_aggregate_sakana.py \
        "audit_out/sakana_shard*.jsonl" scratch/audit_manifest_sakana.jsonl.gz \
        --json results/audit_sakana.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import Counter
from typing import Any


def _is_compile_artifact(row: dict[str, Any]) -> bool:
    # load_inline failure surfaces at cand import -> cand_load_fail; a native
    # runtime compile issue -> cand_native_fail. For a CUDA corpus authored on
    # a different toolchain, BOTH are drift, not silent-wrongness.
    return row["status"] in ("cand_load_fail", "cand_native_fail") or (
        row["status"].startswith("sandbox_") and "compil" in row.get("error", "").lower()
    )


def _is_crash(row: dict[str, Any]) -> bool:
    return row["status"] == "gated" and any(
        "CUDA error" in g.get("reason", "") for g in row.get("gates", {}).values()
    )


# RES-01 (device residency) is confounded for CUDA-only candidates: the gate's
# CPU-input probe cannot run a .cu kernel, so it is excluded from Sakana findings.
_EXCLUDE_GATES = {"RES-01"}


def _any_fail(row: dict[str, Any]) -> bool:
    return any(
        g["status"] == "fail"
        for name, g in row.get("gates", {}).items()
        if name not in _EXCLUDE_GATES
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards")
    ap.add_argument("manifest")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    meta: dict[str, dict[str, Any]] = {}
    opener = gzip.open if args.manifest.endswith(".gz") else open
    with opener(args.manifest, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[r["id"]] = {"max_diff": r.get("sakana_max_diff"), "op_class": r.get("op_class")}

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(args.shards)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                rows.append(r)

    artifacts = [r for r in rows if _is_compile_artifact(r)]
    compiled = [r for r in rows if r["status"] == "gated"]
    findings = [r for r in compiled if _any_fail(r) or _is_crash(r)]
    denom = len(compiled)

    # Precision-regime rigidity (PRC-01/02) is near-universal for hand-written
    # CUDA kernels — they target one dtype by design and reject half precision,
    # which is arguably scope, not a bug. Report the residual with PRC stripped
    # as the environment-robust, defensible finding.
    _prc = {"RES-01", "PRC-01", "PRC-02"}
    findings_no_prc = [
        r
        for r in compiled
        if _is_crash(r)
        or any(g["status"] == "fail" for name, g in r.get("gates", {}).items() if name not in _prc)
    ]

    gate_fail: Counter[str] = Counter()
    for r in compiled:
        for g, info in r.get("gates", {}).items():
            if info["status"] == "fail":
                gate_fail[g] += 1

    def _md(ids: list[str]) -> list[float]:
        out = []
        for i in ids:
            v = meta.get(i, {}).get("max_diff")
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    finding_ids = [r["id"] for r in findings]
    pass_ids = [r["id"] for r in compiled if r not in findings]
    md_fail = _md(finding_ids)
    md_pass = _md(pass_ids)

    def _med(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        return s[len(s) // 2]

    result = {
        "corpus": "SakanaAI/AI-CUDA-Engineer-Archive",
        "population": "Sakana Correct==True",
        "audited_rows": len(rows),
        "compile_artifacts_excluded": len(artifacts),
        "compiled_denominator": denom,
        "contract_findings": len(findings),
        "finding_rate": round(len(findings) / max(1, denom), 4),
        "note": "finding_rate is PRC-dominated (fp32-only kernels); lead with the residual",
        "finding_rate_excl_precision": round(len(findings_no_prc) / max(1, denom), 4),
        "contract_findings_excl_precision": len(findings_no_prc),
        "gate_fail_counts": dict(gate_fail.most_common()),
        "max_diff_median_findings": _med(md_fail),
        "max_diff_median_passes": _med(md_pass),
        "status_counts": dict(Counter(r["status"] for r in rows)),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
