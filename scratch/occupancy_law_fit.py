"""Fit and validate a serial-vs-chunk-parallel occupancy crossover law.

Regime model:
  serial: latency-bound, one program per (batch, ceil(D/block_d)) walks all L;
          t_serial should be roughly flat as long as serial_programs << SM count
          (the SMs never saturate), then grow once programs > SM capacity.
  chunk-parallel: throughput-bound; grid is batch * ceil(D/block_d) * n_chunks
                  so t_chunk scales with work / parallelism and decreases as
                  n_chunks grows.

Crossover criterion: the serial_programs value where t_serial == t_chunk
(linearly interpolated between adjacent rows).

The 3.3x-SM-count claim is derived from the B200 SM count. This script
checks whether the fitted crossover is consistent with that claim or differs.

    uv run python scratch/occupancy_law_fit.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RESULTS = Path("results")

# ── SM count ──────────────────────────────────────────────────────────────────
# env_b200.json does not record multiProcessorCount.
# The NVIDIA B200 SXM has 148 SMs per the NVIDIA product brief (sm_100, GH200
# successor); no per-run query result is stored in any results/*.json file.
# Source: NVIDIA B200 GPU Architecture Whitepaper (2024).
SM_COUNT = 148
SM_SOURCE = "NVIDIA B200 product brief (2024); not recorded in results/*.json"

# ── Serial-programs grid formula (per op) ────────────────────────────────────
# All three ops use grid = (batch, ceil(d_model / block_d)).
# Default block_d:
#   forward_chunked_scan  : min(64, next_power_of_2(d_model)) → 64 for d>=64
#   backward_selective_scan: min(64, max(16, 2048 // next_power_of_2(n_state)))
#                            with n_state=16 → min(64, max(16, 128)) = 64
#   fused_block_backward  : min(64, next_power_of_2(d_model)) → 64 for d>=64
# All swept widths (256, 1024, 2048, 4096) are >= 64, so block_d = 64 for every
# shape in the boundary sweep. Therefore:
#   serial_programs = batch * ceil(width / 64)
SERIAL_PROGRAMS_FORMULA = {
    "forward_chunked_scan": "batch * ceil(width / 64)  [block_d=min(64,next_pow2(D))=64 for D>=64]",
    "backward_selective_scan": "batch * ceil(width / 64)  [block_d=min(64,max(16,2048//next_pow2(N)))=64 at N=16]",
    "fused_block_backward": "batch * ceil(width / 64)  [block_d=min(64,next_pow2(D))=64 for D>=64]",
}


def _serial_programs(batch: int, width: int) -> int:
    block_d = 64
    return batch * math.ceil(width / block_d)


# ── Section 1: fit on cp_occupancy.json ──────────────────────────────────────


def fit_occupancy(rows: list[dict]) -> dict:
    # Sort by serial_programs so interpolation is stable.
    rows = sorted(rows, key=lambda r: r["serial_programs"])

    progs = [r["serial_programs"] for r in rows]
    t_serial = [r["t_serial_ms"] for r in rows]
    t_chunk = [r["t_chunk_ms"] for r in rows]
    speedup = [r["speedup"] for r in rows]

    # Serial flatness: split rows into under-SM (progs < SM_COUNT) and over-SM.
    under = [r for r in rows if r["serial_programs"] < SM_COUNT]
    over = [r for r in rows if r["serial_programs"] >= SM_COUNT]

    def _cv(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        if mean == 0:
            return 0.0
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        return std / mean

    cv_under = _cv([r["t_serial_ms"] for r in under]) if under else None
    cv_over = _cv([r["t_serial_ms"] for r in over]) if over else None

    # Chunk scaling: fit t_chunk ~ a * serial_programs (linear through origin
    # is the simplest throughput model; chunk-parallel grid ~ batch*ceil(D/64)*n_chunks
    # and n_chunks = L/chunk_len is constant at fixed L, so t_chunk should scale
    # linearly with batch*width/64 = serial_programs at fixed L).
    # Simple OLS through origin on (serial_programs, t_chunk).
    sp_vals = [float(p) for p in progs]
    tc_vals = list(t_chunk)
    num = sum(s * t for s, t in zip(sp_vals, tc_vals, strict=True))
    den = sum(s * s for s in sp_vals)
    chunk_slope = num / den if den else None

    # Crossover: find where t_serial = t_chunk by linear interpolation.
    # Walk pairs; the crossover is where speedup crosses 1.0 (t_serial = t_chunk).
    crossover_progs = None
    for i in range(len(rows) - 1):
        sp0, sp1 = speedup[i], speedup[i + 1]
        # speedup > 1 means chunk-parallel wins; crossover = speedup drops to 1.
        if sp0 >= 1.0 >= sp1 or sp0 <= 1.0 <= sp1:
            # Linear interpolation on speedup vs serial_programs.
            p0, p1 = progs[i], progs[i + 1]
            if sp1 == sp0:
                crossover_progs = (p0 + p1) / 2
            else:
                frac = (1.0 - sp0) / (sp1 - sp0)
                crossover_progs = p0 + frac * (p1 - p0)
            break

    crossover_over_sm = crossover_progs / SM_COUNT if crossover_progs is not None else None

    return {
        "rows_n": len(rows),
        "serial_programs_range": [min(progs), max(progs)],
        "t_serial_ms_range": [min(t_serial), max(t_serial)],
        "t_chunk_ms_range": [min(t_chunk), max(t_chunk)],
        "serial_flat_evidence": {
            "under_sm_rows": len(under),
            "over_sm_rows": len(over),
            "cv_under_sm": round(cv_under, 4) if cv_under is not None else None,
            "cv_over_sm": round(cv_over, 4) if cv_over is not None else None,
            "note": (
                "CV (std/mean) of t_serial for rows with serial_programs < SM vs >= SM; "
                "low CV under-SM supports latency-bound flatness claim"
            ),
        },
        "chunk_scaling": {
            "model": "t_chunk ~ slope * serial_programs (OLS through origin, fixed L=16384)",
            "slope_ms_per_program": round(chunk_slope, 6) if chunk_slope is not None else None,
            "r2": _r2_through_origin(sp_vals, tc_vals, chunk_slope),
        },
        "crossover_programs": round(crossover_progs, 1) if crossover_progs is not None else None,
        "crossover_over_sm": round(crossover_over_sm, 3) if crossover_over_sm is not None else None,
        "speedup_at_each_row": [
            {"serial_programs": p, "speedup": s} for p, s in zip(progs, speedup, strict=True)
        ],
    }


def _r2_through_origin(x: list[float], y: list[float], slope: float | None) -> float | None:
    if slope is None:
        return None
    y_pred = [slope * xi for xi in x]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y, y_pred, strict=True))
    y_mean = sum(y) / len(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    return round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else None


# ── Section 2: validate on boundary sweep ────────────────────────────────────


def validate_boundary(boundary: list[dict], crossover_programs: float | None) -> dict:
    scored = [r for r in boundary if r.get("winner") not in ("skipped", None)]

    if crossover_programs is None:
        return {"error": "no crossover fitted; cannot validate"}

    correct = 0
    per_op: dict[str, dict[str, int]] = {}
    wrong_shapes: list[dict] = []

    always_cp_correct = 0
    l_gt_512_correct = 0

    for r in scored:
        op = r["op"]
        batch = r["batch"]
        width = r["width"]
        seq_len = r["seq_len"]
        actual_winner = r["winner"]

        sp = _serial_programs(batch, width)
        predicted = "chunk_parallel" if sp < crossover_programs else "serial"

        if op not in per_op:
            per_op[op] = {"correct": 0, "total": 0}
        per_op[op]["total"] += 1

        if predicted == actual_winner:
            correct += 1
            per_op[op]["correct"] += 1
        else:
            wrong_shapes.append(
                {
                    "op": op,
                    "batch": batch,
                    "seq_len": seq_len,
                    "width": width,
                    "serial_programs": sp,
                    "predicted": predicted,
                    "actual": actual_winner,
                    "cp_over_serial": r.get("cp_over_serial"),
                }
            )

        # Baseline 1: always chunk_parallel.
        if actual_winner == "chunk_parallel":
            always_cp_correct += 1

        # Baseline 2: L > 512 ⇒ chunk_parallel, else serial.
        baseline_l = "chunk_parallel" if seq_len > 512 else "serial"
        if baseline_l == actual_winner:
            l_gt_512_correct += 1

    n = len(scored)
    per_op_summary = {
        op: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else None,
        }
        for op, v in per_op.items()
    }

    return {
        "n_shapes": n,
        "accuracy": round(correct / n, 4) if n else None,
        "correct": correct,
        "per_op": per_op_summary,
        "wrong_shapes": wrong_shapes,
        "baselines": {
            "always_cp": {
                "correct": always_cp_correct,
                "accuracy": round(always_cp_correct / n, 4) if n else None,
            },
            "l_gt_512": {
                "correct": l_gt_512_correct,
                "accuracy": round(l_gt_512_correct / n, 4) if n else None,
            },
        },
    }


# ── Section 3: TMEM-elements check on #904 repro data ────────────────────────


def check_tmem_904(report: dict) -> dict:
    # The repro report records evidence_matches["tmem_budget"] with strings like
    # "Required: 544, Hardware limit: 512". No per-config tile sizes are stored,
    # so we cannot reproduce the f(tile, warps, stages) calculation from the JSON
    # alone. We can parse the Required/Limit values from the evidence strings.
    matches = report.get("evidence_matches", {}).get("tmem_budget", [])
    if not matches:
        return {"status": "insufficient_data", "note": "no tmem_budget evidence matches in report"}

    parsed: list[dict] = []
    for m in matches:
        try:
            parts = m.split(",")
            req = int(parts[0].split(":")[1].strip())
            lim = int(parts[1].split(":")[1].strip())
            parsed.append({"required": req, "limit": lim, "exceeds": req > lim})
        except Exception:
            parsed.append({"raw": m, "parse_error": True})

    unique_vals = {(p["required"], p["limit"]) for p in parsed if "required" in p}
    # The report has no per-config tile breakdown (no num_warps/block_size per row),
    # so the f(tile, warps, stages) → tmem_elements formula cannot be validated
    # against a pass/fail split from this JSON. The evidence does confirm that the
    # failing config exceeds the 512-element TMEM budget.
    note = (
        "repro_904_report.json records Required/Limit pairs from compiler error strings "
        "but contains no per-config tile breakdown; f(tile,warps,stages) vs 512 cannot "
        "be validated against a pass/fail split from this data alone."
    )
    return {
        "status": "partial",
        "evidence_count": len(matches),
        "unique_required_limit_pairs": [
            {"required": req_v, "limit": lim_v} for req_v, lim_v in sorted(unique_vals)
        ],
        "all_exceed_limit": all(p.get("exceeds", False) for p in parsed if "required" in p),
        "note": note,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    occ_rows = json.loads((RESULTS / "cp_occupancy.json").read_text())
    boundary = json.loads((RESULTS / "scan_mode_boundary.json").read_text())
    repro_report = json.loads((RESULTS / "repro_904_report.json").read_text())

    fit = fit_occupancy(occ_rows)
    crossover = fit["crossover_programs"]
    validation = validate_boundary(boundary, crossover)
    tmem = check_tmem_904(repro_report)

    # Assess whether 3.3x claim is consistent with fit.
    three_pt_three_x = 3.3 * SM_COUNT
    claim_assessment: str
    if crossover is None:
        claim_assessment = "no crossover found in data"
    else:
        ratio = crossover / SM_COUNT
        if abs(ratio - 3.3) / 3.3 < 0.15:
            claim_assessment = f"consistent: fitted {ratio:.2f}x vs claimed 3.3x (within 15%)"
        else:
            claim_assessment = (
                f"FALSIFIED: fitted crossover {crossover:.0f} programs = {ratio:.2f}x SM count "
                f"vs claimed 3.3x ({three_pt_three_x:.0f} programs)"
            )

    caveats = [
        "cp_occupancy.json has only 7 rows at fixed L=16384 and n_state=16; "
        "crossover interpolation spans a gap from 128 to 512 programs (4x range).",
        "SM_COUNT=148 is from the NVIDIA B200 product brief; not queried at runtime "
        "in any results/*.json file (env_b200.json lacks multiProcessorCount).",
        "The boundary sweep used n_state=16 for all ops; the serial block_d formula "
        "depends on n_state for backward_selective_scan (block_d=min(64,2048//block_n)); "
        "at n_state=16 block_d=64 as for forward, but at n_state=128 block_d=16 "
        "(serial_programs = 4x higher for same batch/width).",
        "chunk-parallel speedup in the boundary sweep is measured relative to the "
        "shipped serial baseline (speedup~1.0), not wall-clock t_serial vs t_chunk; "
        "the cp_occupancy fit uses raw ms times. The two measurements are consistent "
        "in direction but not directly comparable in magnitude.",
        "Baseline 'L>512 => chunk_parallel' is the shipped selector heuristic; "
        "accuracy below it does not necessarily indicate the law is worse — "
        "the shipped heuristic was tuned on a different data regime.",
    ]

    out = {
        "schema": "occupancy_law v1",
        "sm_count": {"value": SM_COUNT, "source": SM_SOURCE},
        "serial_programs_formula": SERIAL_PROGRAMS_FORMULA,
        "fit": fit,
        "claim_3_3x": {
            "claimed_crossover_programs": round(three_pt_three_x, 1),
            "assessment": claim_assessment,
        },
        "validation": validation,
        "tmem_904": tmem,
        "caveats": caveats,
    }

    out_path = RESULTS / "occupancy_law.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {out_path}")

    # Human-readable summary.
    print("\n=== Fit summary ===")
    print(f"SM count: {SM_COUNT} ({SM_SOURCE})")
    print(f"Crossover: {crossover} programs = {fit['crossover_over_sm']}x SM count")
    print(f"Claim assessment: {claim_assessment}")
    print(f"Chunk slope R²: {fit['chunk_scaling']['r2']}")
    print(f"Serial CV under-SM: {fit['serial_flat_evidence']['cv_under_sm']}")
    print(f"Serial CV over-SM:  {fit['serial_flat_evidence']['cv_over_sm']}")

    print(f"\n=== Validation ({validation['n_shapes']} scored shapes) ===")
    print(
        f"Occupancy law accuracy: {validation['accuracy']:.1%}  ({validation['correct']}/{validation['n_shapes']})"
    )
    print(f"Baseline always-CP:     {validation['baselines']['always_cp']['accuracy']:.1%}")
    print(f"Baseline L>512:         {validation['baselines']['l_gt_512']['accuracy']:.1%}")
    print("\nPer-op accuracy:")
    for op, v in validation["per_op"].items():
        short = op.replace("_chunked_scan", "").replace("_selective_scan", "").replace("_block", "")
        print(f"  {short:30s}  {v['accuracy']:.1%}  ({v['correct']}/{v['total']})")

    print(f"\nWrong shapes ({len(validation['wrong_shapes'])}):")
    for s in validation["wrong_shapes"]:
        print(
            f"  op={s['op']:30s}  b={s['batch']}  L={s['seq_len']:5d}  d={s['width']:4d}  "
            f"sp={s['serial_programs']:4d}  pred={s['predicted']:14s}  actual={s['actual']:14s}  "
            f"cp/serial={s['cp_over_serial']}"
        )

    print("\n=== TMEM #904 check ===")
    print(f"Status: {tmem['status']}")
    print(f"Note: {tmem['note']}")


if __name__ == "__main__":
    main()
