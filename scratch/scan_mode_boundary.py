"""Broad chunk_parallel-vs-serial boundary map across the three SISO ops.

Best-tuned chunk_parallel vs best-tuned serial per (op, shape), scored through
the same ``score_config_candidate`` reward path the E2.c config-RL used. The
E2.c run + the saturated mode-check settled 6 shapes (tuned chunk_parallel >=
best-tuned serial at all of them); this maps the TRUE mode boundary over a full
batch x seq_len x d_model grid so the ship decision (flip the shipped default vs
gate by shape) rests on the boundary, not on 6 points. The untested corner the
prior data flags is tiny widths (256-1024) / short L, where official CUDA beats
both modes (``cp_vs_official.json``) but best-tuned serial vs best-tuned
chunk_parallel was never measured through the reward path.

``chunk_len`` is drawn from {128, 256, 512} and ``chunk_k`` from {4, 8} -- each
divides every swept seq_len, so ``autotune.validate`` never demotes a candidate
to ``invalid_config`` on a divisibility miss (the comparison would otherwise be
silently against the shipped default). Speedups are reported vs the shipped
default (serial heuristic), so a serial tune scores ~1.0 and chunk_parallel wins
a shape iff its best speedup exceeds the best serial speedup there.

Resumable: per-(op, shape, config) rows append to a jsonl; the best-vs-best
summary is re-derived and rewritten after every shape, so a spot preemption
loses at most one shape's scoring. Emits SCAN_MODE_BOUNDARY_DONE for the
self-shutdown watcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lethe.kernels.autotune import ShapeSpec
from lethe.rl.parallel_scoring import ParallelConfigScorer

OPS = ("forward_chunked_scan", "backward_selective_scan", "fused_block_backward")
BATCHES = (1, 2, 8)
SEQ_LENS = (512, 1024, 2048, 4096, 16384)
WIDTHS = (256, 1024, 2048, 4096)
CHUNK_LENS = (128, 256, 512)


def _serial_configs(op: str) -> dict[str, dict[str, Any]]:
    cfgs = {
        "serial_default": {},
        "serial_nw4": {"scan_mode": "serial", "num_warps": 4},
        "serial_nw8": {"scan_mode": "serial", "num_warps": 8},
    }
    if op != "forward_chunked_scan":
        cfgs["serial_nw8_ck8"] = {"scan_mode": "serial", "num_warps": 8, "chunk_k": 8}
    return cfgs


def _chunk_parallel_configs(op: str) -> dict[str, dict[str, Any]]:
    cfgs: dict[str, dict[str, Any]] = {}
    for cl in CHUNK_LENS:
        cfgs[f"cp_cl{cl}_nw4"] = {"scan_mode": "chunk_parallel", "chunk_len": cl, "num_warps": 4}
    cfgs["cp_cl256_nw8"] = {"scan_mode": "chunk_parallel", "chunk_len": 256, "num_warps": 8}
    return cfgs


def _candidates(op: str) -> dict[str, str]:
    merged = {**_serial_configs(op), **_chunk_parallel_configs(op)}
    return {name: json.dumps(body) for name, body in merged.items()}


def _mode_of(name: str) -> str:
    return "chunk_parallel" if name.startswith("cp_") else "serial"


def _load_rows(path: Path) -> dict[tuple[str, int, int, int, str], dict[str, Any]]:
    rows: dict[tuple[str, int, int, int, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(r["op"], r["batch"], r["seq_len"], r["width"], r["config"])] = r
    return rows


def _best(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    passed = [
        r
        for r in rows
        if _mode_of(r["config"]) == mode and r.get("contracts_passed") and r.get("speedup")
    ]
    if not passed:
        return None
    best = max(passed, key=lambda r: r["speedup"])
    return {"config": best["config"], "speedup": best["speedup"]}


def _summarize(
    rows: dict[tuple[str, int, int, int, str], dict[str, Any]],
    work: list[tuple[str, ShapeSpec]],
    skipped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for op, shape in work:
        shape_rows = [
            r
            for (rop, b, s, w, _), r in rows.items()
            if rop == op and b == shape.batch and s == shape.seq_len and w == shape.width
        ]
        if not shape_rows:
            continue
        best_serial = _best(shape_rows, "serial")
        best_cp = _best(shape_rows, "chunk_parallel")
        entry: dict[str, Any] = {
            "op": op,
            "batch": shape.batch,
            "seq_len": shape.seq_len,
            "width": shape.width,
            "best_serial": best_serial,
            "best_chunk_parallel": best_cp,
        }
        if best_serial and best_cp:
            entry["winner"] = (
                "chunk_parallel" if best_cp["speedup"] > best_serial["speedup"] else "serial"
            )
            entry["cp_over_serial"] = round(best_cp["speedup"] / best_serial["speedup"], 3)
        elif best_cp:
            entry["winner"] = "chunk_parallel"
        elif best_serial:
            entry["winner"] = "serial"
        else:
            entry["winner"] = None
        summary.append(entry)
    summary.extend({**s, "winner": "skipped"} for s in skipped)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--timeout-s", type=float, default=1200.0)
    ap.add_argument("--skip-bwd-elems", type=float, default=3.0e8)
    ap.add_argument("--skip-fwd-elems", type=float, default=6.0e8)
    ap.add_argument("--jsonl", default="results/scan_mode_boundary.jsonl")
    ap.add_argument("--out", default="results/scan_mode_boundary.json")
    args = ap.parse_args()

    gpu_ids = tuple(int(g) for g in args.gpus.split(",") if g != "")
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)

    work: list[tuple[str, ShapeSpec]] = []
    skipped: list[dict[str, Any]] = []
    for op in OPS:
        cap = args.skip_fwd_elems if op == "forward_chunked_scan" else args.skip_bwd_elems
        for b in BATCHES:
            for s in SEQ_LENS:
                for w in WIDTHS:
                    if b * s * w > cap:
                        skipped.append(
                            {"op": op, "batch": b, "seq_len": s, "width": w, "reason": "oom_cap"}
                        )
                    else:
                        work.append((op, ShapeSpec(b, s, w)))

    rows = _load_rows(jsonl)
    print(
        f"boundary sweep: {len(work)} shape-ops to map, {len(skipped)} skipped (oom cap), "
        f"{len(rows)} rows already done",
        flush=True,
    )

    for i, (op, shape) in enumerate(work):
        candidates = _candidates(op)
        todo = {
            name: txt
            for name, txt in candidates.items()
            if (op, shape.batch, shape.seq_len, shape.width, name) not in rows
        }
        tag = f"[{i + 1}/{len(work)}] {op} b{shape.batch}/L{shape.seq_len}/d{shape.width}"
        if not todo:
            print(f"{tag}: all {len(candidates)} configs cached", flush=True)
            continue
        print(f"{tag}: scoring {len(todo)} configs", flush=True)
        scorer = ParallelConfigScorer(
            op=op, gpu_ids=gpu_ids, shape=shape, timeout_s=args.timeout_s, measure_speedup=True
        )
        names = list(todo)
        results = scorer.score_batch([todo[n] for n in names])
        with jsonl.open("a") as fh:
            for name, res in zip(names, results, strict=True):
                b = res.get("bench") or {}
                row = {
                    "op": op,
                    "batch": shape.batch,
                    "seq_len": shape.seq_len,
                    "width": shape.width,
                    "config": name,
                    "config_json": todo[name],
                    "contracts_passed": res.get("contracts_passed"),
                    "speedup": res.get("speedup"),
                    "status": res.get("status"),
                    "t_candidate_ms": b.get("t_candidate_ms"),
                    "t_baseline_ms": b.get("t_baseline_ms"),
                }
                rows[(op, shape.batch, shape.seq_len, shape.width, name)] = row
                fh.write(json.dumps(row) + "\n")
        summary = _summarize(rows, work, skipped)
        out.write_text(json.dumps(summary, indent=2))
        passed = [r for r in summary if r.get("op") == op and r["winner"] not in (None, "skipped")]
        if passed:
            last = passed[-1]
            print(
                f"{tag}: winner={last['winner']} cp/serial={last.get('cp_over_serial')}",
                flush=True,
            )

    summary = _summarize(rows, work, skipped)
    out.write_text(json.dumps(summary, indent=2))
    cp_wins = sum(1 for r in summary if r["winner"] == "chunk_parallel")
    serial_wins = sum(1 for r in summary if r["winner"] == "serial")
    print(
        f"SCAN_MODE_BOUNDARY_SUMMARY chunk_parallel_wins={cp_wins} serial_wins={serial_wins} "
        f"skipped={len(skipped)} total={len(summary)}",
        flush=True,
    )
    print("SCAN_MODE_BOUNDARY_DONE", flush=True)


if __name__ == "__main__":
    main()
