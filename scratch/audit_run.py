"""Run the rigor-gap audit over a manifest from audit_extract_drkernel.py.

Resumable: ids already present in the output file are skipped on restart
(spot-box safe). Shardable: --shard/--num-shards partition the filtered row
list for one-process-per-GPU runs (launcher sets CUDA_VISIBLE_DEVICES).

Usage (box):
    CUDA_VISIBLE_DEVICES=0 uv run python scratch/audit_run.py \
        scratch/audit_manifest_drkernel.jsonl.gz audit_out/results_shard0.jsonl \
        --device cuda --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from typing import Any

from lethe.verifier.sandbox import run_in_subprocess

SSM_ADJACENT_CLASSES = "matmul,attention,softmax,scan,norm,conv,reduction"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classes", default=SSM_ADJACENT_CLASSES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    classes = set(args.classes.split(","))
    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    opener = gzip.open if args.manifest.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(args.manifest, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["op_class"] in classes:
                rows.append(row)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    n_done_already = sum(1 for r in rows if r["id"] in done)
    print(f"[shard {args.shard}] {len(rows)} rows, {n_done_already} already done", flush=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            t0 = time.time()
            res = run_in_subprocess(
                "lethe.verifier.audit_harness",
                "audit_worker",
                (row["ref"], row["cand"], {"device": args.device}),
                timeout_s=args.timeout,
                memory_limit_mb=0,
            )
            if res.success and isinstance(res.output, dict):
                payload: dict[str, Any] = dict(res.output)
            else:
                payload = {
                    "status": f"sandbox_{res.error_class.name.lower()}",
                    "error": res.stderr[-400:],
                }
            payload["id"] = row["id"]
            payload["op_class"] = row["op_class"]
            payload["final_speedup"] = row.get("final_speedup")
            payload["elapsed_s"] = round(time.time() - t0, 1)
            out.write(json.dumps(payload) + "\n")
            out.flush()
            if (i + 1) % 10 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(rows)}", flush=True)
    print(f"[shard {args.shard}] DONE", flush=True)


if __name__ == "__main__":
    main()
