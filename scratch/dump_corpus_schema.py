"""Dump columns + one truncated sample row for the T4 second-corpus candidates.

Run on the box (fast HF network). Reveals the exact schema so the extractors
wrap KernelBook (Triton) and Sakana (CUDA) into the Model/ModelNew convention
correctly. CPU-only; safe to run alongside a GPU burst.

    ~/cuteenv/bin/python scratch/dump_corpus_schema.py > ~/box_out_verifier/schemas.txt 2>&1
"""

from __future__ import annotations

import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

TARGETS = [
    ("GPUMODE/KernelBook", "default/train/0000.parquet"),
    ("SakanaAI/AI-CUDA-Engineer-Archive", "default/level_1/0000.parquet"),
]


def main() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for repo, fname in TARGETS:
        print(f"\n########## {repo} :: {fname}")
        try:
            path = hf_hub_download(
                repo_id=repo, filename=fname, repo_type="dataset", revision="refs/convert/parquet"
            )
            table = pq.read_table(path)
            print(f"rows={table.num_rows} columns={table.column_names}")
            row = table.slice(0, 1).to_pylist()[0]
            for key, val in row.items():
                s = str(val)
                print(f"--- {key} (type={type(val).__name__}, len={len(s)}):")
                print(s[:1200])
            # correctness distribution if present
            for col in ("Correct", "correct"):
                if col in table.column_names:
                    import collections

                    dist = collections.Counter(str(x) for x in table.column(col).to_pylist())
                    print(f"[{col} distribution over this shard]: {dict(dist)}")
        except Exception as exc:  # schema discovery — surface, do not raise
            print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
