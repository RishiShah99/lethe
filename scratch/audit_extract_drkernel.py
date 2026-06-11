"""Extract an audit manifest from the Dr. Kernel coldstart-8k parquet.

Each row's reference is `original_python_code` (KernelBench convention);
the candidate is the last ```python block containing ModelNew, searched
from the final assistant turn backwards. Op classes are regex-classified
from the reference source; the driver filters to SSM-adjacent classes.

Usage:
    uv run --with pyarrow python scratch/audit_extract_drkernel.py \
        scratch/drkernel_coldstart.parquet scratch/audit_manifest_drkernel.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import Counter

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# Priority-ordered: first match wins. Patterns run against the reference
# source only (the task definition, not the candidate's implementation).
_OP_CLASSES: list[tuple[str, re.Pattern[str]]] = [
    (
        "attention",
        re.compile(r"scaled_dot_product_attention|MultiheadAttention|attention", re.IGNORECASE),
    ),
    ("scan", re.compile(r"\bcumsum\b|\bcumprod\b|\bcummax\b|\bcummin\b|selective_scan|\bscan\b")),
    ("matmul", re.compile(r"\bmatmul\b|\bbmm\b|\bmm\b|\baddmm\b|\beinsum\b|nn\.Linear|\s@\s")),
    ("conv", re.compile(r"Conv[123]d|conv[123]d|conv_transpose")),
    ("norm", re.compile(r"LayerNorm|RMSNorm|GroupNorm|BatchNorm|InstanceNorm|layer_norm|rms_norm")),
    ("softmax", re.compile(r"softmax|log_softmax", re.IGNORECASE)),
    (
        "reduction",
        re.compile(
            r"\.sum\(|\.mean\(|\.max\(|\.min\(|\.prod\(|logsumexp|\bamax\b|\bamin\b|\.var\(|\.std\(|torch\.sum|torch\.mean|torch\.max|torch\.min"
        ),
    ),
    (
        "elementwise",
        re.compile(
            r"relu|gelu|silu|sigmoid|tanh|\babs\b|\bexp\b|\blog\b|clamp|\badd\b|\bmul\b|\bsub\b|\bdiv\b",
            re.IGNORECASE,
        ),
    ),
]


def classify(ref_code: str) -> str:
    for name, pattern in _OP_CLASSES:
        if pattern.search(ref_code):
            return name
    return "other"


def extract_candidate(messages: list[dict[str, str]]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        blocks = _CODE_BLOCK.findall(msg.get("content", ""))
        for block in reversed(blocks):
            if "ModelNew" in block:
                return block
    return None


def main() -> None:
    import pyarrow.parquet as pq

    src, dst = sys.argv[1], sys.argv[2]
    table = pq.read_table(src)
    rows = table.to_pylist()
    stats: Counter[str] = Counter()
    n_written = 0
    with gzip.open(dst, "wt", encoding="utf-8") as out:
        for row in rows:
            ref = row.get("original_python_code") or ""
            if "class Model" not in ref or "get_inputs" not in ref:
                stats["skip_no_reference"] += 1
                continue
            cand = extract_candidate(row.get("messages") or [])
            if cand is None:
                stats["skip_no_candidate"] += 1
                continue
            op_class = classify(ref)
            stats[f"class_{op_class}"] += 1
            record = {
                "id": str(row.get("uuid")),
                "op_class": op_class,
                "ref": ref,
                "cand": cand,
                "final_speedup": row.get("final_speedup"),
                "num_rounds": row.get("num_rounds"),
            }
            out.write(json.dumps(record) + "\n")
            n_written += 1
    print(f"rows={len(rows)} written={n_written}")
    for key, count in sorted(stats.items()):
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
