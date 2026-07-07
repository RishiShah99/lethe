"""Extract an audit manifest from SakanaAI/AI-CUDA-Engineer-Archive.

The native differential: Sakana's OWN harness labelled these kernels `Correct`
(a `torch.allclose`-class check). We re-audit the `Correct==True` rows through
our 10-gate contract battery. Any gate failure = "Sakana's harness accepts a
kernel our contracts reject" — the CUDA mirror of the Dr. Kernel finding.

Schema (verified on the box): `PyTorch_Code_Module` is a KernelBench-convention
`class Model` + `get_inputs`/`get_init_inputs`; `CUDA_Code` is a full
`torch/extension.h` source that self-binds via `PYBIND11_MODULE(..., forward)`.
We wrap `CUDA_Code` into a `ModelNew` via `torch.utils.cpp_extension.load_inline`
(the PYBIND11 block is stripped so load_inline generates the `forward` binding).

Compile failures are toolchain drift (Sakana authored on a different CUDA stack)
and are EXCLUDED from the denominator exactly as the Dr. Kernel artifact rule
excludes Triton CompilationError — only runtime value/NaN/determinism defects
are environment-robust.

Usage (box):
    ~/cuteenv/bin/python scratch/audit_extract_sakana.py \
        --levels 1 --correct-only --limit 400 \
        scratch/audit_manifest_sakana.jsonl.gz
    # validate the CUDA adapter on N rows end-to-end BEFORE the batch audit:
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/audit_extract_sakana.py --validate 2
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re

_PYBIND_RE = re.compile(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{.*?\}\s*$", re.DOTALL)
_FWD_RE = re.compile(
    r"((?:torch::Tensor|at::Tensor|std::vector\s*<[^>]+>)\s+forward\s*\([^)]*\))", re.DOTALL
)


# op-class regex (mirror scratch/audit_extract_drkernel.py::classify, kept local
# so this file has no cross-scratch import at runtime)
_OP_CLASSES: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"scaled_dot_product_attention|MultiheadAttention|attention", re.I)),
    ("scan", re.compile(r"\bcumsum\b|\bcumprod\b|\bcummax\b|\bcummin\b|selective_scan|\bscan\b")),
    ("matmul", re.compile(r"\bmatmul\b|\bbmm\b|\bmm\b|\baddmm\b|\beinsum\b|nn\.Linear|\s@\s")),
    ("conv", re.compile(r"Conv[123]d|conv[123]d|conv_transpose")),
    ("norm", re.compile(r"LayerNorm|RMSNorm|GroupNorm|BatchNorm|InstanceNorm|layer_norm|rms_norm")),
    ("softmax", re.compile(r"softmax|log_softmax", re.I)),
    (
        "reduction",
        re.compile(
            r"\.sum\(|\.mean\(|\.max\(|\.min\(|\.prod\(|logsumexp|torch\.(sum|mean|max|min)"
        ),
    ),
    ("elementwise", re.compile(r"relu|gelu|silu|sigmoid|tanh|\babs\b|\bexp\b|clamp", re.I)),
]


def _classify(ref: str) -> str:
    for name, pat in _OP_CLASSES:
        if pat.search(ref):
            return name
    return "other"


def _make_cand_source(cuda_code: str, ext_name: str) -> str:
    # load_inline auto-generates the pybind glue for `functions` in a main.cpp
    # that must SEE the declaration — so strip the source's own PYBIND11_MODULE
    # and hand load_inline a forward-declaration as cpp_sources. If no `forward`
    # signature is found, keep the source's PYBIND11_MODULE and bind nothing.
    # Sources are base64-embedded to avoid every quote/backslash escaping trap.
    m = _FWD_RE.search(cuda_code)
    if m:
        stripped = _PYBIND_RE.sub("", cuda_code)
        cpp_b64 = base64.b64encode((m.group(1).strip() + ";").encode()).decode()
        cuda_b64 = base64.b64encode(stripped.encode()).decode()
        funcs = '["forward"]'
    else:
        cpp_b64 = base64.b64encode(b"").decode()
        cuda_b64 = base64.b64encode(cuda_code.encode()).decode()
        funcs = "None"
    return (
        "import base64\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.cpp_extension import load_inline\n"
        f'_cpp = base64.b64decode("{cpp_b64}").decode()\n'
        f'_cuda = base64.b64decode("{cuda_b64}").decode()\n'
        f'_mod = load_inline(name="{ext_name}", cpp_sources=_cpp, cuda_sources=_cuda,\n'
        f"                   functions={funcs}, with_cuda=True, verbose=False)\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self, *a, **k):\n"
        "        super().__init__()\n"
        "    def forward(self, *inputs):\n"
        # inputs arrive on the harness device; do NOT force .cuda() — that would
        # trip RES-01 (input cpu -> output cuda) as an adapter artifact.
        "        return _mod.forward(*inputs)\n"
    )


def _iter_parquet(levels: list[int]):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for lv in levels:
        path = hf_hub_download(
            repo_id="SakanaAI/AI-CUDA-Engineer-Archive",
            filename=f"default/level_{lv}/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
        )
        for row in pq.read_table(path).to_pylist():
            yield lv, row


def build_manifest(args: argparse.Namespace) -> None:
    n_written = 0
    n_seen = 0
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for lv, row in _iter_parquet(args.levels):
            n_seen += 1
            if args.correct_only and not row.get("Correct"):
                continue
            ref = row.get("PyTorch_Code_Module") or ""
            cuda = row.get("CUDA_Code") or ""
            if "class Model" not in ref or "get_inputs" not in ref or not cuda.strip():
                continue
            rid = f"sakana_L{lv}_{row.get('Task_ID')}_{row.get('__index_level_0__')}"
            rec = {
                "id": rid,
                "op_class": _classify(ref),
                "ref": ref,
                "cand": _make_cand_source(cuda, f"sk_{lv}_{row.get('Task_ID')}_{n_written}"),
                "final_speedup": row.get("CUDA_Speedup_Native"),
                "sakana_correct": bool(row.get("Correct")),
                "sakana_max_diff": row.get("Max_Diff"),
            }
            out.write(json.dumps(rec) + "\n")
            n_written += 1
            if args.limit and n_written >= args.limit:
                break
    print(f"seen={n_seen} written={n_written} -> {args.out}")


def validate(n: int) -> None:
    """Compile + audit the first n Correct rows end-to-end on the box."""
    from lethe.verifier.audit_harness import audit_worker

    count = 0
    for _lv, row in _iter_parquet([1]):
        if not row.get("Correct"):
            continue
        ref = row.get("PyTorch_Code_Module") or ""
        cuda = row.get("CUDA_Code") or ""
        if "class Model" not in ref or "get_inputs" not in ref:
            continue
        cand = _make_cand_source(cuda, f"skval_{count}")
        print(
            f"\n=== validate {count}: {row.get('Op_Name')} (Sakana Max_Diff={row.get('Max_Diff')})"
        )
        res = audit_worker(ref, cand, {"device": "cuda"})
        gates = res.get("gates", {})
        summary = {g: v.get("status") for g, v in gates.items()}
        print("  status:", res.get("status"), res.get("error", ""))
        print("  gates:", summary)
        count += 1
        if count >= n:
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default="scratch/audit_manifest_sakana.jsonl.gz")
    ap.add_argument("--levels", default="1", help="comma list, e.g. 1,2,3")
    ap.add_argument("--correct-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--validate", type=int, default=0, help="compile+audit N rows, no manifest")
    args = ap.parse_args()
    args.levels = [int(x) for x in str(args.levels).split(",")]
    if args.validate:
        validate(args.validate)
        return
    build_manifest(args)


if __name__ == "__main__":
    main()
