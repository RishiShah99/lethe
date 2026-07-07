"""Reproduce state-spaces/mamba#904 on Blackwell (B200, sm_100).

The claim under test: Mamba-3's backward selective-scan Triton kernel fails
to compile at num_warps >= 4 on sm_100 because Triton's eager TMEM-promotion
pass exceeds Blackwell's 512-element budget (`Required: 576, Hardware limit:
512`); users see ptxas C7907. Surviving num_warps=2 configs spill registers.

This driver is deliberately exploratory: it introspects the installed
mamba_ssm, runs a Mamba-3 forward+backward on one GPU, captures everything,
and greps the evidence. Outputs:
    out/repro_904.log          full stdout/stderr of the attempt
    out/repro_904_report.json  structured findings

Run (single GPU is enough):
    CUDA_VISIBLE_DEVICES=0 uv run python scratch/repro_904.py
"""

import importlib
import io
import json
import os
import re
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import torch

OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# Evidence patterns. The first three mirror the verifier's C7907 detection
# (src/lethe/verifier/compile.py) so the repro doubles as a
# cross-check that our load-bearing classifier would catch the real thing.
EVIDENCE_PATTERNS = {
    "c7907": re.compile(r"C7907", re.IGNORECASE),
    "internal_compiler_error": re.compile(r"internal compiler error", re.IGNORECASE),
    "ptxas_error": re.compile(r"ptxas.*error", re.IGNORECASE),
    "tmem_budget": re.compile(r"Required:\s*\d+.*Hardware limit:\s*\d+", re.IGNORECASE),
    "tmem_mention": re.compile(r"\bTMEM\b|tensor memory", re.IGNORECASE),
    "register_spill": re.compile(r"spill", re.IGNORECASE),
    "out_of_resource": re.compile(r"out of resource|OutOfResources", re.IGNORECASE),
    "shared_memory": re.compile(r"shared memory", re.IGNORECASE),
}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n== {title}\n{'=' * 70}", flush=True)


def introspect() -> dict[str, Any]:
    section("0. Introspection")
    info: dict[str, Any] = {}
    import mamba_ssm

    info["mamba_ssm_version"] = getattr(mamba_ssm, "__version__", "unknown")
    print("mamba_ssm", info["mamba_ssm_version"])

    candidates = [
        "mamba_ssm.modules.mamba3",
        "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step",
        "mamba_ssm.ops.triton.mamba3.angle_dt",
        "mamba_ssm.ops.tilelang.mamba3.mamba3_mimo",
    ]
    importable: dict[str, bool] = {}
    for name in candidates:
        try:
            importlib.import_module(name)
            importable[name] = True
        except Exception as exc:  # record-everything mode
            importable[name] = False
            print(f"  {name}: IMPORT FAILED — {type(exc).__name__}: {exc}")
    info["importable"] = importable
    print(json.dumps(importable, indent=2))
    return info


def build_layer() -> tuple[Any, dict[str, Any]]:
    section("1. Build Mamba3 layer (module defaults, bf16)")
    from mamba_ssm.modules.mamba3 import Mamba3

    cfg: dict[str, Any] = {"d_model": 2048}
    layer = Mamba3(**cfg).cuda().to(torch.bfloat16)
    n_params = sum(p.numel() for p in layer.parameters())
    meta = {
        "config": cfg,
        "n_params": n_params,
        "defaults": {
            k: getattr(layer, k)
            for k in ("d_state", "headdim", "nheads", "mimo_rank", "chunk_size")
            if hasattr(layer, k)
        },
    }
    print(json.dumps(meta, indent=2, default=str))
    return layer, meta


def run_fwd_bwd(layer: Any, batch: int, seqlen: int) -> dict[str, Any]:
    section(f"2. Forward + backward (batch={batch}, seqlen={seqlen})")
    result: dict[str, Any] = {"batch": batch, "seqlen": seqlen}
    x = torch.randn(
        batch, seqlen, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    try:
        y = layer(x)
        print("forward OK:", tuple(y.shape), flush=True)
        result["forward"] = "ok"
    except Exception:
        result["forward"] = "FAILED"
        result["forward_traceback"] = traceback.format_exc()
        print(result["forward_traceback"], flush=True)
        return result

    try:
        y.sum().backward()
        torch.cuda.synchronize()
        print("backward OK", flush=True)
        result["backward"] = "ok"
        result["grad_norm"] = float(x.grad.float().norm().item()) if x.grad is not None else None
    except Exception:
        result["backward"] = "FAILED"
        result["backward_traceback"] = traceback.format_exc()
        print(result["backward_traceback"], flush=True)
    return result


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

    log_buffer = io.StringIO()

    class Tee(io.TextIOBase):
        def __init__(self, *streams: Any) -> None:
            self.streams = streams

        def write(self, s: str) -> int:
            for st in self.streams:
                st.write(s)
                st.flush()
            return len(s)

    report: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "TRITON_PRINT_AUTOTUNING": os.environ.get("TRITON_PRINT_AUTOTUNING"),
    }
    try:
        import triton

        report["triton"] = triton.__version__
    except ImportError:
        report["triton"] = None

    tee = Tee(sys.__stdout__, log_buffer)
    with redirect_stdout(tee), redirect_stderr(tee):
        try:
            report["introspection"] = introspect()
            layer, layer_meta = build_layer()
            report["layer"] = layer_meta
            report["runs"] = [
                run_fwd_bwd(layer, batch=2, seqlen=2048),
                run_fwd_bwd(layer, batch=4, seqlen=4096),
            ]
        except Exception:
            report["driver_error"] = traceback.format_exc()
            print(report["driver_error"], flush=True)

    full_log = log_buffer.getvalue()
    report["evidence"] = {
        name: bool(pat.search(full_log)) for name, pat in EVIDENCE_PATTERNS.items()
    }
    matches = {
        name: pat.findall(full_log)[:5]
        for name, pat in EVIDENCE_PATTERNS.items()
        if pat.search(full_log)
    }
    report["evidence_matches"] = matches

    (OUT_DIR / "repro_904.log").write_text(full_log, encoding="utf-8")
    (OUT_DIR / "repro_904_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    section("3. Evidence summary")
    print(json.dumps(report["evidence"], indent=2))
    print("\nreport -> out/repro_904_report.json")


if __name__ == "__main__":
    main()
