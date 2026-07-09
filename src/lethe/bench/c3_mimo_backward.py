"""C3 benchmark: our Triton MIMO backward vs the loop oracle.

No open-source MIMO backward exists to compare against: the official repo
ships the MIMO forward (TileLang) and decode-step (Triton) with readable
source, but its training backward (``mamba_mimo_bwd_combined``) is a
compiled TileLang artifact with fused-rotary semantics, a different op
signature (raw B/C + angles vs our pre-rotated B/C), source unfetchable.
That absence is the comparator gap; the comparator set is therefore:

- ``ours_triton``: this repo's hand-written MIMO backward (C3)
- ``reference_loop_bwd``: the Python-loop oracle's autograd (small shapes;
  the only other implementation of this op anywhere)

Plus the #904-contrast artifact carried over from C2: ``num_warps_sweep``
compiles and times our kernel at num_warps 2/4/8 with per-specialisation
ptxas resources. Our kernel has no ``tl.dot`` for the TMEM-promotion pass
to touch, so the sweep succeeding on sm_100 is the claim, recorded with
evidence.

Usage: ``uv run python -m
lethe.bench.c3_mimo_backward --out ~/out/c3_bench.json``
(add ``--quick`` for a fast smoke pass).
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lethe.kernels.ops import mimo_backward, triton_mimo_bwd_resource_meta
from lethe.kernels.references.mimo_backward import (
    MimoGrads,
    reference_mimo_backward,
)

from .c2_backward_selective_scan import _parity_stats, _time

MimoArgs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

# The Python-loop oracle materialises the per-step state plus its autograd
# graph; cap the loop length to keep its run feasible.
_REFERENCE_LOOP_MAX_SEQ = 256


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seq_len: int
    rank: int
    nheads: int
    headdim: int
    n_state: int

    def label(self) -> str:
        return (
            f"B{self.batch}xL{self.seq_len}xR{self.rank}"
            f"xH{self.nheads}xP{self.headdim}xN{self.n_state}"
        )


# Training-like sizing from the official mamba3 defaults: headdim 64,
# d_state 128, mimo_rank in {1, 2, 4}; H=32 makes d_inner=2048.
SHAPES = [
    ShapeSpec(2, 256, 2, 4, 16, 32),  # oracle shape: every comparator runs
    ShapeSpec(8, 2048, 1, 32, 64, 128),  # R sweep at the training shape
    ShapeSpec(8, 2048, 2, 32, 64, 128),
    ShapeSpec(8, 2048, 4, 32, 64, 128),
    ShapeSpec(8, 4096, 2, 32, 64, 128),
    ShapeSpec(2, 16384, 2, 32, 64, 128),  # long-sequence regime
]
QUICK_SHAPES = [ShapeSpec(2, 256, 2, 4, 16, 32), ShapeSpec(4, 2048, 2, 16, 64, 128)]

DTYPES = [torch.float32, torch.bfloat16]

SWEEP_SHAPE = ShapeSpec(8, 2048, 2, 32, 64, 128)
SWEEP_WARPS = (2, 4, 8)


def _make_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> tuple[MimoArgs, Tensor]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    b, length, r, h, p, n = (
        spec.batch,
        spec.seq_len,
        spec.rank,
        spec.nheads,
        spec.headdim,
        spec.n_state,
    )
    x = torch.randn(b, length, h, p, device=dev).to(dtype)
    b_proj = torch.randn(b, length, r, h, n, device=dev).to(dtype)
    c_proj = torch.randn(b, length, r, h, n, device=dev).to(dtype)
    dt = (torch.rand(b, length, h, device=dev) * 0.1 + 1e-3).to(dtype)
    alpha = torch.exp(-dt.float() * torch.rand(h, device=dev)).to(dtype)
    mimo_x = (1.0 / r + torch.randn(h, r, p, device=dev) * 0.1).to(dtype)
    mimo_o = (1.0 / r + torch.randn(h, r, p, device=dev) * 0.1).to(dtype)
    dy = torch.randn(b, length, h, p, device=dev).to(dtype)
    return (x, b_proj, c_proj, dt, alpha, mimo_x, mimo_o), dy


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    args, dy = _make_inputs(spec, dtype)
    trials = 20 if quick else 50
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "seq_len": spec.seq_len,
            "rank": spec.rank,
            "nheads": spec.nheads,
            "headdim": spec.headdim,
            "n_state": spec.n_state,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "impls": {},
        "skipped": {
            "official": (
                "no open MIMO backward exists; the TileLang "
                "mamba_mimo_bwd_combined is a fused-rotary compiled artifact "
                "with a different op signature"
            )
        },
        "parity": {},
    }
    impls: dict[str, Any] = row["impls"]
    parity: dict[str, Any] = row["parity"]

    grads_ours = mimo_backward(*args, dy)
    impls["ours_triton"] = _time(lambda: mimo_backward(*args, dy), warmup=10, trials=trials)

    if spec.seq_len <= _REFERENCE_LOOP_MAX_SEQ and dtype == torch.float32:
        grads_ref = reference_mimo_backward(*args, dy)
        parity["ours_vs_reference"] = {
            field: _parity_stats(got, want)
            for field, got, want in zip(MimoGrads._fields, grads_ours, grads_ref, strict=True)
        }
        impls["reference_loop_bwd"] = _time(
            lambda: reference_mimo_backward(*args, dy), warmup=1, trials=3
        )
    else:
        row["skipped"]["reference_loop_bwd"] = "sequence too long for the Python-loop oracle"

    ours_ms = impls["ours_triton"]["median_ms"]
    row["speedups_vs_ours"] = {
        name: timing["median_ms"] / ours_ms
        for name, timing in impls.items()
        if name != "ours_triton"
    }
    return row


def _specialization_table() -> list[dict[str, Any]]:
    """Per-compiled-specialisation resources from the kernel cache.

    Attribute layout is triton-version dependent; missing fields record as
    None rather than guessing (same convention as the C2 table).
    """
    from lethe.kernels.ops import _triton_mimo_bwd

    jit_fn = _triton_mimo_bwd._mimo_bwd_kernel
    compiled: list[Any] = []
    caches = getattr(jit_fn, "device_caches", None)
    if isinstance(caches, dict):
        for entry in caches.values():
            cache_dict = entry[0] if isinstance(entry, tuple) else entry
            if isinstance(cache_dict, dict):
                compiled.extend(cache_dict.values())
    rows: list[dict[str, Any]] = []
    for kernel in compiled:
        metadata = getattr(kernel, "metadata", None)
        rows.append(
            {
                "num_warps": getattr(metadata, "num_warps", None),
                "n_regs": getattr(kernel, "n_regs", None),
                "spill_bytes": getattr(kernel, "n_spills", None),
                "shared_bytes": getattr(metadata, "shared", None),
                "name": getattr(metadata, "name", None),
            }
        )
    return rows


def _num_warps_sweep(quick: bool) -> dict[str, Any]:
    """Compile + run + time our kernel at num_warps 2/4/8 on sm_100."""
    from lethe.kernels.ops import _triton_mimo_bwd

    spec = QUICK_SHAPES[1] if quick else SWEEP_SHAPE
    args, dy = _make_inputs(spec, torch.float32)
    trials = 10 if quick else 30
    sweep: dict[str, Any] = {"shape": spec.label(), "configs": {}}
    base: tuple[Tensor, ...] | None = None

    def _runner(warps_cfg: int) -> Callable[[], tuple[Tensor, ...]]:
        def run() -> tuple[Tensor, ...]:
            return _triton_mimo_bwd.launch_mimo_backward(*args, dy, num_warps=warps_cfg)

        return run

    for warps in SWEEP_WARPS:
        entry: dict[str, Any] = {}
        run_cfg = _runner(warps)
        try:
            grads = run_cfg()
            entry["compiles"] = True
            entry.update(_time(run_cfg, warmup=5, trials=trials))
            if base is None:
                base = grads
            else:
                entry["max_scale_rel_vs_first_config"] = max(
                    _parity_stats(g, b)["scale_rel"] for g, b in zip(grads, base, strict=True)
                )
        except Exception:
            entry["compiles"] = False
            entry["error"] = traceback.format_exc(limit=5)
        sweep["configs"][f"num_warps={warps}"] = entry
    sweep["specializations"] = _specialization_table()
    sweep["official_mamba3_context"] = (
        "the sibling SISO Triton backward fails compile at all num_warps>=4 "
        "on sm_100 (TMEM 544 > 512); no open MIMO backward exists to sweep. "
        "evidence: docs/904_reproducer.md, results/repro_904_report.json"
    )
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "out" / "c3_bench.json",
        help="output JSON path",
    )
    parser.add_argument("--quick", action="store_true", help="small shapes, fewer trials")
    cli = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")

    triton_version: str | None
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:  # pragma: no cover
        triton_version = None

    report: dict[str, Any] = {
        "op": "mimo_backward",
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "triton": triton_version,
            "python": platform.python_version(),
        },
        "runs": [],
    }

    shapes = QUICK_SHAPES if cli.quick else SHAPES
    for spec in shapes:
        for dtype in DTYPES:
            print(f"[bench] {spec.label()} {dtype} ...", flush=True)
            report["runs"].append(_run_shape(spec, dtype, cli.quick))
            torch.cuda.empty_cache()

    print("[bench] num_warps sweep ...", flush=True)
    report["num_warps_sweep"] = _num_warps_sweep(cli.quick)
    report["resource_meta"] = triton_mimo_bwd_resource_meta()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
