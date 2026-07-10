"""C4 benchmark: our fused rotary scan vs the loop oracle."""

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

from lethe.kernels.ops import complex_scan_rope, triton_complex_rope_resource_meta
from lethe.kernels.references.complex_scan_rope import reference_complex_scan_rope

from .c2_backward_selective_scan import _parity_stats, _time

RopeArgs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

# The Python-loop oracle steps t-by-t in eager torch; cap its length.
_REFERENCE_LOOP_MAX_SEQ = 256


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seq_len: int
    nheads: int
    headdim: int
    n_state: int
    num_angles: int

    def label(self) -> str:
        return (
            f"B{self.batch}xL{self.seq_len}xH{self.nheads}"
            f"xP{self.headdim}xN{self.n_state}xS{self.num_angles}"
        )


# Training-like mamba3 sizing: headdim 64, d_state 128, full rotary S=N/2, H=32 (d_inner 2048).
SHAPES = [
    ShapeSpec(2, 256, 4, 16, 32, 16),  # oracle shape: every comparator runs
    ShapeSpec(8, 2048, 32, 64, 128, 64),  # training shape, full rotary
    ShapeSpec(8, 2048, 32, 64, 128, 16),  # partial rotary (32 of 128 lanes)
    ShapeSpec(8, 4096, 32, 64, 128, 64),
    ShapeSpec(2, 16384, 32, 64, 128, 64),  # long-sequence regime
]
QUICK_SHAPES = [ShapeSpec(2, 256, 4, 16, 32, 16), ShapeSpec(4, 2048, 16, 64, 128, 64)]

DTYPES = [torch.float32, torch.bfloat16]

SWEEP_SHAPE = ShapeSpec(8, 2048, 32, 64, 128, 64)
SWEEP_WARPS = (2, 4, 8)


def _make_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> RopeArgs:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    b, length, h, p, n, s = (
        spec.batch,
        spec.seq_len,
        spec.nheads,
        spec.headdim,
        spec.n_state,
        spec.num_angles,
    )
    x = torch.randn(b, length, h, p, device=dev).to(dtype)
    bb = torch.randn(b, length, h, n, device=dev).to(dtype)
    cc = torch.randn(b, length, h, n, device=dev).to(dtype)
    dt = (torch.rand(b, length, h, device=dev) * 0.1 + 1e-3).to(dtype)
    a = (-torch.rand(h, device=dev)).to(dtype)
    angle = torch.randn(b, length, h, s, device=dev).to(dtype)
    return x, bb, cc, dt, a, angle


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    args = _make_inputs(spec, dtype)
    trials = 20 if quick else 50
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "seq_len": spec.seq_len,
            "nheads": spec.nheads,
            "headdim": spec.headdim,
            "n_state": spec.n_state,
            "num_angles": spec.num_angles,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "impls": {},
        "skipped": {
            "official": (
                "no open full-sequence rotary scan exists; official ships the "
                "decode-step rotary kernel and a standalone angle_dt cumsum, "
                "and the training path is a compiled TileLang artifact"
            )
        },
        "parity": {},
    }
    impls: dict[str, Any] = row["impls"]

    y_ours = complex_scan_rope(*args)
    impls["ours_triton"] = _time(lambda: complex_scan_rope(*args), warmup=10, trials=trials)

    if spec.seq_len <= _REFERENCE_LOOP_MAX_SEQ and dtype == torch.float32:
        y_ref = reference_complex_scan_rope(*args)
        row["parity"]["ours_vs_reference"] = _parity_stats(y_ours, y_ref)
        impls["reference_loop"] = _time(
            lambda: reference_complex_scan_rope(*args), warmup=1, trials=3
        )
    else:
        row["skipped"]["reference_loop"] = "sequence too long for the Python-loop oracle"

    ours_ms = impls["ours_triton"]["median_ms"]
    row["speedups_vs_ours"] = {
        name: timing["median_ms"] / ours_ms
        for name, timing in impls.items()
        if name != "ours_triton"
    }
    return row


def _specialization_table() -> list[dict[str, Any]]:
    """Per-compiled-specialisation resources from the kernel cache."""
    from lethe.kernels.ops import _triton_complex_rope

    jit_fn = _triton_complex_rope._complex_rope_kernel
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
    """Compile + run + time the kernel at num_warps 2/4/8 on sm_100."""
    from lethe.kernels.ops import _triton_complex_rope

    spec = QUICK_SHAPES[1] if quick else SWEEP_SHAPE
    args = _make_inputs(spec, torch.float32)
    trials = 10 if quick else 30
    sweep: dict[str, Any] = {"shape": spec.label(), "configs": {}}
    base: Tensor | None = None

    def _runner(warps_cfg: int) -> Callable[[], Tensor]:
        def run() -> Tensor:
            return _triton_complex_rope.launch_complex_scan_rope(*args, num_warps=warps_cfg)

        return run

    for warps in SWEEP_WARPS:
        entry: dict[str, Any] = {}
        run_cfg = _runner(warps)
        try:
            y = run_cfg()
            entry["compiles"] = True
            entry.update(_time(run_cfg, warmup=5, trials=trials))
            if base is None:
                base = y
            else:
                entry["scale_rel_vs_first_config"] = _parity_stats(y, base)["scale_rel"]
        except Exception:
            entry["compiles"] = False
            entry["error"] = traceback.format_exc(limit=5)
        sweep["configs"][f"num_warps={warps}"] = entry
    sweep["specializations"] = _specialization_table()
    sweep["official_mamba3_context"] = (
        "the sibling SISO Triton backward fails compile at all num_warps>=4 "
        "on sm_100 (TMEM 544 > 512); no open full-sequence rotary scan "
        "exists to sweep. evidence: docs/904_reproducer.md, "
        "results/repro_904_report.json"
    )
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "out" / "c4_bench.json",
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
        "op": "complex_scan_rope",
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
    report["resource_meta"] = triton_complex_rope_resource_meta()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
