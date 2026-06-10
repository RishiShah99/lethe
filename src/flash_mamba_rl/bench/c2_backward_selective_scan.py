"""C2 benchmark: our Triton backward scan vs official mamba_ssm vs eager.

Comparators per (shape, dtype), each skipped with a recorded reason when
infeasible:

- ``ours_triton``       — this repo's hand-written backward kernel (C2)
- ``official_cuda_bwd`` — VJP through ``mamba_ssm`` ``selective_scan_fn``
  (the Mamba-1 CUDA backward — healthy on Blackwell, unlike the Mamba-3
  Triton backward this op replaces; timed via ``torch.autograd.grad`` with
  ``retain_graph``, so the number includes autograd dispatch like ours
  includes launcher reductions)
- ``official_eager_bwd`` — VJP through ``selective_scan_ref`` (vectorised
  torch eager; materialises [B, D, L, N] *and* its graph, so memory-gated)
- ``reference_loop_bwd`` — our Python-loop oracle's autograd (small shapes)

Plus the #904 contrast artifact: ``num_warps_sweep`` compiles and times our
kernel at num_warps 2/4/8 and records per-specialisation ptxas resources
(regs/spills/smem). The official Mamba-3 Triton backward fails to compile
at every num_warps >= 4 config on sm_100 (TMEM budget, see
docs/904_reproducer.md and results/repro_904_report.json); ours has no
tl.dot for the TMEM-promotion pass to touch, so the sweep succeeding *is*
the claim, recorded with the evidence.

Usage (on the box): ``uv run python -m
flash_mamba_rl.bench.c2_backward_selective_scan --out ~/out/c2_bench.json``
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

from flash_mamba_rl.kernels.ops import backward_selective_scan, triton_bwd_scan_resource_meta
from flash_mamba_rl.kernels.references import reference_backward_selective_scan
from flash_mamba_rl.verifier.op_harness import BWD_GRAD_FIELDS
from flash_mamba_rl.verifier.timing import benchmark

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_ref,
    )

    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False

# Fixed-arity alias so star-unpacking binds precisely at call sites.
ScanArgs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

# Eager comparators materialise [B, D, L, N] intermediates plus their
# autograd graph; gate on the base intermediate size.
_EAGER_BYTES_CAP = 4 * 1024**3
# The Python-loop oracle launches ~5 kernels per timestep forward and again
# backward; cap the loop length.
_REFERENCE_LOOP_MAX_SEQ = 512


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seq_len: int
    d_model: int
    n_state: int

    def label(self) -> str:
        return f"B{self.batch}xL{self.seq_len}xD{self.d_model}xN{self.n_state}"

    def eager_bytes(self) -> int:
        return self.batch * self.seq_len * self.d_model * self.n_state * 4


SHAPES = [
    ShapeSpec(2, 512, 1024, 16),  # oracle shape: every comparator runs
    ShapeSpec(8, 2048, 4096, 16),  # Mamba-1.4B-ish layer at training length
    ShapeSpec(8, 4096, 4096, 16),
    ShapeSpec(4, 8192, 4096, 16),
    ShapeSpec(2, 16384, 4096, 16),  # long-sequence story
]
QUICK_SHAPES = [ShapeSpec(2, 512, 1024, 16), ShapeSpec(4, 2048, 2048, 16)]

DTYPES = [torch.float32, torch.bfloat16]

# The num_warps sweep shape: training-sized, matches the repro's regime.
SWEEP_SHAPE = ShapeSpec(8, 2048, 4096, 16)
SWEEP_WARPS = (2, 4, 8)


def _make_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> tuple[ScanArgs, Tensor]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    u = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev).to(dtype)
    delta = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev).to(dtype)
    a = -torch.rand(spec.d_model, spec.n_state, device=dev, dtype=torch.float32)
    b_proj = torch.randn(spec.batch, spec.seq_len, spec.n_state, device=dev).to(dtype)
    c_proj = torch.randn(spec.batch, spec.seq_len, spec.n_state, device=dev).to(dtype)
    d_skip = torch.randn(spec.d_model, device=dev, dtype=torch.float32)
    dy = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev).to(dtype)
    return (u, delta, a.to(dtype), b_proj, c_proj, d_skip.to(dtype)), dy


def _time(fn: Callable[[], Any], *, warmup: int, trials: int) -> dict[str, float]:
    result = benchmark(fn, (), warmup=warmup, trials=trials)
    return {
        "median_ms": result.median_ms,
        "std_ms": result.std_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "n_trials": float(result.n_trials),
    }


def _official_leaves(args: ScanArgs) -> tuple[Tensor, ...]:
    """Official [B, D, L] / [B, 1, N, L] layout, as autograd leaves.

    A and D upcast from the *rounded* low-precision values so both kernels
    consume identical operand bits (the official kernel requires fp32
    weights); parity numbers then measure implementation differences, not
    operand-rounding differences.
    """
    u, delta, a, b_proj, c_proj, d_skip = args
    return (
        u.transpose(1, 2).contiguous().requires_grad_(True),
        delta.transpose(1, 2).contiguous().requires_grad_(True),
        a.float().detach().requires_grad_(True),
        b_proj.transpose(1, 2).unsqueeze(1).contiguous().requires_grad_(True),
        c_proj.transpose(1, 2).unsqueeze(1).contiguous().requires_grad_(True),
        d_skip.float().detach().requires_grad_(True),
    )


def _parity_stats(got: Tensor, want: Tensor) -> dict[str, float]:
    """Absolute max error plus the comparand's scale.

    grad_A magnitudes grow superlinearly with L in the near-integrator
    regime, so a bare max_err reads as divergence when it is reorder noise
    riding a large output — scale_rel (max_err / |want|_inf) is the
    comparable number across shapes.
    """
    diff = (got.float() - want.float()).abs()
    ref_inf = float(want.float().abs().max().item())
    max_err = float(diff.max().item())
    return {
        "max_err": max_err,
        "ref_inf": ref_inf,
        "scale_rel": max_err / max(1.0, ref_inf),
    }


def _from_official_grads(grads: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    gu, gdelta, ga, gb, gc, gd = grads
    return (
        gu.transpose(1, 2),
        gdelta.transpose(1, 2),
        ga,
        gb.squeeze(1).transpose(1, 2),
        gc.squeeze(1).transpose(1, 2),
        gd,
    )


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    args, dy = _make_inputs(spec, dtype)
    trials = 20 if quick else 50
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "seq_len": spec.seq_len,
            "d_model": spec.d_model,
            "n_state": spec.n_state,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "impls": {},
        "skipped": {},
        "parity": {},
    }
    impls: dict[str, Any] = row["impls"]
    skipped: dict[str, str] = row["skipped"]
    parity: dict[str, Any] = row["parity"]

    grads_ours = backward_selective_scan(*args, dy, chunk_size=64)
    impls["ours_triton"] = _time(
        lambda: backward_selective_scan(*args, dy, chunk_size=64), warmup=10, trials=trials
    )

    if _HAS_MAMBA:
        leaves = _official_leaves(args)
        y = selective_scan_fn(*leaves, delta_softplus=True)
        dy_o = dy.transpose(1, 2).contiguous()
        if dtype == torch.float32:
            grads_official = _from_official_grads(
                torch.autograd.grad(y, leaves, dy_o, retain_graph=True)
            )
            parity["ours_vs_official"] = {
                field: _parity_stats(got, want)
                for field, got, want in zip(
                    BWD_GRAD_FIELDS, grads_ours, grads_official, strict=True
                )
            }
        impls["official_cuda_bwd"] = _time(
            lambda: torch.autograd.grad(y, leaves, dy_o, retain_graph=True),
            warmup=10,
            trials=trials,
        )
        del y

        if spec.eager_bytes() <= _EAGER_BYTES_CAP and dtype == torch.float32:
            leaves_ref = _official_leaves(args)
            y_ref = selective_scan_ref(*leaves_ref, delta_softplus=True)
            impls["official_eager_bwd"] = _time(
                lambda: torch.autograd.grad(y_ref, leaves_ref, dy_o, retain_graph=True),
                warmup=2,
                trials=5,
            )
            del y_ref
        else:
            skipped["official_eager_bwd"] = "memory cap or non-fp32"
    else:
        skipped["official_cuda_bwd"] = "mamba_ssm not installed"

    if spec.seq_len <= _REFERENCE_LOOP_MAX_SEQ and dtype == torch.float32:
        grads_ref = reference_backward_selective_scan(*args, dy, chunk_size=64)
        parity["ours_vs_reference"] = {
            field: _parity_stats(got, want)
            for field, got, want in zip(BWD_GRAD_FIELDS, grads_ours, grads_ref, strict=True)
        }
        impls["reference_loop_bwd"] = _time(
            lambda: reference_backward_selective_scan(*args, dy, chunk_size=64),
            warmup=1,
            trials=3,
        )
    else:
        skipped["reference_loop_bwd"] = "sequence too long for the Python-loop oracle"

    ours_ms = impls["ours_triton"]["median_ms"]
    row["speedups_vs_ours"] = {
        name: timing["median_ms"] / ours_ms
        for name, timing in impls.items()
        if name != "ours_triton"
    }
    return row


def _specialization_table() -> list[dict[str, Any]]:
    """Per-compiled-specialisation resources from the kernel cache.

    Unlike the max-envelope ``resource_meta()``, this keeps one row per
    cached CompiledKernel with its launch config — the per-num_warps
    evidence the #904 contrast needs. Attribute layout is triton-version
    dependent; missing fields record as None rather than guessing.
    """
    from flash_mamba_rl.kernels.ops import _triton_bwd_scan

    jit_fn = _triton_bwd_scan._bwd_scan_kernel
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
    """Compile + run + time our kernel at num_warps 2/4/8.

    The official Mamba-3 Triton backward dies at compile for every
    num_warps >= 4 config on sm_100 (`Required: 544, Hardware limit: 512`);
    each successful entry here is the direct counterexample for our
    formulation, with per-config ptxas resources alongside.
    """
    from flash_mamba_rl.kernels.ops import _triton_bwd_scan

    spec = QUICK_SHAPES[1] if quick else SWEEP_SHAPE
    args, dy = _make_inputs(spec, torch.float32)
    trials = 10 if quick else 30
    sweep: dict[str, Any] = {"shape": spec.label(), "configs": {}}
    base: tuple[Tensor, ...] | None = None

    def _runner(warps_cfg: int) -> Callable[[], tuple[Tensor, ...]]:
        def run() -> tuple[Tensor, ...]:
            return _triton_bwd_scan.launch_backward_scan(*args, dy, num_warps=warps_cfg)

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
    sweep["official_mamba3_triton_bwd"] = (
        "fails compile at all num_warps>=4 on sm_100 (TMEM 544 > 512); "
        "evidence: docs/904_reproducer.md, results/repro_904_report.json"
    )
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "out" / "c2_bench.json",
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

    mamba_version: str | None = None
    if _HAS_MAMBA:
        import mamba_ssm

        mamba_version = getattr(mamba_ssm, "__version__", "unknown")

    report: dict[str, Any] = {
        "op": "backward_selective_scan",
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "triton": triton_version,
            "mamba_ssm": mamba_version,
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
    report["resource_meta"] = triton_bwd_scan_resource_meta()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
