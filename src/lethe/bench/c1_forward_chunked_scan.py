"""C1 benchmark: our Triton scan vs official mamba_ssm vs torch eager."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lethe.kernels.ops import forward_chunked_scan, triton_scan_resource_meta
from lethe.kernels.references import reference_forward_chunked_scan
from lethe.verifier.timing import benchmark

from .c2_backward_selective_scan import _parity_stats

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        selective_scan_ref,
    )

    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False

# Eager comparators materialise [B, D, L, N] intermediates; gate on this.
_EAGER_BYTES_CAP = 6 * 1024**3
# The Python-loop oracle launches ~5 kernels per timestep; cap the loop.
_REFERENCE_LOOP_MAX_SEQ = 1024


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
    ShapeSpec(2, 16384, 4096, 16),  # long-sequence regime
]
QUICK_SHAPES = [ShapeSpec(2, 512, 1024, 16), ShapeSpec(4, 2048, 2048, 16)]

DTYPES = [torch.float32, torch.bfloat16]


def _make_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> tuple[Tensor, ...]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    u = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev).to(dtype)
    delta = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev).to(dtype)
    a = -torch.rand(spec.d_model, spec.n_state, device=dev, dtype=torch.float32)
    b_proj = torch.randn(spec.batch, spec.seq_len, spec.n_state, device=dev).to(dtype)
    c_proj = torch.randn(spec.batch, spec.seq_len, spec.n_state, device=dev).to(dtype)
    d_skip = torch.randn(spec.d_model, device=dev, dtype=torch.float32)
    return u, delta, a.to(dtype), b_proj, c_proj, d_skip.to(dtype)


def _time(fn: Callable[[], Tensor], *, warmup: int, trials: int) -> dict[str, float]:
    result = benchmark(fn, (), warmup=warmup, trials=trials)
    return {
        "median_ms": result.median_ms,
        "std_ms": result.std_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "n_trials": float(result.n_trials),
    }


def _official_layout(args: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Pre-transpose to the official [B, D, L] / [B, 1, N, L] layout."""
    u, delta, a, b_proj, c_proj, d_skip = args
    return (
        u.transpose(1, 2).contiguous(),
        delta.transpose(1, 2).contiguous(),
        a.float(),
        b_proj.transpose(1, 2).unsqueeze(1).contiguous(),
        c_proj.transpose(1, 2).unsqueeze(1).contiguous(),
        d_skip.float(),
    )


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    args = _make_inputs(spec, dtype)
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

    y_ours = forward_chunked_scan(*args, chunk_size=64)
    impls["ours_triton"] = _time(
        lambda: forward_chunked_scan(*args, chunk_size=64), warmup=10, trials=trials
    )

    if _HAS_MAMBA:
        official_args = _official_layout(args)
        y_official = selective_scan_fn(*official_args, delta_softplus=True)
        parity["ours_vs_official"] = _parity_stats(y_ours, y_official.transpose(1, 2))
        impls["official_cuda"] = _time(
            lambda: selective_scan_fn(*official_args, delta_softplus=True),
            warmup=10,
            trials=trials,
        )

        if spec.eager_bytes() <= _EAGER_BYTES_CAP and dtype == torch.float32:
            impls["official_eager"] = _time(
                lambda: selective_scan_ref(*official_args, delta_softplus=True),
                warmup=2,
                trials=5,
            )
        else:
            skipped["official_eager"] = "memory cap or non-fp32"
    else:
        skipped["official_cuda"] = "mamba_ssm not installed"

    if spec.seq_len <= _REFERENCE_LOOP_MAX_SEQ and dtype == torch.float32:
        y_ref = reference_forward_chunked_scan(*args, chunk_size=64)
        parity["ours_vs_reference"] = _parity_stats(y_ours, y_ref)
        impls["reference_loop"] = _time(
            lambda: reference_forward_chunked_scan(*args, chunk_size=64), warmup=1, trials=3
        )
    else:
        skipped["reference_loop"] = "sequence too long for the Python-loop oracle"

    ours_ms = impls["ours_triton"]["median_ms"]
    row["speedups_vs_ours"] = {
        name: timing["median_ms"] / ours_ms
        for name, timing in impls.items()
        if name != "ours_triton"
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "out" / "c1_bench.json",
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
        "op": "forward_chunked_scan",
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

    report["resource_meta"] = triton_scan_resource_meta()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
