"""C5 benchmark: the fused block vs unfused compositions of the same math."""

from __future__ import annotations

import argparse
import json
import math
import platform
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lethe.kernels.ops import (
    forward_chunked_scan,
    fused_block_forward,
    triton_fused_block_resource_meta,
)
from lethe.kernels.references import reference_fused_block_forward

from .c2_backward_selective_scan import _parity_stats, _time

FusedArgs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

# The Python-loop oracle steps t-by-t in eager torch; cap its length.
_REFERENCE_LOOP_MAX_SEQ = 256

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    _HAS_MAMBA = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_MAMBA = False

try:
    from causal_conv1d import causal_conv1d_fn

    _HAS_CAUSAL_CONV = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_CAUSAL_CONV = False


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    l_out: int
    d_model: int
    n_state: int
    conv_k: int

    def label(self) -> str:
        return f"B{self.batch}xL{self.l_out}xD{self.d_model}xN{self.n_state}xK{self.conv_k}"


# Training sizing matches C1/C2 (d_inner=4096) with Mamba-3's d_state 128, d_conv=4.
SHAPES = [
    ShapeSpec(2, 256, 256, 16, 4),  # oracle shape: every comparator runs
    ShapeSpec(8, 2048, 4096, 128, 4),  # training shape
    ShapeSpec(8, 4096, 4096, 128, 4),
    ShapeSpec(2, 16384, 4096, 128, 4),  # long-sequence regime
]
QUICK_SHAPES = [ShapeSpec(2, 256, 256, 16, 4), ShapeSpec(4, 2048, 1024, 128, 4)]

DTYPES = [torch.float32, torch.bfloat16]

SWEEP_SHAPE = ShapeSpec(8, 2048, 4096, 128, 4)
SWEEP_WARPS = (2, 4, 8)


def _make_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> FusedArgs:
    """x arrives pre-padded with K-1 zeros, matching the harness adapter."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    b, length, d, n, k = spec.batch, spec.l_out, spec.d_model, spec.n_state, spec.conv_k
    x_core = torch.randn(b, length, d, device=dev).to(dtype)
    x = F.pad(x_core, (0, 0, k - 1, 0))
    conv_w = (torch.randn(d, 1, k, device=dev) / math.sqrt(k)).to(dtype)
    conv_b = (0.5 * torch.randn(d, device=dev)).to(dtype)
    delta = torch.randn(b, length, d, device=dev).to(dtype)
    a = (-torch.rand(d, n, device=dev)).to(dtype)
    b_proj = torch.randn(b, length, n, device=dev).to(dtype)
    c_proj = torch.randn(b, length, n, device=dev).to(dtype)
    d_skip = torch.randn(d, device=dev).to(dtype)
    norm_w = (1.0 + 0.25 * torch.randn(d, device=dev)).to(dtype)
    return x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w


def _rms_norm(y_scan: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return y_scan / rms * weight


def _composed_ours_scan(args: FusedArgs) -> Tensor:
    x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w = args
    conv = F.conv1d(x.transpose(1, 2), conv_w, conv_b, groups=x.shape[-1]).transpose(1, 2)
    z = F.silu(conv)
    y_scan = forward_chunked_scan(z.contiguous(), delta, a, b_proj, c_proj, d_skip, chunk_size=64)
    return _rms_norm(y_scan, norm_w)


def _composed_official(args: FusedArgs, conv_k: int) -> Tensor:
    x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w = args
    if _HAS_CAUSAL_CONV:
        # causal_conv1d pads internally; feed it the unpadded core to match our explicit pre-pad.
        x_core_t = x[:, conv_k - 1 :, :].transpose(1, 2).contiguous()
        z_t = causal_conv1d_fn(x_core_t, conv_w[:, 0, :], conv_b, activation="silu")
    else:
        conv = F.conv1d(x.transpose(1, 2), conv_w, conv_b, groups=x.shape[-1])
        z_t = F.silu(conv)
    y_scan = selective_scan_fn(
        z_t.contiguous(),
        delta.transpose(1, 2).contiguous(),
        a,
        b_proj.transpose(1, 2).unsqueeze(1).contiguous(),
        c_proj.transpose(1, 2).unsqueeze(1).contiguous(),
        d_skip,
        delta_softplus=True,
    ).transpose(1, 2)
    return _rms_norm(y_scan, norm_w)


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    args = _make_inputs(spec, dtype)
    trials = 20 if quick else 50
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "l_out": spec.l_out,
            "d_model": spec.d_model,
            "n_state": spec.n_state,
            "conv_k": spec.conv_k,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "impls": {},
        "skipped": {
            "official_fused_block": (
                "mamba_inner_fn fuses a different composition (z-gating, "
                "fused in/out projections, no post-scan RMSNorm); the "
                "composed official path below is the honest anchor"
            )
        },
        "parity": {},
    }
    impls: dict[str, Any] = row["impls"]

    y_ours = fused_block_forward(*args, conv_kernel_size=spec.conv_k, chunk_size=64)
    impls["ours_triton"] = _time(
        lambda: fused_block_forward(*args, conv_kernel_size=spec.conv_k, chunk_size=64),
        warmup=10,
        trials=trials,
    )

    y_comp = _composed_ours_scan(args)
    row["parity"]["ours_vs_composed_ours_scan"] = _parity_stats(y_ours, y_comp)
    impls["composed_ours_scan"] = _time(lambda: _composed_ours_scan(args), warmup=10, trials=trials)

    if _HAS_MAMBA and dtype == torch.float32:
        try:
            y_off = _composed_official(args, spec.conv_k)
            row["parity"]["ours_vs_composed_official"] = _parity_stats(y_ours, y_off)
            impls["composed_official"] = _time(
                lambda: _composed_official(args, spec.conv_k), warmup=10, trials=trials
            )
            row["causal_conv1d_installed"] = _HAS_CAUSAL_CONV
        except Exception:
            row["skipped"]["composed_official"] = traceback.format_exc(limit=3)
    elif not _HAS_MAMBA:
        row["skipped"]["composed_official"] = "mamba_ssm not installed"
    else:
        row["skipped"]["composed_official"] = "official scan anchored at fp32 only"

    if spec.l_out <= _REFERENCE_LOOP_MAX_SEQ and dtype == torch.float32:
        y_ref = reference_fused_block_forward(*args, conv_kernel_size=spec.conv_k, chunk_size=64)
        row["parity"]["ours_vs_reference"] = _parity_stats(y_ours, y_ref)
        impls["reference_loop"] = _time(
            lambda: reference_fused_block_forward(
                *args, conv_kernel_size=spec.conv_k, chunk_size=64
            ),
            warmup=1,
            trials=3,
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
    """Per-compiled-specialisation resources from both kernels' caches."""
    from lethe.kernels.ops import _triton_fused_block

    rows: list[dict[str, Any]] = []
    for jit_fn in (_triton_fused_block._conv_scan_kernel, _triton_fused_block._rmsnorm_kernel):
        compiled: list[Any] = []
        caches = getattr(jit_fn, "device_caches", None)
        if isinstance(caches, dict):
            for entry in caches.values():
                cache_dict = entry[0] if isinstance(entry, tuple) else entry
                if isinstance(cache_dict, dict):
                    compiled.extend(cache_dict.values())
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
    """Compile + run + time the fused path at num_warps 2/4/8 on sm_100."""
    from lethe.kernels.ops import _triton_fused_block

    spec = QUICK_SHAPES[1] if quick else SWEEP_SHAPE
    args = _make_inputs(spec, torch.float32)
    trials = 10 if quick else 30
    sweep: dict[str, Any] = {"shape": spec.label(), "configs": {}}
    base: Tensor | None = None

    def _runner(warps_cfg: int) -> Callable[[], Tensor]:
        def run() -> Tensor:
            return _triton_fused_block.launch_fused_block_forward(*args, 1e-5, num_warps=warps_cfg)

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
        "on sm_100 (TMEM 544 > 512); the official fused-block training path "
        "is a compiled TileLang artifact. evidence: docs/904_reproducer.md, "
        "results/repro_904_report.json"
    )
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "out" / "c5_bench.json",
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
        "op": "fused_block_forward",
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "triton": triton_version,
            "python": platform.python_version(),
            "mamba_ssm": _HAS_MAMBA,
            "causal_conv1d": _HAS_CAUSAL_CONV,
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
    report["resource_meta"] = triton_fused_block_resource_meta()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
