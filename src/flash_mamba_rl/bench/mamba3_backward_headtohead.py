"""Claim #3: our backward vs the *crippled* official Mamba-3 backward on B200.

The THESIS make-or-break number. The c2/c6 benches compare ``ours_triton``
against ``selective_scan_fn`` — the **Mamba-1 SSD CUDA** backward, which is
healthy on Blackwell. That is the wrong baseline and is why we "looked 2.7x
slow." The honest head-to-head is against the **Mamba-3** backward that
#904 cripples: the path that launches ``mamba3_siso_bwd_kernel_dqkv``, which
on sm_100 loses every ``num_warps >= 4`` config to the TMEM-budget overflow
(544 > 512) and runs at the spilled ``num_warps=2`` survivor.

Two official comparators, both exercising that kernel:

- ``official_siso_combined_bwd`` — the matched one: the Mamba-3 SISO
  chunk-scan training function ``mamba3_siso_combined`` (its backward
  dispatches to the crippled kernel), differentiated, vs our
  ``backward_selective_scan`` at the same (B, L, d_model). Its signature
  (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, ...) is Mamba-3-specific,
  so rather than hand-build those tensors (which would mean guessing Mamba-3
  internals) we *capture* the exact tensors a real ``Mamba3`` module feeds
  the function via a monkeypatch hook, detach them into fresh leaves, and
  time the backward on those — the op runs at the model's real head config,
  recorded in the row. If capture fails (the module calls a different
  internal) it records a skip; the discovery dump
  (``scratch/mamba3_bwd_discover.py``) shows the real path.
- ``official_mamba3_module_bwd`` — the robust backstop: the full ``Mamba3``
  SISO module backward (always exercises the kernel, but includes in/out
  projections, so it is *not* shape-matched to our scan-only op — labelled,
  for sanity not for the headline).

Methodology mirrors the c2/c6 benches: the official forward runs once
outside the timed region and the timed backward reuses its forward-saved
activations (conservative against us — every timed call of ours re-stages
the forward in-kernel). Wall-clock is the headline; ``achieved_tflops`` /
``mfu`` (under the explicit SSM FLOP model below, applied identically to
both impls, so the MFU ratio equals the time ratio) is order-of-magnitude
context, not the claim.

    CUDA_VISIBLE_DEVICES=0 uv run python -m \
        flash_mamba_rl.bench.mamba3_backward_headtohead --out ~/out/headtohead.json
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

from flash_mamba_rl.kernels.ops import backward_selective_scan
from flash_mamba_rl.verifier.timing import benchmark

try:
    from mamba_ssm.modules.mamba3 import Mamba3

    _HAS_MAMBA3 = True
except Exception:  # pragma: no cover - box-only dependency
    _HAS_MAMBA3 = False

# SSM selective-scan FLOP model (forward), per (batch, len, d_model, d_state)
# element: discretized decay apply (1 mul) + input term dt*B*x (2 mul) +
# state add (1) + output C*h reduction (1 mul + 1 add) ~= 6 FLOPs; round to
# 8 for the dt/exp discretization tail. Backward redoes the forward sweep and
# carries an adjoint sweep -> ~2x. The constant is rough; it cancels in the
# ours/official MFU ratio, which equals the wall-clock ratio. Documented so
# the achieved-TFLOP/s numbers are reproducible, not load-bearing.
_SSM_FWD_FLOPS_PER_ELEM = 8
_SSM_BWD_FLOPS_PER_ELEM = 16

# Stated device peaks (TFLOP/s) for the MFU denominator; B200 SXM.
_PEAK_TFLOPS = {
    "float32": 1100.0,  # TF32 tensor-core class; the scan is not matmul-bound
    "bfloat16": 2250.0,  # dense bf16 tensor-core
}


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seq_len: int
    d_model: int
    n_state: int
    headdim: int

    @property
    def nheads(self) -> int:
        return self.d_model // self.headdim

    def label(self) -> str:
        return f"B{self.batch}xL{self.seq_len}xD{self.d_model}xN{self.n_state}xP{self.headdim}"

    def bwd_flops(self) -> int:
        return _SSM_BWD_FLOPS_PER_ELEM * self.batch * self.seq_len * self.d_model * self.n_state


# Matched to the c2 bench shapes; headdim=64, n_state=128 is the Mamba-3
# SISO default (d_state=128). d_model multiples of headdim.
SHAPES = [
    ShapeSpec(2, 512, 1024, 128, 64),  # small, every comparator runs
    ShapeSpec(8, 2048, 4096, 128, 64),  # training shape — the #904 regime
    ShapeSpec(8, 4096, 4096, 128, 64),
    ShapeSpec(2, 16384, 4096, 128, 64),  # long-sequence story
]
QUICK_SHAPES = [ShapeSpec(2, 512, 1024, 128, 64), ShapeSpec(4, 2048, 2048, 128, 64)]

DTYPES = [torch.float32, torch.bfloat16]


def _time(fn: Callable[[], Any], *, warmup: int, trials: int) -> dict[str, float]:
    result = benchmark(fn, (), warmup=warmup, trials=trials)
    return {
        "median_ms": result.median_ms,
        "std_ms": result.std_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "n_trials": float(result.n_trials),
    }


def _perf(timing: dict[str, float], spec: ShapeSpec, dtype: torch.dtype) -> dict[str, float]:
    ms = timing["median_ms"]
    tflops = spec.bwd_flops() / (ms * 1e-3) / 1e12
    peak = _PEAK_TFLOPS[str(dtype).removeprefix("torch.")]
    return {**timing, "achieved_tflops": tflops, "mfu": tflops / peak}


def _ours_scan_inputs(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0) -> tuple[Tensor, ...]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    b, length, d, n = spec.batch, spec.seq_len, spec.d_model, spec.n_state
    u = torch.randn(b, length, d, device=dev).to(dtype)
    delta = torch.randn(b, length, d, device=dev).to(dtype)
    a = (-torch.rand(d, n, device=dev)).to(dtype)
    b_proj = torch.randn(b, length, n, device=dev).to(dtype)
    c_proj = torch.randn(b, length, n, device=dev).to(dtype)
    d_skip = torch.randn(d, device=dev).to(dtype)
    dy = torch.randn(b, length, d, device=dev).to(dtype)
    return u, delta, a, b_proj, c_proj, d_skip, dy


def _capture_siso_args(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0):  # type: ignore[no-untyped-def]
    """Run one Mamba3 forward, capture the args it feeds mamba3_siso_combined.

    Monkeypatches the function in its defining module so the captured tensors
    are exactly what the real model passes — no Mamba-3-internal guessing.
    Returns (callable, captured_args_tuple, captured_kwargs).
    """
    import sys

    import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as siso_mod

    torch.manual_seed(seed)
    dev = torch.device("cuda")
    real_fn = siso_mod.mamba3_siso_combined
    captured: dict[str, Any] = {}

    def _spy(*args: Any, **kwargs: Any) -> Any:
        if "args" not in captured:
            captured["args"] = args
            captured["kwargs"] = kwargs
        return real_fn(*args, **kwargs)

    # The module may import the function by value (from ... import
    # mamba3_siso_combined), so the live reference lives in the caller's
    # namespace, not only the defining module. Patch every loaded module
    # that exposes the symbol bound to the real fn.
    patched: list[Any] = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "mamba3_siso_combined", None) is real_fn:
            mod.mamba3_siso_combined = _spy  # type: ignore[attr-defined]
            patched.append(mod)

    layer = Mamba3(d_model=spec.d_model).to(dev).to(dtype)
    x = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev, dtype=dtype)
    try:
        layer(x)
    finally:
        for mod in patched:
            mod.mamba3_siso_combined = real_fn
    if "args" not in captured:
        raise RuntimeError("Mamba3.forward did not call mamba3_siso_combined (see discovery dump)")
    return real_fn, captured["args"], captured["kwargs"]


def _official_siso_combined(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0):  # type: ignore[no-untyped-def]
    """Forward (graph retained) of mamba3_siso_combined on detached leaves.

    Returns (y, leaves, dy, meta) for repeated ``torch.autograd.grad`` timing.
    Leaves are the differentiable float captured tensors detached fresh, so
    the timed backward isolates the scan (the crippled kernel), exactly as
    our backward_selective_scan timing isolates the scan from any projection.
    """
    real_fn, args, kwargs = _capture_siso_args(spec, dtype, seed)

    def _leaf(t: Any) -> Any:
        if isinstance(t, Tensor) and t.is_floating_point():
            return t.detach().requires_grad_(True)
        return t.detach() if isinstance(t, Tensor) else t

    new_args = tuple(_leaf(a) for a in args)
    new_kwargs = {k: _leaf(v) for k, v in kwargs.items()}
    leaves = [a for a in new_args if isinstance(a, Tensor) and a.requires_grad]
    leaves += [v for v in new_kwargs.values() if isinstance(v, Tensor) and v.requires_grad]

    out = real_fn(*new_args, **new_kwargs)
    y = out[0] if isinstance(out, tuple) else out
    dy = torch.randn_like(y)
    captured_shapes = {
        f"arg{i}": tuple(a.shape) for i, a in enumerate(args) if isinstance(a, Tensor)
    }
    return y, leaves, dy, captured_shapes


def _official_mamba3_module(spec: ShapeSpec, dtype: torch.dtype, seed: int = 0):  # type: ignore[no-untyped-def]
    """Full Mamba3 SISO module forward, graph retained — robust backstop."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    layer = Mamba3(d_model=spec.d_model).to(dev).to(dtype)
    x = torch.randn(spec.batch, spec.seq_len, spec.d_model, device=dev, dtype=dtype)
    x = x.requires_grad_(True)
    leaves = [x, *layer.parameters()]
    y = layer(x)
    dy = torch.randn_like(y)
    return y, leaves, dy


def _run_shape(spec: ShapeSpec, dtype: torch.dtype, quick: bool) -> dict[str, Any]:
    trials = 20 if quick else 50
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "seq_len": spec.seq_len,
            "d_model": spec.d_model,
            "n_state": spec.n_state,
            "headdim": spec.headdim,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "bwd_flops": spec.bwd_flops(),
        "impls": {},
        "skipped": {},
    }
    impls: dict[str, Any] = row["impls"]
    skipped: dict[str, str] = row["skipped"]

    u, delta, a, b_proj, c_proj, d_skip, dy = _ours_scan_inputs(spec, dtype)
    impls["ours_triton"] = _perf(
        _time(
            lambda: backward_selective_scan(u, delta, a, b_proj, c_proj, d_skip, dy, chunk_size=64),
            warmup=10,
            trials=trials,
        ),
        spec,
        dtype,
    )

    if not _HAS_MAMBA3:
        skipped["official_siso_combined_bwd"] = "mamba_ssm Mamba3 not installed"
        skipped["official_mamba3_module_bwd"] = "mamba_ssm Mamba3 not installed"
        return _finalize(row, impls)

    try:
        impls["official_siso_combined_bwd"], captured_shapes = _bench_official_combined(
            spec, dtype, trials
        )
        row["official_captured_shapes"] = captured_shapes
        torch.cuda.empty_cache()
    except Exception:
        skipped["official_siso_combined_bwd"] = traceback.format_exc(limit=4)

    try:
        impls["official_mamba3_module_bwd"] = _bench_official_module(spec, dtype, trials)
        row["module_bwd_note"] = "projection-inclusive; not shape-matched to ours — sanity only"
        torch.cuda.empty_cache()
    except Exception:
        skipped["official_mamba3_module_bwd"] = traceback.format_exc(limit=4)

    return _finalize(row, impls)


def _bench_official_combined(
    spec: ShapeSpec, dtype: torch.dtype, trials: int
) -> tuple[dict[str, float], dict[str, Any]]:
    y, leaves, dy_o, captured_shapes = _official_siso_combined(spec, dtype)
    torch.autograd.grad(y, leaves, dy_o, retain_graph=True, allow_unused=True)
    timing = _time(
        lambda: torch.autograd.grad(y, leaves, dy_o, retain_graph=True, allow_unused=True),
        warmup=10,
        trials=trials,
    )
    return _perf(timing, spec, dtype), captured_shapes


def _bench_official_module(spec: ShapeSpec, dtype: torch.dtype, trials: int) -> dict[str, float]:
    ym, leaves_m, dym = _official_mamba3_module(spec, dtype)
    torch.autograd.grad(ym, leaves_m, dym, retain_graph=True)
    timing = _time(
        lambda: torch.autograd.grad(ym, leaves_m, dym, retain_graph=True),
        warmup=5,
        trials=max(10, trials // 2),
    )
    return _perf(timing, spec, dtype)


def _finalize(row: dict[str, Any], impls: dict[str, Any]) -> dict[str, Any]:
    ours = impls.get("ours_triton", {}).get("median_ms")
    if ours:
        row["ours_speedup_vs"] = {
            name: timing["median_ms"] / ours
            for name, timing in impls.items()
            if name != "ours_triton"
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.home() / "out" / "headtohead.json")
    parser.add_argument("--quick", action="store_true")
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
    if _HAS_MAMBA3:
        import mamba_ssm

        mamba_version = getattr(mamba_ssm, "__version__", "unknown")

    report: dict[str, Any] = {
        "claim": "ours backward vs crippled official Mamba-3 backward (mamba3_siso_bwd_kernel_dqkv)",
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
            "triton": triton_version,
            "mamba_ssm": mamba_version,
            "python": platform.python_version(),
        },
        "flop_model": {
            "fwd_flops_per_elem": _SSM_FWD_FLOPS_PER_ELEM,
            "bwd_flops_per_elem": _SSM_BWD_FLOPS_PER_ELEM,
            "peak_tflops": _PEAK_TFLOPS,
            "note": "rough; cancels in the ours/official MFU ratio (= wall-clock ratio)",
        },
        "runs": [],
    }

    shapes = QUICK_SHAPES if cli.quick else SHAPES
    for spec in shapes:
        for dtype in DTYPES:
            print(f"[headtohead] {spec.label()} {dtype} ...", flush=True)
            report["runs"].append(_run_shape(spec, dtype, cli.quick))
            torch.cuda.empty_cache()

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(report, indent=2))
    print(f"[headtohead] wrote {cli.out}")


if __name__ == "__main__":
    main()
