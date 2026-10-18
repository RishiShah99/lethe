"""PRC-03 precision audit: does a KernelBench-convention candidate compute at fp32 precision?

Black-box. The candidate's error against an fp64 run of the reference is compared with the
error of the strict fp32 reference run on inputs and parameters rounded to a 10-bit mantissa
(what fp16 and TF32 do). No fixed tolerance works: fp32 accumulation error grows with reduction
length while half-precision input rounding does not, so the yardstick is measured per problem.

RMS errors over elements finite in every run, per output and draw:
  flag          e_cand >= e_tf32 / 2, or non-finite where fp64 and fp32 are finite
  inconclusive  e_tf32 < 4 * e_fp32, or the fp64 reference ran an op below fp64
  pass          otherwise
Unit verdict = worst over outputs and draws. Backend flags are strict fp32 for the run, so TF32
counts only when the candidate's own source enables it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

from lethe.verifier.audit_harness import AUDIT_SEED, _exec_source, _trunc

FLAG_VS_TF32_BITS = -1.0
MIN_SEPARATION_BITS = 2.0
DEFAULT_DRAWS = 3

_EPS32 = torch.finfo(torch.float32).eps
_SEVERITY = {"pass": 0, "inconclusive": 1, "flag": 2}


def round_mantissa(t: Tensor, bits: int) -> Tensor:
    """Round an fp32 tensor to `bits` explicit mantissa bits (nearest, ties away), fp32 range."""
    drop = 23 - bits
    i = t.contiguous().view(torch.int32)
    return ((i + (1 << (drop - 1))) & ~((1 << drop) - 1)).view(torch.float32)


def _backend_flags() -> dict[str, Any]:
    return {
        "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _set_backend_flags(flags: dict[str, Any]) -> None:
    torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = flags["matmul.allow_tf32"]
    torch.backends.cudnn.allow_tf32 = flags["cudnn.allow_tf32"]


_STRICT = {
    "matmul.allow_tf32": False,
    "cudnn.allow_tf32": False,
    "float32_matmul_precision": "highest",
}


@contextlib.contextmanager
def _flags(flags: dict[str, Any]) -> Iterator[None]:
    saved = _backend_flags()
    _set_backend_flags(flags)
    try:
        yield
    finally:
        _set_backend_flags(saved)


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    saved = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(saved)


class _LowPrecisionOps(TorchDispatchMode):  # type: ignore[no-untyped-call]
    """Counts aten ops that return a real or complex tensor below fp64."""

    def __init__(self) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.ops: dict[str, int] = {}

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        out = func(*args, **(kwargs or {}))
        for o in _tensor_leaves(out):
            if (o.is_floating_point() or o.is_complex()) and o.dtype not in (
                torch.float64,
                torch.complex128,
            ):
                name = str(func.overloadpacket.__name__)
                self.ops[name] = self.ops.get(name, 0) + 1
                break
        return out


def _tensor_leaves(out: Any) -> list[Tensor]:
    if isinstance(out, Tensor):
        return [out]
    if isinstance(out, tuple | list):
        return [t for item in out for t in _tensor_leaves(item)]
    if isinstance(out, dict):
        return [t for item in out.values() for t in _tensor_leaves(item)]
    return []


def _float_outputs(out: Any) -> list[Tensor]:
    return [t.detach() for t in _tensor_leaves(out) if t.is_floating_point()]


def _map_float(model: nn.Module, fn: Callable[[Tensor], Tensor]) -> nn.Module:
    """Apply `fn` to every floating parameter, buffer and plain tensor attribute, in place."""
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.data = fn(p.data)
        for m in model.modules():
            for name, b in list(m._buffers.items()):
                if b is not None and b.is_floating_point():
                    m._buffers[name] = fn(b)
            for name, v in list(vars(m).items()):
                if isinstance(v, Tensor) and v.is_floating_point():
                    m.__dict__[name] = fn(v)
    return model


def _map_inputs(inputs: list[Any], fn: Callable[[Tensor], Tensor]) -> list[Any]:
    return [fn(a) if isinstance(a, Tensor) and a.is_floating_point() else a for a in inputs]


def _rounder(bits: int) -> Callable[[Tensor], Tensor]:
    def fn(t: Tensor) -> Tensor:
        return round_mantissa(t, bits) if t.dtype == torch.float32 else t

    return fn


def _to_double(t: Tensor) -> Tensor:
    return t.double()


def _clone(t: Tensor) -> Tensor:
    return t.clone()


def _rms(t: Tensor) -> float:
    return float(t.pow(2).mean().sqrt().item()) if t.numel() else 0.0


def _log2_ratio(a: float, b: float) -> float:
    if a == 0.0:
        return -math.inf
    return math.inf if b == 0.0 else math.log2(a / b)


def _finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def _compare(
    cand: Tensor, ref64: Tensor, ref32: Tensor, ref_tf32: Tensor, ref_bf16: Tensor
) -> dict[str, Any]:
    dev = ref64.device
    c, r32, rt, rb = (t.to(dev, torch.float64) for t in (cand, ref32, ref_tf32, ref_bf16))
    finite = torch.isfinite(ref64) & torch.isfinite(r32)
    nonfinite_mismatch = int((finite & ~torch.isfinite(c)).sum().item())
    mask = finite & torch.isfinite(c) & torch.isfinite(rt) & torch.isfinite(rb)
    r64 = ref64[mask]
    rms64 = _rms(r64)
    e_cand = _rms(c[mask] - r64)
    e_fp32 = max(_rms(r32[mask] - r64), _EPS32 * rms64)
    e_tf32 = _rms(rt[mask] - r64)
    e_bf16 = _rms(rb[mask] - r64)
    vs_tf32 = _log2_ratio(e_cand, e_tf32)
    separation = _log2_ratio(e_tf32, e_fp32)
    if nonfinite_mismatch:
        verdict = "flag"
    elif e_fp32 == 0.0 or separation < MIN_SEPARATION_BITS:
        verdict = "inconclusive"
    elif vs_tf32 >= FLAG_VS_TF32_BITS:
        verdict = "flag"
    else:
        verdict = "pass"
    return {
        "shape": list(cand.shape),
        "cand_dtype": str(cand.dtype).removeprefix("torch."),
        "n_compared": int(mask.sum().item()),
        "nonfinite_mismatch": nonfinite_mismatch,
        "rms_ref64": rms64,
        "e_cand": e_cand,
        "e_fp32": e_fp32,
        "e_tf32": e_tf32,
        "e_bf16": e_bf16,
        "bits_lost": _finite_or_none(_log2_ratio(e_cand, e_fp32)),
        "vs_tf32": _finite_or_none(vs_tf32),
        "vs_bf16": _finite_or_none(_log2_ratio(e_cand, e_bf16)),
        "separation": _finite_or_none(separation),
        "verdict": verdict,
    }


def _max_or_none(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return max(vals) if vals else None


def precision_worker(ref_source: str, cand_source: str, config: dict[str, Any]) -> dict[str, Any]:
    """Measure one (reference, candidate) KernelBench source pair; returns a JSON-safe dict.

    config: device (default "cpu"), draws (get_inputs() draws, seeds AUDIT_SEED + i, default 3),
    train (module mode for all models, default False), no_grad (candidate grad mode, default
    False), strict_backend (strict fp32 backend flags before the sources load, default True).
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    entry_flags = _backend_flags()
    try:
        if config.get("strict_backend", True):
            _set_backend_flags(_STRICT)
        return _precision_worker_body(ref_source, cand_source, config)
    finally:
        _set_backend_flags(entry_flags)
        sys.stdout = real_stdout


def _precision_worker_body(
    ref_source: str, cand_source: str, config: dict[str, Any]
) -> dict[str, Any]:
    dev = torch.device(str(config.get("device", "cpu")))
    n_draws = max(1, int(config.get("draws", DEFAULT_DRAWS)))
    train = bool(config.get("train", False))
    no_grad = bool(config.get("no_grad", False))
    meta: dict[str, Any] = {
        "config": {
            "device": str(dev),
            "draws": n_draws,
            "train": train,
            "no_grad": no_grad,
            "strict_backend": bool(config.get("strict_backend", True)),
        }
    }

    def build(ctor: Callable[..., nn.Module], init_args: list[Any]) -> nn.Module:
        torch.manual_seed(AUDIT_SEED)
        return ctor(*init_args).to(dev).train(train)

    try:
        ref_ns = _exec_source(ref_source, "Model")
        get_inputs = ref_ns["get_inputs"]
        get_init = ref_ns.get("get_init_inputs")
        torch.manual_seed(AUDIT_SEED)
        init_args = list(get_init() or []) if get_init is not None else []
        with _flags(_STRICT):
            ref32 = build(ref_ns["Model"], init_args)
            ref_tf32 = _map_float(build(ref_ns["Model"], init_args), _rounder(10))
            ref_bf16 = _map_float(build(ref_ns["Model"], init_args), _rounder(7))
    except Exception as exc:
        return {"status": "ref_broken", "error": f"{type(exc).__name__}: {_trunc(exc)}", **meta}

    def draw(i: int) -> list[Any]:
        torch.manual_seed(AUDIT_SEED + i)
        return [a.to(dev) if isinstance(a, Tensor) else a for a in get_inputs()]

    if not any(isinstance(a, Tensor) and a.is_floating_point() for a in draw(0)):
        return {"status": "not_auditable", "error": "no floating-point tensor input", **meta}

    def run_ref(model: nn.Module, inputs: list[Any]) -> list[Tensor]:
        with _flags(_STRICT), torch.no_grad():
            return _float_outputs(model(*_map_inputs(inputs, _clone)))

    def run_ref64(model: nn.Module, inputs: list[Any]) -> tuple[list[Tensor], dict[str, int]]:
        log = _LowPrecisionOps()
        with _default_dtype(torch.float64), torch.no_grad(), log:
            out = _float_outputs(model(*_map_inputs(inputs, _to_double)))
        return out, log.ops

    ref64_dev = dev
    ref64 = _map_float(build(ref_ns["Model"], init_args), _to_double)
    try:
        run_ref64(ref64, draw(0))
    except Exception as exc:
        if dev.type == "cpu":
            return {
                "status": "ref64_failed",
                "error": f"{type(exc).__name__}: {_trunc(exc)}",
                **meta,
            }
        ref64_dev = torch.device("cpu")
        ref64 = _map_float(ref64.to(ref64_dev), _to_double)
        try:
            run_ref64(ref64, [a.cpu() if isinstance(a, Tensor) else a for a in draw(0)])
        except Exception as exc_cpu:
            return {
                "status": "ref64_failed",
                "error": f"{type(exc_cpu).__name__}: {_trunc(exc_cpu)}",
                "error_on_device": f"{type(exc).__name__}: {_trunc(exc)}",
                **meta,
            }

    try:
        cand_ns = _exec_source(cand_source, "ModelNew")
        meta["backend_flags_after_cand_import"] = _backend_flags()
        cand = build(cand_ns["ModelNew"], init_args)
    except Exception as exc:
        return {"status": "cand_load_fail", "error": f"{type(exc).__name__}: {_trunc(exc)}", **meta}

    draws: list[dict[str, Any]] = []
    low_ops: dict[str, int] = {}
    for i in range(n_draws):
        inputs = draw(i)
        try:
            r64, ops = run_ref64(
                ref64, [a.to(ref64_dev) if isinstance(a, Tensor) else a for a in inputs]
            )
            r32 = run_ref(ref32, inputs)
            rt = run_ref(ref_tf32, _map_inputs(inputs, _rounder(10)))
            rb = run_ref(ref_bf16, _map_inputs(inputs, _rounder(7)))
        except Exception as exc:
            return {
                "status": "ref_broken",
                "error": f"{type(exc).__name__}: {_trunc(exc)}",
                "draw": i,
                **meta,
            }
        for k, v in ops.items():
            low_ops[k] = low_ops.get(k, 0) + v
        try:
            with torch.set_grad_enabled(not no_grad):
                c = _float_outputs(cand(*_map_inputs(inputs, _clone)))
        except Exception as exc:
            return {
                "status": "cand_native_fail" if i == 0 else "cand_draw_fail",
                "error": f"{type(exc).__name__}: {_trunc(exc)}",
                "draw": i,
                **meta,
            }
        shapes_c, shapes_r = [list(t.shape) for t in c], [list(t.shape) for t in r64]
        if shapes_c != shapes_r:
            return {
                "status": "cand_output_mismatch",
                "error": f"candidate float outputs {shapes_c} vs reference {shapes_r}",
                "draw": i,
                **meta,
            }
        if not r64:
            return {"status": "not_auditable", "error": "no floating-point output", **meta}
        outputs = [_compare(*ts) for ts in zip(c, r64, r32, rt, rb, strict=True)]
        draws.append({"seed": AUDIT_SEED + i, "outputs": outputs})

    per_output = [o for d in draws for o in d["outputs"]]
    verdict = max((o["verdict"] for o in per_output), key=_SEVERITY.__getitem__)
    reason = ""
    if low_ops:
        verdict = "inconclusive" if verdict == "pass" else verdict
        reason = "fp64 reference ran ops below fp64: " + _trunc(low_ops, 200)
    seps = [o["separation"] for o in per_output if o["separation"] is not None]
    return {
        "status": "measured",
        "verdict": verdict,
        "reason": reason,
        "worst": {
            "bits_lost": _max_or_none([o["bits_lost"] for o in per_output]),
            "vs_tf32": _max_or_none([o["vs_tf32"] for o in per_output]),
            "min_separation": min(seps) if seps else None,
            "nonfinite_mismatch": sum(o["nonfinite_mismatch"] for o in per_output),
        },
        "ref64": {"device": str(ref64_dev), "low_precision_ops": low_ops},
        "draws": draws,
        **meta,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("ref", help="reference source (defines Model, get_inputs, get_init_inputs)")
    p.add_argument("cand", help="candidate source (defines ModelNew)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    p.add_argument("--train", action="store_true")
    p.add_argument("--no-grad", action="store_true")
    p.add_argument("--no-strict-backend", action="store_true")
    a = p.parse_args(argv)
    with open(a.ref, encoding="utf-8") as f:
        ref_src = f.read()
    with open(a.cand, encoding="utf-8") as f:
        cand_src = f.read()
    result = precision_worker(
        ref_src,
        cand_src,
        {
            "device": a.device,
            "draws": a.draws,
            "train": a.train,
            "no_grad": a.no_grad,
            "strict_backend": not a.no_strict_backend,
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "measured" else 1


if __name__ == "__main__":
    sys.exit(main())
