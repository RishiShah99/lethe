"""Audit harness: foreign KernelBench-convention kernel pairs through the contract gates."""

from __future__ import annotations

import copy
import importlib.util
import math
import os
import sys
import tempfile
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from lethe.verifier.contracts import GateResult, run_all_gates

AUDIT_SEED = 1337

# Marks reference-inapplicable failures as skipped coverage, not candidate failures.
_RI_MARKER = "[ref-inapplicable::fmr-ri-9e3779b97f4a7c15]"

# Gate subset within the audited corpora's claimed scope (no CMP-02/RES-02).
AUDIT_GATE_NAMES: tuple[str, ...] = (
    "gate_cmp_01_input_variation",
    "gate_cmp_03_shape_polymorphism",
    "gate_ord_01_reduction_order_tolerance",
    "gate_ord_02_atomic_determinism",
    "gate_ord_03_noncommutative_reduction",
    "gate_prc_01_precision_regime",
    "gate_prc_02_mixed_precision_accumulation",
    "gate_exc_01_exceptional_values",
    "gate_exc_02_subnormal_handling",
    "gate_res_01_memory_residency",
)

GATE_SHORT_NAMES: dict[str, str] = {
    "gate_cmp_01_input_variation": "CMP-01",
    "gate_cmp_03_shape_polymorphism": "CMP-03",
    "gate_ord_01_reduction_order_tolerance": "ORD-01",
    "gate_ord_02_atomic_determinism": "ORD-02",
    "gate_ord_03_noncommutative_reduction": "ORD-03",
    "gate_prc_01_precision_regime": "PRC-01",
    "gate_prc_02_mixed_precision_accumulation": "PRC-02",
    "gate_exc_01_exceptional_values": "EXC-01",
    "gate_exc_02_subnormal_handling": "EXC-02",
    "gate_res_01_memory_residency": "RES-01",
}

# 8 random + zeros + large + small + denormals (fp32) + long_seq.
_CMP01_N_CHECKS = 13
_PRC01_N_CHECKS = 3
_EXC01_N_CHECKS = 3


def _trunc(obj: Any, limit: int = 300) -> str:
    text = str(obj)
    return text if len(text) <= limit else text[:limit] + "..."


def _exec_source(source: str, required: str) -> dict[str, Any]:
    # @triton.jit needs a real source file, so we write a temp .py kept alive for the worker's lifetime.
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"audit_{required.lower()}_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    name = f"_audit_{required.lower()}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    ns = vars(mod)
    if required not in ns:
        raise KeyError(f"source defines no {required}")
    return dict(ns)


def _first_tensor(out: Any) -> Tensor:
    if isinstance(out, Tensor):
        return out
    if isinstance(out, tuple | list):
        for item in out:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"no tensor in output of type {type(out).__name__}")


class _ModuleAdapter:
    """Close a KernelBench module over its non-primary inputs as a single-tensor callable."""

    def __init__(
        self,
        model: nn.Module,
        inputs: list[Any],
        primary_idx: int,
        *,
        is_reference: bool,
        ctor: Callable[..., nn.Module] | None = None,
        init_args: list[Any] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._models: dict[torch.dtype, nn.Module] = {torch.float32: model}
        self._aux: dict[torch.dtype, list[Any]] = {torch.float32: inputs}
        self._primary_idx = primary_idx
        self._is_reference = is_reference
        self._ctor = ctor
        self._init_args = init_args if init_args is not None else []
        self._device = device
        self._prev_raw: Tensor | None = None
        self.aliased = False
        self.multi_output = False

    def _rebuild(self) -> nn.Module:
        # Seeded re-instantiation reproduces fp32 params; deepcopy (fallback) can choke on locks/caches.
        if self._ctor is None:
            return copy.deepcopy(self._models[torch.float32])
        torch.manual_seed(AUDIT_SEED)
        model = self._ctor(*self._init_args)
        if self._device is not None:
            model = model.to(self._device)
        return model

    def _variant(self, dtype: torch.dtype) -> tuple[nn.Module, list[Any]]:
        if dtype not in self._models:
            base_aux = self._aux[torch.float32]
            model = self._rebuild()
            if self._is_reference:
                with torch.no_grad():
                    for p in model.parameters():
                        p.copy_(p.to(dtype).float())
                    for b in model.buffers():
                        if b.is_floating_point():
                            b.copy_(b.to(dtype).float())
                aux = [
                    a.to(dtype).float() if isinstance(a, Tensor) and a.is_floating_point() else a
                    for a in base_aux
                ]
            else:
                model = model.to(dtype)
                aux = [
                    a.to(dtype) if isinstance(a, Tensor) and a.is_floating_point() else a
                    for a in base_aux
                ]
            model.eval()
            self._models[dtype] = model
            self._aux[dtype] = aux
        return self._models[dtype], self._aux[dtype]

    def _run(self, t: Tensor) -> Tensor:
        dtype = t.dtype
        if dtype == torch.float32:
            model, aux = self._models[torch.float32], self._aux[torch.float32]
            primary: Tensor = t
        else:
            model, aux = self._variant(dtype)
            primary = t.float() if self._is_reference else t
        args = list(aux)
        args[self._primary_idx] = primary
        raw = model(*args)
        if isinstance(raw, tuple | list) and len(raw) > 1:
            self.multi_output = True
        out = _first_tensor(raw)
        if self._is_reference and dtype != torch.float32:
            out = out.to(dtype)
        return out

    def __call__(self, t: Tensor) -> Tensor:
        if self._is_reference:
            try:
                return self._run(t)
            except Exception as exc:
                raise RuntimeError(
                    f"{_RI_MARKER} {type(exc).__name__}: {_trunc(exc, 200)}"
                ) from exc
        out = self._run(t)
        if (
            self._prev_raw is not None
            and out.numel() > 0
            and out.data_ptr() == self._prev_raw.data_ptr()
        ):
            self.aliased = True
        self._prev_raw = out
        return out.clone()


def _shape_variants(shape: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Non-native shape variants; reference applicability decides which count."""
    dims = list(shape)
    variants: list[tuple[int, ...]] = []
    if dims:
        if dims[0] > 1:
            variants.append((dims[0] // 2, *dims[1:]))
        variants.append((dims[0] * 2, *dims[1:]))
    if len(dims) >= 2:
        variants.append((*dims[:-1], dims[-1] * 2))
        if dims[-1] > 1 and dims[-1] % 2 == 0:
            variants.append((*dims[:-1], dims[-1] // 2))
    seen: set[tuple[int, ...]] = {shape}
    out: list[tuple[int, ...]] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _gate_status(name: str, result: GateResult, n_checks: int | None) -> dict[str, Any]:
    """Collapse a GateResult to {status, reason, skipped} with N/A reclassification."""
    if result.passed:
        return {"status": "pass", "reason": "", "skipped": 0}
    if result.reason.startswith("gate crashed") or result.reason == "not_implemented":
        return {"status": "error", "reason": _trunc(result.reason), "skipped": 0}

    failures = result.details.get("failures")
    if failures is None:
        if _RI_MARKER in result.reason:
            return {"status": "na", "reason": _trunc(result.reason), "skipped": 1}
        return {"status": "fail", "reason": _trunc(result.reason), "skipped": 0}

    if name == "gate_res_01_memory_residency":
        real = [f for f in failures if "output device" in f]
        skipped = [f for f in failures if "output device" not in f]
        n_checks = len(result.details.get("devices_tested", [])) or None
    elif name == "gate_prc_01_precision_regime":
        # fp32 duplicates CMP-01 at an untunable tolerance; CMP-01 owns fp32, PRC-01 audits half-precision.
        real = [f for f in failures if _RI_MARKER not in f and "float32" not in f]
        skipped = [f for f in failures if _RI_MARKER in f or "float32" in f]
    else:
        real = [f for f in failures if _RI_MARKER not in f]
        skipped = [f for f in failures if _RI_MARKER in f]

    if real:
        return {
            "status": "fail",
            "reason": _trunc("; ".join(real[:3])),
            "skipped": len(skipped),
        }
    if n_checks is not None and len(skipped) >= n_checks:
        return {
            "status": "na",
            "reason": _trunc(skipped[0]) if skipped else "",
            "skipped": len(skipped),
        }
    return {"status": "pass", "reason": "", "skipped": len(skipped)}


def audit_worker(ref_source: str, cand_source: str, config: dict[str, Any]) -> dict[str, Any]:
    """Audit one (reference, candidate) source pair; returns a picklable result dict."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return _audit_worker_body(ref_source, cand_source, config)
    finally:
        sys.stdout = real_stdout


def _audit_worker_body(ref_source: str, cand_source: str, config: dict[str, Any]) -> dict[str, Any]:
    device = str(config.get("device", "cpu"))
    torch.manual_seed(AUDIT_SEED)

    try:
        ref_ns = _exec_source(ref_source, "Model")
        get_inputs = ref_ns["get_inputs"]
        get_init = ref_ns.get("get_init_inputs")
        torch.manual_seed(AUDIT_SEED)
        init_args = list(get_init() or []) if get_init is not None else []
        torch.manual_seed(AUDIT_SEED)
        ref_model = ref_ns["Model"](*init_args)
        torch.manual_seed(AUDIT_SEED)
        inputs = list(get_inputs())
    except Exception as exc:
        return {"status": "ref_broken", "error": f"{type(exc).__name__}: {_trunc(exc)}"}

    dev = torch.device(device)
    primary_idx: int | None = None
    for i, a in enumerate(inputs):
        if isinstance(a, Tensor) and a.is_floating_point():
            primary_idx = i
            break
    if primary_idx is None:
        return {"status": "not_auditable", "error": "no floating-point tensor input"}

    try:
        ref_model = ref_model.to(dev).eval()
        inputs = [a.to(dev) if isinstance(a, Tensor) else a for a in inputs]
        native = inputs[primary_idx]
        ref_adapter = _ModuleAdapter(
            ref_model,
            inputs,
            primary_idx,
            is_reference=True,
            ctor=ref_ns["Model"],
            init_args=init_args,
            device=dev,
        )
        ref_adapter(native)
    except TypeError as exc:
        return {"status": "not_auditable", "error": f"{type(exc).__name__}: {_trunc(exc)}"}
    except Exception as exc:
        return {"status": "ref_broken", "error": f"{type(exc).__name__}: {_trunc(exc)}"}

    try:
        cand_ns = _exec_source(cand_source, "ModelNew")
        torch.manual_seed(AUDIT_SEED)
        cand_model = cand_ns["ModelNew"](*init_args)
        cand_model = cand_model.to(dev).eval()
    except Exception as exc:
        return {"status": "cand_load_fail", "error": f"{type(exc).__name__}: {_trunc(exc)}"}

    cand_adapter = _ModuleAdapter(
        cand_model,
        inputs,
        primary_idx,
        is_reference=False,
        ctor=cand_ns["ModelNew"],
        init_args=init_args,
        device=dev,
    )
    try:
        cand_adapter(native)
    except Exception as exc:
        return {
            "status": "cand_native_fail",
            "error": f"{type(exc).__name__}: {_trunc(exc)}",
            "native_shape": list(native.shape),
        }

    shape = tuple(native.shape)
    variants = _shape_variants(shape)
    # C1 model: reorder noise ~ eps*sqrt(chain)*scale; chain extent unknown, bound by element count.
    fp32_atol = max(1e-5, 4 * 1.19e-7 * math.sqrt(native.numel()))
    gate_overrides: dict[str, dict[str, Any]] = {
        "gate_cmp_01_input_variation": {"atol": fp32_atol},
        "gate_cmp_03_shape_polymorphism": {"atol": fp32_atol, "shapes": variants},
        "gate_ord_01_reduction_order_tolerance": {"reduction_elements": int(native.numel())},
        "gate_ord_03_noncommutative_reduction": {"atol": 1e-3, "rtol": 1e-3},
        "gate_prc_02_mixed_precision_accumulation": {"scale_atol_by_ref_inf": True},
        "gate_exc_02_subnormal_handling": {"atol": 1e-4},
    }
    raw = run_all_gates(
        cand_adapter,
        ref_adapter,
        gate_names=list(AUDIT_GATE_NAMES),
        gate_overrides=gate_overrides,
        shape=shape,
        device=device,
        seed=None,  # closed artifact: keep the legacy AUDIT_SEED-driven draws
    )

    n_checks: dict[str, int | None] = {
        "gate_cmp_01_input_variation": _CMP01_N_CHECKS,
        "gate_cmp_03_shape_polymorphism": len(variants) or 1,
        "gate_prc_01_precision_regime": _PRC01_N_CHECKS,
        "gate_exc_01_exceptional_values": _EXC01_N_CHECKS,
    }
    gates = {
        GATE_SHORT_NAMES[name]: _gate_status(name, result, n_checks.get(name, 1))
        for name, result in raw.items()
    }
    return {
        "status": "gated",
        "primary_idx": primary_idx,
        "native_shape": list(shape),
        "gates": gates,
        "output_aliasing": cand_adapter.aliased,
        "multi_output": cand_adapter.multi_output,
    }
