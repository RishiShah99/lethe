"""Kernel Contract gates — arXiv 2604.22032.

Six gates are fully implemented; six are stubbed with NotImplementedError
pending cross-check against the paper's exact spec.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Implemented gates
# ---------------------------------------------------------------------------


def gate_cmp_01_input_variation(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    n_random: int = 8,
    **kwargs: Any,
) -> GateResult:
    """CMP-01: Input variation — random + adversarial inputs must match.

    Generates *n_random* random inputs and a bank of adversarial inputs
    (zeros, large magnitudes, small magnitudes, denormals, long sequences).
    For every input the candidate and reference outputs are compared with
    ``torch.allclose``.
    """
    failures: list[str] = []

    def _check(tensor: torch.Tensor, label: str) -> None:
        try:
            out_ref = reference(tensor)
            out_cand = candidate(tensor)
        except Exception as exc:
            failures.append(f"{label}: exception — {exc}")
            return
        if out_cand.shape != out_ref.shape:
            failures.append(f"{label}: shape mismatch {out_cand.shape} vs {out_ref.shape}")
            return
        if not torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=rtol):
            max_err = (out_cand.float() - out_ref.float()).abs().max().item()
            failures.append(f"{label}: max_err={max_err:.3e} > atol={atol}")

    # Random inputs
    for i in range(n_random):
        t = torch.randn(shape, dtype=dtype)
        _check(t, f"random[{i}]")

    # Adversarial: zeros
    _check(torch.zeros(shape, dtype=dtype), "zeros")

    # Adversarial: large magnitude
    _check(torch.full(shape, 1e6, dtype=dtype), "large_1e6")

    # Adversarial: small magnitude
    _check(torch.full(shape, 1e-6, dtype=dtype), "small_1e-6")

    # Adversarial: denormals (smallest positive subnormal)
    if dtype == torch.float32:
        denorm_val = torch.tensor(1.175494e-38, dtype=torch.float32)
        _check(torch.full(shape, denorm_val.item(), dtype=dtype), "denormals")

    # Adversarial: longer sequence (double the second dim)
    long_shape = (shape[0], shape[1] * 4, *shape[2:])
    _check(torch.randn(long_shape, dtype=dtype), "long_seq")

    if failures:
        return GateResult(
            passed=False,
            reason=f"{len(failures)} input variation(s) failed",
            details={"failures": failures},
        )
    return GateResult(passed=True, reason="all inputs matched", details={})


def gate_cmp_03_shape_polymorphism(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    **kwargs: Any,
) -> GateResult:
    """CMP-03: Shape polymorphism — vary batch, length, and model-dim independently."""
    shapes = [
        (1, 16, 8),
        (2, 32, 16),
        (4, 64, 32),
        (8, 128, 16),
        (1, 256, 64),
    ]
    failures: list[str] = []
    for shape in shapes:
        t = torch.randn(shape, dtype=dtype)
        try:
            out_ref = reference(t)
            out_cand = candidate(t)
        except Exception as exc:
            failures.append(f"shape={shape}: exception — {exc}")
            continue
        if out_cand.shape != out_ref.shape:
            failures.append(
                f"shape={shape}: output shape mismatch {out_cand.shape} vs {out_ref.shape}"
            )
            continue
        if not torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=rtol):
            max_err = (out_cand.float() - out_ref.float()).abs().max().item()
            failures.append(f"shape={shape}: max_err={max_err:.3e}")

    if failures:
        return GateResult(
            passed=False,
            reason=f"{len(failures)} shape(s) failed",
            details={"failures": failures},
        )
    return GateResult(passed=True, reason="all shapes matched", details={})


def gate_ord_01_reduction_order_tolerance(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 512, 32),
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GateResult:
    """ORD-01: Reduction-order tolerance — loosen atol based on sqrt(N).

    Floating-point reductions are order-dependent; a kernel that reduces in a
    different order than the reference may still be numerically correct.  The
    tolerance is scaled by ``eps * sqrt(N) * dtype_eps`` where N is the number
    of elements being reduced (last dimension by convention).
    """
    n_elements = shape[-1]
    dtype_eps: float
    if dtype == torch.float16:
        dtype_eps = 9.77e-4  # torch.finfo(torch.float16).eps
    elif dtype == torch.bfloat16:
        dtype_eps = 7.81e-3  # torch.finfo(torch.bfloat16).eps
    else:
        dtype_eps = 1.19e-7  # torch.finfo(torch.float32).eps

    atol = dtype_eps * math.sqrt(n_elements)

    t = torch.randn(shape, dtype=dtype)
    try:
        out_ref = reference(t)
        out_cand = candidate(t)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"exception during execution: {exc}",
            details={"atol_used": atol},
        )

    if out_cand.shape != out_ref.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch {out_cand.shape} vs {out_ref.shape}",
            details={"atol_used": atol},
        )

    ok = torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=0.0)
    max_err = (out_cand.float() - out_ref.float()).abs().max().item()
    return GateResult(
        passed=ok,
        reason="within reduction-order tolerance"
        if ok
        else f"max_err={max_err:.3e} > atol={atol:.3e}",
        details={"atol_used": atol, "max_err": max_err, "n_elements": n_elements},
    )


def gate_ord_02_atomic_determinism(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    n_runs: int = 5,
    **kwargs: Any,
) -> GateResult:
    """ORD-02: Atomic determinism — repeated calls with identical input must be byte-identical."""
    t = torch.randn(shape, dtype=dtype)
    outputs: list[torch.Tensor] = []
    for _ in range(n_runs):
        try:
            out = candidate(t.clone())
        except Exception as exc:
            return GateResult(
                passed=False,
                reason=f"exception during run: {exc}",
                details={"n_runs": n_runs},
            )
        outputs.append(out)

    for i in range(1, n_runs):
        if not torch.equal(outputs[0], outputs[i]):
            return GateResult(
                passed=False,
                reason=f"run 0 and run {i} differ (non-deterministic)",
                details={"n_runs": n_runs},
            )
    return GateResult(
        passed=True,
        reason=f"all {n_runs} runs byte-identical",
        details={"n_runs": n_runs},
    )


def gate_prc_01_precision_regime(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    **kwargs: Any,
) -> GateResult:
    """PRC-01: Precision regime — test FP32, FP16, BF16 with per-dtype tolerances."""
    dtype_configs: list[tuple[torch.dtype, float, float]] = [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ]
    failures: list[str] = []
    for dtype, atol, rtol in dtype_configs:
        t = torch.randn(shape, dtype=dtype)
        try:
            out_ref = reference(t)
            out_cand = candidate(t)
        except Exception as exc:
            failures.append(f"{dtype}: exception — {exc}")
            continue
        if out_cand.shape != out_ref.shape:
            failures.append(f"{dtype}: shape mismatch")
            continue
        if not torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=rtol):
            max_err = (out_cand.float() - out_ref.float()).abs().max().item()
            failures.append(f"{dtype}: max_err={max_err:.3e} > atol={atol}")

    if failures:
        return GateResult(
            passed=False,
            reason=f"{len(failures)} dtype(s) failed",
            details={"failures": failures},
        )
    return GateResult(passed=True, reason="all dtypes within tolerance", details={})


def gate_exc_01_exceptional_values(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GateResult:
    """EXC-01: Exceptional values — NaN and signed-Inf positions must agree.

    Compares ``isposinf`` and ``isneginf`` independently rather than the
    union ``isinf`` mask, so a kernel that silently flips +inf to -inf
    (or vice versa) is rejected.
    """
    failures: list[str] = []

    def _check_exceptional(t: torch.Tensor, label: str) -> None:
        try:
            out_ref = reference(t)
            out_cand = candidate(t)
        except Exception as exc:
            failures.append(f"{label}: exception — {exc}")
            return
        if out_cand.shape != out_ref.shape:
            failures.append(f"{label}: shape mismatch")
            return
        nan_ref = torch.isnan(out_ref)
        nan_cand = torch.isnan(out_cand)
        if not torch.equal(nan_ref, nan_cand):
            failures.append(f"{label}: NaN mask mismatch")
        posinf_ref = torch.isposinf(out_ref)
        posinf_cand = torch.isposinf(out_cand)
        if not torch.equal(posinf_ref, posinf_cand):
            failures.append(f"{label}: +Inf mask mismatch (sign flip?)")
        neginf_ref = torch.isneginf(out_ref)
        neginf_cand = torch.isneginf(out_cand)
        if not torch.equal(neginf_ref, neginf_cand):
            failures.append(f"{label}: -Inf mask mismatch (sign flip?)")

    # Input with NaN scattered at known positions
    t_nan = torch.randn(shape, dtype=dtype)
    t_nan.view(-1)[:: max(1, t_nan.numel() // 8)] = float("nan")
    _check_exceptional(t_nan, "scattered_nan")

    # Input with +Inf
    t_inf = torch.randn(shape, dtype=dtype)
    t_inf.view(-1)[:: max(1, t_inf.numel() // 8)] = float("inf")
    _check_exceptional(t_inf, "scattered_pos_inf")

    # Input with -Inf
    t_neginf = torch.randn(shape, dtype=dtype)
    t_neginf.view(-1)[:: max(1, t_neginf.numel() // 8)] = float("-inf")
    _check_exceptional(t_neginf, "scattered_neg_inf")

    if failures:
        return GateResult(
            passed=False,
            reason=f"{len(failures)} exceptional-value check(s) failed",
            details={"failures": failures},
        )
    return GateResult(
        passed=True,
        reason="NaN/+Inf/-Inf masks agree with reference",
        details={},
    )


# ---------------------------------------------------------------------------
# Stubbed gates (not yet cross-checked against arXiv 2604.22032)
# ---------------------------------------------------------------------------


def gate_cmp_02_gradient_correctness(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """CMP-02: Gradient correctness — autograd gradients must match reference.

    Verifies that ``torch.autograd.gradcheck`` passes for the candidate kernel,
    ensuring backward-pass correctness within a tolerance derived from finite
    differences.
    """
    raise NotImplementedError("gate_cmp_02_gradient_correctness not yet implemented")


def gate_ord_03_noncommutative_reduction(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """ORD-03: Non-commutative reduction — operations that are not commutative
    (e.g., sequential scan / prefix-sum) must produce outputs that match the
    reference *exactly*, regardless of parallelisation strategy.
    """
    raise NotImplementedError("gate_ord_03_noncommutative_reduction not yet implemented")


def gate_prc_02_mixed_precision_accumulation(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """PRC-02: Mixed-precision accumulation — verify that intermediate
    accumulation in a higher precision (e.g., FP32 accumulator for FP16 input)
    yields outputs within the expected tolerance of the full-precision reference.
    """
    raise NotImplementedError("gate_prc_02_mixed_precision_accumulation not yet implemented")


def gate_exc_02_subnormal_handling(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """EXC-02: Subnormal handling — inputs in the subnormal (denormal) range
    must be handled consistently with the reference; flush-to-zero behaviour
    must match if the reference also flushes.
    """
    raise NotImplementedError("gate_exc_02_subnormal_handling not yet implemented")


def gate_res_01_memory_residency(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """RES-01: Memory residency / no host roundtrip — outputs must remain on the
    same device as the inputs (no silent CPU↔GPU copies that inflate latency).
    """
    raise NotImplementedError("gate_res_01_memory_residency not yet implemented")


def gate_res_02_resource_limits(  # TODO: cross-check name + spec against arXiv 2604.22032
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> GateResult:
    """RES-02: Resource limits — register count and shared-memory usage must not
    exceed the hardware maximums for the target architecture (queried via
    ``triton.compiler.get_arch_linear_id`` or ptxas ``--verbose`` output).
    """
    raise NotImplementedError("gate_res_02_resource_limits not yet implemented")


# ---------------------------------------------------------------------------
# Aggregate runner
# ---------------------------------------------------------------------------

_ALL_GATE_NAMES: list[str] = [
    "gate_cmp_01_input_variation",
    "gate_cmp_02_gradient_correctness",
    "gate_cmp_03_shape_polymorphism",
    "gate_ord_01_reduction_order_tolerance",
    "gate_ord_02_atomic_determinism",
    "gate_ord_03_noncommutative_reduction",
    "gate_prc_01_precision_regime",
    "gate_prc_02_mixed_precision_accumulation",
    "gate_exc_01_exceptional_values",
    "gate_exc_02_subnormal_handling",
    "gate_res_01_memory_residency",
    "gate_res_02_resource_limits",
]

_GATE_MAP: dict[str, Callable[..., GateResult]] = {
    "gate_cmp_01_input_variation": gate_cmp_01_input_variation,
    "gate_cmp_02_gradient_correctness": gate_cmp_02_gradient_correctness,
    "gate_cmp_03_shape_polymorphism": gate_cmp_03_shape_polymorphism,
    "gate_ord_01_reduction_order_tolerance": gate_ord_01_reduction_order_tolerance,
    "gate_ord_02_atomic_determinism": gate_ord_02_atomic_determinism,
    "gate_ord_03_noncommutative_reduction": gate_ord_03_noncommutative_reduction,
    "gate_prc_01_precision_regime": gate_prc_01_precision_regime,
    "gate_prc_02_mixed_precision_accumulation": gate_prc_02_mixed_precision_accumulation,
    "gate_exc_01_exceptional_values": gate_exc_01_exceptional_values,
    "gate_exc_02_subnormal_handling": gate_exc_02_subnormal_handling,
    "gate_res_01_memory_residency": gate_res_01_memory_residency,
    "gate_res_02_resource_limits": gate_res_02_resource_limits,
}


def run_all_gates(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> dict[str, GateResult]:
    """Run all 12 Kernel Contract gates and return results keyed by gate name.

    Stubbed gates that raise ``NotImplementedError`` are captured and recorded
    as ``passed=False`` with reason ``"not_implemented"``.
    """
    results: dict[str, GateResult] = {}
    for name in _ALL_GATE_NAMES:
        gate_fn = _GATE_MAP[name]
        try:
            results[name] = gate_fn(candidate, reference, **kwargs)
        except NotImplementedError:
            results[name] = GateResult(
                passed=False,
                reason="not_implemented",
                details={"stub": True},
            )
    return results
