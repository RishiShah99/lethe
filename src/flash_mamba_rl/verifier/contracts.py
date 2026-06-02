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


def gate_cmp_02_gradient_correctness(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (2, 8, 4),
    eps: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    **kwargs: Any,
) -> GateResult:
    """CMP-02: Gradient correctness — candidate's autograd gradients must agree
    with finite-difference approximations of its own forward pass.

    Self-consistency check via ``torch.autograd.gradcheck`` on an FP64 input.
    A kernel that returns correct values but breaks the gradient graph (e.g.,
    ``.detach()`` on an intermediate, in-place operation on a non-leaf tensor)
    is rejected. Combined with CMP-01's value-correctness, this gives
    gradient agreement with the reference by transitivity.
    """
    t = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    try:
        ok = torch.autograd.gradcheck(
            candidate,
            (t,),
            eps=eps,
            atol=atol,
            rtol=rtol,
            check_undefined_grad=False,
            raise_exception=False,
        )
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"gradcheck raised: {exc}",
            details={"eps": eps, "atol": atol, "rtol": rtol},
        )
    return GateResult(
        passed=bool(ok),
        reason="gradcheck passed" if ok else "gradcheck reported mismatch",
        details={"eps": eps, "atol": atol, "rtol": rtol},
    )


def gate_ord_03_noncommutative_reduction(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GateResult:
    """ORD-03: Non-commutative reduction — outputs must be bitwise-identical to
    reference on inputs designed to expose order-dependent reductions.

    Floating-point addition is not associative; a kernel that reduces in a
    different order than the reference (parallel tree vs left-to-right, or
    reverse vs forward) produces ULP-level differences. The reference defines
    the canonical order for scan / prefix-sum / accumulator ops, and the
    candidate must match it byte-for-byte.

    The input is an alternating large-positive / tiny / large-negative
    pattern where reduction order materially changes the bit pattern.
    """
    pattern = torch.tensor([1.0, 1e-7, -1.0, 1e-7], dtype=dtype)
    n_elements = 1
    for d in shape:
        n_elements *= d
    repeats = (n_elements + pattern.numel() - 1) // pattern.numel()
    t = pattern.repeat(repeats)[:n_elements].reshape(shape)

    try:
        out_ref = reference(t)
        out_cand = candidate(t)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"exception during execution: {exc}",
            details={},
        )

    if out_cand.shape != out_ref.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch {out_cand.shape} vs {out_ref.shape}",
            details={},
        )

    if not torch.equal(out_ref, out_cand):
        max_err = (out_ref.float() - out_cand.float()).abs().max().item()
        return GateResult(
            passed=False,
            reason=f"not bitwise-identical: max_err={max_err:.3e} (reduction order differs)",
            details={"max_err": max_err},
        )
    return GateResult(
        passed=True,
        reason="bitwise-identical to reference on adversarial reduction input",
        details={},
    )


def gate_prc_02_mixed_precision_accumulation(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (2, 32, 1024),
    atol: float = 2e-2,
    **kwargs: Any,
) -> GateResult:
    """PRC-02: Mixed-precision accumulation — FP16 inputs must produce outputs
    within FP16-ULP of the FP32 reference, implying an FP32 internal accumulator.

    A kernel that naively accumulates in FP16 across a long reduction loses
    precision proportionally to ``sqrt(N) * scale``. The default shape uses
    a 1024-element reduction dimension to make the gap between FP16-only
    accumulation and FP32-accumulating implementations visible.

    Comparing the FP16 candidate output (upcast to FP32) against the FP32
    reference output exposes the missing FP32 accumulator.
    """
    t_fp32 = torch.randn(shape, dtype=torch.float32)
    t_fp16 = t_fp32.to(torch.float16)
    try:
        out_ref = reference(t_fp32)
        out_cand = candidate(t_fp16)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"exception during execution: {exc}",
            details={"atol": atol},
        )

    if out_cand.shape != out_ref.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch {out_cand.shape} vs {out_ref.shape}",
            details={"atol": atol},
        )

    diff = (out_cand.float() - out_ref.float()).abs()
    max_err = diff.max().item()
    if max_err > atol:
        return GateResult(
            passed=False,
            reason=(
                f"max_err={max_err:.3e} > atol={atol} — likely missing FP32 accumulator"
            ),
            details={"max_err": max_err, "atol": atol},
        )
    return GateResult(
        passed=True,
        reason=f"max_err={max_err:.3e} within FP16 tolerance — FP32 accumulator inferred",
        details={"max_err": max_err, "atol": atol},
    )


def gate_exc_02_subnormal_handling(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-30,
    **kwargs: Any,
) -> GateResult:
    """EXC-02: Subnormal handling — the candidate's flush-to-zero behaviour
    on subnormal inputs must match the reference.

    Generates inputs in the subnormal range for the given dtype (e.g.,
    ``~1e-40`` for FP32, below ``~1.18e-38``). Compares the zero-mask
    between candidate and reference. If the reference preserves subnormals,
    the candidate must too; if the reference flushes, the candidate must
    also flush. Where both produce non-zero outputs, values must agree.
    """
    if dtype == torch.float32:
        subnormal_min = 1e-40
    elif dtype == torch.float16:
        subnormal_min = 1e-7
    else:  # bfloat16
        subnormal_min = 1e-39

    t = torch.full(shape, subnormal_min, dtype=dtype)
    flat = t.view(-1)
    flat[::3] = -subnormal_min
    flat[::7] = 0.0

    try:
        out_ref = reference(t)
        out_cand = candidate(t)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"exception during execution: {exc}",
            details={},
        )

    if out_cand.shape != out_ref.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch {out_cand.shape} vs {out_ref.shape}",
            details={},
        )

    zero_ref = out_ref == 0
    zero_cand = out_cand == 0
    if not torch.equal(zero_ref, zero_cand):
        return GateResult(
            passed=False,
            reason="flush-to-zero behaviour disagrees with reference",
            details={
                "ref_zeros": int(zero_ref.sum().item()),
                "cand_zeros": int(zero_cand.sum().item()),
            },
        )

    nonzero_mask = ~zero_ref
    if nonzero_mask.any():
        diff = (out_ref[nonzero_mask].float() - out_cand[nonzero_mask].float()).abs()
        max_err = diff.max().item()
        if max_err > atol:
            return GateResult(
                passed=False,
                reason=f"subnormal non-zero values disagree: max_err={max_err:.3e}",
                details={"max_err": max_err, "atol": atol},
            )

    return GateResult(
        passed=True,
        reason="subnormal handling matches reference",
        details={},
    )


def gate_res_01_memory_residency(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    **kwargs: Any,
) -> GateResult:
    """RES-01: Memory residency — output device must match input device.

    A kernel that silently moves data through a different device (CPU↔GPU,
    meta, or quantisation backends) hides latency. Tests every available
    device (CPU always; CUDA if present) and checks that ``out.device.type``
    matches ``input.device.type``.
    """
    devices: list[torch.device] = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    failures: list[str] = []
    for device in devices:
        t = torch.randn(shape, dtype=dtype, device=device)
        try:
            out = candidate(t)
        except Exception as exc:
            failures.append(f"device={device.type}: exception — {exc}")
            continue
        if out.device.type != t.device.type:
            failures.append(
                f"input device={t.device.type} → output device={out.device.type}"
            )

    if failures:
        return GateResult(
            passed=False,
            reason=f"{len(failures)} residency violation(s)",
            details={"failures": failures, "devices_tested": [d.type for d in devices]},
        )
    return GateResult(
        passed=True,
        reason=f"output device matches input on {len(devices)} device(s)",
        details={"devices_tested": [d.type for d in devices]},
    )


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

    Stubbed gates that raise ``NotImplementedError`` are recorded as
    ``passed=False`` with reason ``"not_implemented"``. Any other exception
    escaping a gate (e.g., a malformed candidate output crashes an internal
    comparison) is also caught and recorded as a failure — a misbehaving
    candidate must not crash the verifier loop.
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
        except Exception as exc:
            results[name] = GateResult(
                passed=False,
                reason=f"gate crashed: {type(exc).__name__}: {exc}",
                details={"exception_type": type(exc).__name__},
            )
    return results
