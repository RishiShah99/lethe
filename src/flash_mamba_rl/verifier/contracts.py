"""Kernel Contract gates — arXiv 2604.22032.

All 12 gates implemented. RES-02 is evidence-based: it validates resource
metadata extracted at compile time (registers, shared memory, TMEM) against
hardware limits, and reports not-applicable when no metadata is supplied
(plain-Python candidates have no compiled artifact to measure).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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
    device: str | torch.device = "cpu",
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

    The absolute tolerance is scaled by the reference output's magnitude
    (``max(1, |out_ref|_inf)``): on a large-magnitude problem the error of
    *any* correctly-rounded but differently-ordered implementation carries
    the scale of the large intermediates, and elements that cancel to near
    zero would otherwise demand cancellation noise below one ULP of those
    intermediates. ``atol`` is interpreted at unit scale.
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
        ref32 = out_ref.float()
        finite = ref32[torch.isfinite(ref32)]
        scale = max(1.0, finite.abs().max().item()) if finite.numel() else 1.0
        atol_eff = atol * scale
        # equal_nan is positional: a candidate matching the reference's own
        # NaNs passes; minting NaN where the reference is finite still fails.
        if not torch.allclose(out_cand.float(), ref32, atol=atol_eff, rtol=rtol, equal_nan=True):
            max_err = (out_cand.float() - ref32).abs().max().item()
            failures.append(f"{label}: max_err={max_err:.3e} > atol={atol_eff:.3e} (scaled)")

    # Random inputs
    for i in range(n_random):
        t = torch.randn(shape, dtype=dtype, device=device)
        _check(t, f"random[{i}]")

    # Adversarial: zeros
    _check(torch.zeros(shape, dtype=dtype, device=device), "zeros")

    # Adversarial: large magnitude
    _check(torch.full(shape, 1e6, dtype=dtype, device=device), "large_1e6")

    # Adversarial: small magnitude
    _check(torch.full(shape, 1e-6, dtype=dtype, device=device), "small_1e-6")

    # Adversarial: denormals (smallest positive subnormal)
    if dtype == torch.float32:
        denorm_val = torch.tensor(1.175494e-38, dtype=torch.float32)
        _check(torch.full(shape, denorm_val.item(), dtype=dtype, device=device), "denormals")

    # Adversarial: longer sequence (4x the second dim; 1-D primaries grow dim 0)
    long_shape = (shape[0], shape[1] * 4, *shape[2:]) if len(shape) >= 2 else (shape[0] * 4,)
    _check(torch.randn(long_shape, dtype=dtype, device=device), "long_seq")

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
    device: str | torch.device = "cpu",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    shapes: list[tuple[int, ...]] | None = None,
    **kwargs: Any,
) -> GateResult:
    """CMP-03: Shape polymorphism — vary batch, length, and model-dim independently.

    ``atol`` is interpreted at unit scale and multiplied by the reference
    output's magnitude (see CMP-01): cancellation elements carry the ULP
    noise of the large intermediates, not of their own small value.

    ``shapes`` overrides the default scan-convention list for ops whose
    primary tensor is not [batch, seq, d_model] (the audit harness derives
    variants from a task's native input shape).
    """
    if shapes is None:
        shapes = [
            (1, 16, 8),
            (2, 32, 16),
            (4, 64, 32),
            (8, 128, 16),
            (1, 256, 64),
        ]
    failures: list[str] = []
    for shape in shapes:
        t = torch.randn(shape, dtype=dtype, device=device)
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
        ref32 = out_ref.float()
        finite = ref32[torch.isfinite(ref32)]
        scale = max(1.0, finite.abs().max().item()) if finite.numel() else 1.0
        if not torch.allclose(
            out_cand.float(), ref32, atol=atol * scale, rtol=rtol, equal_nan=True
        ):
            max_err = (out_cand.float() - ref32).abs().max().item()
            failures.append(f"shape={shape}: max_err={max_err:.3e} > atol={atol * scale:.3e}")

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
    device: str | torch.device = "cpu",
    reduction_elements: int | None = None,
    safety_factor: float = 4.0,
    **kwargs: Any,
) -> GateResult:
    """ORD-01: Reduction-order tolerance — atol scaled to the reduction extent.

    Floating-point reductions are order-dependent; a kernel that reduces in
    a different order than the reference may still be numerically correct.
    Two correctly-rounded implementations of an N-term accumulation differ
    by a random walk of rounding errors, so the tolerance is
    ``safety_factor * dtype_eps * sqrt(N) * max(1, |out_ref|_inf)``:

    - ``N`` is ``reduction_elements`` — the op's true accumulation extent
      (for a scan, the sequence length; for a row-sum, the row width).
      Defaults to ``shape[-1]`` (the row-op convention).
    - the output-magnitude factor carries the scale of the intermediates;
      a unit-scale atol would demand sub-ULP cancellation noise.
    - ``safety_factor`` covers the stochastic spread of the random-walk
      bound (measured ~1.1x the sqrt-N estimate for the C1 Triton scan on
      B200; 4x keeps honest kernels clear while staying orders of
      magnitude below wrong-math errors).
    """
    n_elements = reduction_elements if reduction_elements is not None else shape[-1]
    dtype_eps: float
    if dtype == torch.float16:
        dtype_eps = 9.77e-4  # torch.finfo(torch.float16).eps
    elif dtype == torch.bfloat16:
        dtype_eps = 7.81e-3  # torch.finfo(torch.bfloat16).eps
    else:
        dtype_eps = 1.19e-7  # torch.finfo(torch.float32).eps

    t = torch.randn(shape, dtype=dtype, device=device)
    try:
        out_ref = reference(t)
        out_cand = candidate(t)
    except Exception as exc:
        return GateResult(
            passed=False,
            reason=f"exception during execution: {exc}",
            details={"n_elements": n_elements},
        )

    if out_cand.shape != out_ref.shape:
        return GateResult(
            passed=False,
            reason=f"shape mismatch {out_cand.shape} vs {out_ref.shape}",
            details={"n_elements": n_elements},
        )

    scale = max(1.0, out_ref.float().abs().max().item())
    atol = safety_factor * dtype_eps * math.sqrt(n_elements) * scale

    ok = torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=0.0)
    max_err = (out_cand.float() - out_ref.float()).abs().max().item()
    return GateResult(
        passed=ok,
        reason="within reduction-order tolerance"
        if ok
        else f"max_err={max_err:.3e} > atol={atol:.3e}",
        details={
            "atol_used": atol,
            "max_err": max_err,
            "n_elements": n_elements,
            "output_scale": scale,
        },
    )


def gate_ord_02_atomic_determinism(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    n_runs: int = 5,
    **kwargs: Any,
) -> GateResult:
    """ORD-02: Atomic determinism — repeated calls with identical input must be
    byte-identical, and must not hand back one aliased output buffer.

    Two hazards, one gate. Determinism is checked on *cloned* snapshots, so a
    candidate that recomputes in place into a returned buffer is compared
    snapshot-to-snapshot, not against a live tensor a later call overwrites.
    Aliasing is checked by holding every raw output alive at once and
    requiring distinct ``data_ptr``s: a candidate that returns the same
    storage across calls (one cached buffer handed back — the cross-call
    aliasing the audit found, where holding two results corrupts the first)
    is rejected. Keeping the outputs live is what makes the check sound on
    CUDA — the caching allocator would otherwise recycle a freed output's
    address and make independent correct buffers look aliased.
    """
    t = torch.randn(shape, dtype=dtype, device=device)
    raw_outputs: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    for _ in range(n_runs):
        try:
            out = candidate(t.clone())
        except Exception as exc:
            return GateResult(
                passed=False,
                reason=f"exception during run: {exc}",
                details={"n_runs": n_runs},
            )
        raw_outputs.append(out)  # kept alive so the allocator can't reuse an address
        snapshots.append(out.clone())

    for i in range(1, n_runs):
        if not torch.equal(snapshots[0], snapshots[i]):
            return GateResult(
                passed=False,
                reason=f"run 0 and run {i} differ (non-deterministic)",
                details={"n_runs": n_runs},
            )

    data_ptrs = [o.data_ptr() for o in raw_outputs]
    if n_runs > 1 and len(set(data_ptrs)) < n_runs:
        return GateResult(
            passed=False,
            reason="cross-call output-buffer aliasing (a cached buffer reused across calls)",
            details={"n_runs": n_runs, "distinct_buffers": len(set(data_ptrs))},
        )
    return GateResult(
        passed=True,
        reason=f"all {n_runs} runs byte-identical, distinct output buffers",
        details={"n_runs": n_runs},
    )


def gate_prc_01_precision_regime(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    device: str | torch.device = "cpu",
    **kwargs: Any,
) -> GateResult:
    """PRC-01: Precision regime — test FP32, FP16, BF16 with per-dtype tolerances.

    Per-dtype ``atol`` is interpreted at unit scale and multiplied by the
    reference output's magnitude (see CMP-01): cancellation elements carry
    the ULP noise of the large intermediates, not of their own small value.
    """
    dtype_configs: list[tuple[torch.dtype, float, float]] = [
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ]
    failures: list[str] = []
    for dtype, atol, rtol in dtype_configs:
        t = torch.randn(shape, dtype=dtype, device=device)
        try:
            out_ref = reference(t)
            out_cand = candidate(t)
        except Exception as exc:
            failures.append(f"{dtype}: exception — {exc}")
            continue
        if out_cand.shape != out_ref.shape:
            failures.append(f"{dtype}: shape mismatch")
            continue
        ref32 = out_ref.float()
        finite = ref32[torch.isfinite(ref32)]
        scale = max(1.0, finite.abs().max().item()) if finite.numel() else 1.0
        if not torch.allclose(
            out_cand.float(), ref32, atol=atol * scale, rtol=rtol, equal_nan=True
        ):
            max_err = (out_cand.float() - ref32).abs().max().item()
            failures.append(f"{dtype}: max_err={max_err:.3e} > atol={atol * scale:.3e}")

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
    device: str | torch.device = "cpu",
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
    t_nan = torch.randn(shape, dtype=dtype, device=device)
    t_nan.view(-1)[:: max(1, t_nan.numel() // 8)] = float("nan")
    _check_exceptional(t_nan, "scattered_nan")

    # Input with +Inf
    t_inf = torch.randn(shape, dtype=dtype, device=device)
    t_inf.view(-1)[:: max(1, t_inf.numel() // 8)] = float("inf")
    _check_exceptional(t_inf, "scattered_pos_inf")

    # Input with -Inf
    t_neginf = torch.randn(shape, dtype=dtype, device=device)
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
    device: str | torch.device = "cpu",
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
    t = torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
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
    device: str | torch.device = "cpu",
    atol: float = 0.0,
    rtol: float = 0.0,
    **kwargs: Any,
) -> GateResult:
    """ORD-03: Non-commutative reduction — candidate must match the reference
    on inputs designed to expose order-dependent reductions.

    Floating-point addition is not associative; the alternating
    large-positive / tiny / large-negative input makes reduction order
    matter materially: numerically unstable orderings (reversed
    accumulation, the cumprod-ratio scan trick, lost compensation terms)
    produce errors orders of magnitude above ULP noise here.

    Tolerances default to zero — bitwise identity, the right contract when
    the candidate can replicate the reference's exact operation order (CPU
    torch compositions). Hardware kernels reduce in trees and contract
    multiply-adds into FMAs, so bitwise identity against a torch eager
    reference is unachievable in principle; op harnesses override atol/rtol
    to a near-ULP budget instead (the C1 Triton scan measures ~2e-6 against
    the eager reference on B200 where unstable orderings err > 1e-2 — the
    adversarial input keeps the gate discriminative without bitwise).
    """
    pattern = torch.tensor([1.0, 1e-7, -1.0, 1e-7], dtype=dtype, device=device)
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

    bitwise = atol == 0.0 and rtol == 0.0
    if bitwise:
        ok = torch.equal(out_ref, out_cand)
    else:
        ok = torch.allclose(out_cand.float(), out_ref.float(), atol=atol, rtol=rtol)
    if not ok:
        max_err = (out_ref.float() - out_cand.float()).abs().max().item()
        kind = "bitwise-identical" if bitwise else f"within atol={atol:.1e}/rtol={rtol:.1e}"
        return GateResult(
            passed=False,
            reason=f"not {kind}: max_err={max_err:.3e} (reduction order differs)",
            details={"max_err": max_err, "atol": atol, "rtol": rtol},
        )
    return GateResult(
        passed=True,
        reason=(
            "bitwise-identical to reference on adversarial reduction input"
            if bitwise
            else "within tolerance on adversarial reduction input"
        ),
        details={"atol": atol, "rtol": rtol},
    )


def gate_prc_02_mixed_precision_accumulation(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (2, 32, 1024),
    device: str | torch.device = "cpu",
    atol: float = 2e-2,
    rtol: float = 0.0,
    scale_atol_by_ref_inf: bool = False,
    **kwargs: Any,
) -> GateResult:
    """PRC-02: Mixed-precision accumulation — FP16 inputs must produce outputs
    within FP16-ULP of the FP32 reference, implying an FP32 internal accumulator.

    A kernel that naively accumulates in FP16 across a long reduction loses
    precision proportionally to ``sqrt(N) * scale``. The default shape uses
    a 1024-element reduction dimension to make the gap between FP16-only
    accumulation and FP32-accumulating implementations visible.

    Comparing the FP16 candidate output (upcast to FP32) against the FP32
    reference output exposes the missing FP32 accumulator. The candidate's
    irreducible error includes the fp16 *input* rounding (the reference
    consumes unrounded operands by design), which is relative to local
    output magnitude — ops whose outputs span magnitudes (amplifying
    entries) need a small elementwise ``rtol`` so unit-scale ``atol``
    keeps the accumulator discrimination where outputs are O(1).

    ``scale_atol_by_ref_inf`` multiplies ``atol`` by ``max(1, |ref|_inf)``
    (the CMP-01 convention) for outputs whose error carries the magnitude
    of large *intermediates* even at elements that cancel to near zero —
    elementwise rtol cannot express that. Measured on B200 for the
    backward scan's grad_A view at (2, 1024, 32): honest fp32-accumulator
    floors are 4.9e-4 (Triton) / 9.0e-4 (eager) of output scale, an fp16
    carry sits at 1.25e-2 of scale, so a unit atol of 3e-3 keeps >3x
    margin on both sides. Default off: at unit output scales the flat
    interpretation is identical and stays bit-compatible with the C1
    calibration.
    """
    t_fp32 = torch.randn(shape, dtype=torch.float32, device=device)
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

    ref32 = out_ref.float()
    scale = 1.0
    if scale_atol_by_ref_inf:
        finite = ref32[torch.isfinite(ref32)]
        scale = max(1.0, finite.abs().max().item()) if finite.numel() else 1.0
    atol_eff = atol * scale
    cand32 = out_cand.float()
    diff = (cand32 - ref32).abs()
    max_err = diff.max().item()
    details: dict[str, Any] = {"max_err": max_err, "atol": atol, "rtol": rtol}
    if scale_atol_by_ref_inf:
        details["output_scale"] = scale
        details["atol_effective"] = atol_eff
    # Matching NaNs and bitwise-equal values (covers same-sign Inf) are ok —
    # the candidate is only charged where it diverges from the reference.
    within = (
        (diff <= atol_eff + rtol * ref32.abs())
        | (torch.isnan(cand32) & torch.isnan(ref32))
        | (cand32 == ref32)
    )
    ok = bool(within.all().item())
    if not ok:
        return GateResult(
            passed=False,
            reason=(
                f"max_err={max_err:.3e} exceeds atol={atol_eff:.3e} + rtol={rtol}*|ref| — "
                "likely missing FP32 accumulator"
            ),
            details=details,
        )
    return GateResult(
        passed=True,
        reason=f"max_err={max_err:.3e} within FP16 tolerance — FP32 accumulator inferred",
        details=details,
    )


def gate_exc_02_subnormal_handling(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    shape: tuple[int, ...] = (4, 64, 32),
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
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

    t = torch.full(shape, subnormal_min, dtype=dtype, device=device)
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
            failures.append(f"input device={t.device.type} → output device={out.device.type}")

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


# Conservative defaults when no device is queryable. 255 registers/thread is
# the architectural max across sm_70..sm_100; 227 KB dynamic shared memory per
# block covers H100 (228 KB opt-in) and B200; 512 TMEM elements is the sm_100
# budget from the #904 family ("Required: 544, Hardware limit: 512").
_DEFAULT_RESOURCE_LIMITS: dict[str, int] = {
    "n_regs": 255,
    "shared_bytes": 227 * 1024,
    "tmem_elems": 512,
}


def gate_res_02_resource_limits(
    candidate: Callable[..., torch.Tensor],
    reference: Callable[..., torch.Tensor],
    *,
    resource_meta: dict[str, int] | None = None,
    resource_limits: dict[str, int] | None = None,
    **kwargs: Any,
) -> GateResult:
    """RES-02: compiled-artifact resource usage must not exceed hardware limits.

    Evidence-based gate: the verifier's compile stage extracts per-kernel
    resource metadata (``n_regs`` and ``shared_bytes`` from the Triton
    CompiledKernel handle / ptxas ``--verbose``; ``tmem_elems`` from the
    OutOfResources diagnostics on Blackwell) and passes it here as
    ``resource_meta``. The gate compares each supplied field against the
    limit table.

    With no ``resource_meta`` the gate passes as not-applicable — a plain
    Python candidate has no compiled artifact to measure, and absence of
    evidence is not a violation. ``resource_limits`` overrides the defaults
    (use the actual device's queried properties when available).

    ``spill_bytes`` in the metadata is recorded in details as a warning but
    does not fail the gate: register spilling is legal-but-crippled (the
    num_warps=2 survivors in #904 spill 42-50 KB), and the perf consequence
    is the benchmark's job to expose, not this gate's.
    """
    if resource_meta is None:
        return GateResult(
            passed=True,
            reason="no resource metadata supplied; gate not applicable",
            details={"applicable": False},
        )

    limits = dict(_DEFAULT_RESOURCE_LIMITS)
    if resource_limits:
        limits.update(resource_limits)

    violations: list[str] = []
    checked: dict[str, tuple[int, int]] = {}
    for key, limit in limits.items():
        if key in resource_meta:
            used = int(resource_meta[key])
            checked[key] = (used, limit)
            if used > limit:
                violations.append(f"{key}: used {used} > limit {limit}")

    spill = int(resource_meta.get("spill_bytes", 0))

    if violations:
        return GateResult(
            passed=False,
            reason=f"{len(violations)} resource limit violation(s)",
            details={
                "applicable": True,
                "violations": violations,
                "checked": checked,
                "spill_bytes": spill,
            },
        )
    return GateResult(
        passed=True,
        reason=f"within limits on {len(checked)} measured resource(s)",
        details={
            "applicable": True,
            "checked": checked,
            "spill_bytes": spill,
            **({"warning": f"register spill of {spill} bytes"} if spill > 0 else {}),
        },
    )


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
    *,
    gate_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    gate_names: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, GateResult]:
    """Run Kernel Contract gates and return results keyed by gate name.

    ``kwargs`` (e.g. ``shape``, ``dtype``, ``device``) are forwarded to every
    gate uniformly; ``gate_overrides`` maps a gate name to kwargs applied to
    that gate only, on top of the shared ``kwargs`` (use it for op-specific
    shapes or to supply RES-02's ``resource_meta``).

    ``gate_names`` selects a subset (default: all 12). The audit harness uses
    this to exclude gates outside an audited corpus's claimed scope.

    Stubbed gates that raise ``NotImplementedError`` are recorded as
    ``passed=False`` with reason ``"not_implemented"``. Any other exception
    escaping a gate (e.g., a malformed candidate output crashes an internal
    comparison) is also caught and recorded as a failure — a misbehaving
    candidate must not crash the verifier loop.
    """
    results: dict[str, GateResult] = {}
    for name in gate_names if gate_names is not None else _ALL_GATE_NAMES:
        gate_fn = _GATE_MAP[name]
        gate_kwargs = dict(kwargs)
        if gate_overrides and name in gate_overrides:
            gate_kwargs.update(gate_overrides[name])
        try:
            results[name] = gate_fn(candidate, reference, **gate_kwargs)
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
