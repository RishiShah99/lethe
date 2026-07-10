"""Adversarial test: every cheating kernel must be rejected by the verifier."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from lethe.verifier.contracts import (
    GateResult,
    gate_ord_02_atomic_determinism,
    gate_prc_01_precision_regime,
    run_all_gates,
)
from tests.cheating_kernels import (
    buffer_aliasing,
    device_silent_move,
    dropout_in_disguise,
    fp16_accumulator,
    fp32_only_correct,
    grad_wrong_kernel,
    inf_propagation_bug,
    memoizes_first_input,
    nan_on_extremes,
    nan_on_subnormal,
    no_op_via_side_channel,
    nondeterministic,
    parallel_reduction_bug,
    returns_cached,
    returns_input,
    shape_specific,
    subnormal_flush_bug,
    wrong_dtype_promotion,
    wrong_shape_output,
)
from tests.cheating_kernels._reference import reference_op

# Implemented gates only; stubbed gates report passed=False reason="not_implemented".
_IMPLEMENTED_GATES: tuple[str, ...] = (
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
)


def _real_rejections(results: dict[str, GateResult]) -> dict[str, GateResult]:
    """Return implemented gates that genuinely rejected the candidate."""
    rejections: dict[str, GateResult] = {}
    for name in _IMPLEMENTED_GATES:
        result = results[name]
        if not result.passed and result.reason != "not_implemented":
            rejections[name] = result
    return rejections


def _reset_module_caches() -> None:
    """Clear any module-level memoisation in stateful cheating kernels."""
    memoizes_first_input._cache.clear()
    buffer_aliasing._buffer.clear()


@pytest.fixture(autouse=True)
def _seed_and_reset() -> None:
    torch.manual_seed(0)
    _reset_module_caches()


def _assert_caught_by_some_gate(
    cheat: Callable[[Tensor], Tensor],
    *,
    expected_gates: tuple[str, ...] = (),
) -> dict[str, GateResult]:
    """Run all gates; assert at least one implemented gate rejects."""
    results = run_all_gates(cheat, reference_op)
    rejections = _real_rejections(results)
    assert rejections, (
        "Verifier hole: no implemented gate rejected this cheating kernel.\n"
        f"All gate results: { {k: (v.passed, v.reason) for k, v in results.items()} }"
    )
    for gate_name in expected_gates:
        assert gate_name in rejections, (
            f"Expected {gate_name} to reject the cheat, but it returned "
            f"passed={results[gate_name].passed} reason={results[gate_name].reason!r}"
        )
    return results


class TestReturnsInput:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            returns_input.cheating_op,
            expected_gates=("gate_cmp_01_input_variation",),
        )


class TestReturnsCached:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            returns_cached.cheating_op,
            expected_gates=("gate_cmp_01_input_variation",),
        )


class TestNoOpViaSideChannel:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            no_op_via_side_channel.cheating_op,
            expected_gates=("gate_cmp_03_shape_polymorphism",),
        )


class TestMemoizesFirstInput:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            memoizes_first_input.cheating_op,
            expected_gates=("gate_cmp_01_input_variation",),
        )


class TestFp32OnlyCorrect:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            fp32_only_correct.cheating_op,
            expected_gates=("gate_prc_01_precision_regime",),
        )


class TestNanOnExtremes:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            nan_on_extremes.cheating_op,
            expected_gates=("gate_cmp_01_input_variation",),
        )


class TestShapeSpecific:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            shape_specific.cheating_op,
            expected_gates=("gate_cmp_03_shape_polymorphism",),
        )


class TestNondeterministic:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            nondeterministic.cheating_op,
            expected_gates=("gate_ord_02_atomic_determinism",),
        )


class TestBufferAliasing:
    def test_caught(self) -> None:
        # Value-correct everywhere; only violation is returning one aliased buffer across calls.
        _assert_caught_by_some_gate(
            buffer_aliasing.cheating_op,
            expected_gates=("gate_ord_02_atomic_determinism",),
        )


class TestWrongDtypePromotion:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            wrong_dtype_promotion.cheating_op,
            expected_gates=("gate_prc_01_precision_regime",),
        )


class TestInfPropagationBug:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            inf_propagation_bug.cheating_op,
            expected_gates=("gate_exc_01_exceptional_values",),
        )


class TestDropoutInDisguise:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            dropout_in_disguise.cheating_op,
            expected_gates=("gate_ord_02_atomic_determinism",),
        )


class TestWrongShapeOutput:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            wrong_shape_output.cheating_op,
            expected_gates=("gate_cmp_01_input_variation",),
        )


class TestGradWrongKernel:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            grad_wrong_kernel.cheating_op,
            expected_gates=("gate_cmp_02_gradient_correctness",),
        )


class TestParallelReductionBug:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            parallel_reduction_bug.cheating_op,
            expected_gates=("gate_ord_03_noncommutative_reduction",),
        )


class TestFp16Accumulator:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            fp16_accumulator.cheating_op,
            expected_gates=("gate_prc_02_mixed_precision_accumulation",),
        )


class TestSubnormalFlushBug:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            subnormal_flush_bug.cheating_op,
            expected_gates=("gate_exc_02_subnormal_handling",),
        )


class TestNanOnSubnormal:
    def test_caught(self) -> None:
        # Regression: NaN only on subnormal inputs used to slip past EXC-02's zero-mask check.
        _assert_caught_by_some_gate(
            nan_on_subnormal.cheating_op,
            expected_gates=("gate_exc_02_subnormal_handling",),
        )


class TestDeviceSilentMove:
    def test_caught(self) -> None:
        _assert_caught_by_some_gate(
            device_silent_move.cheating_op,
            expected_gates=("gate_res_01_memory_residency",),
        )


class TestReferencePassesAllGates:
    def test_reference_against_itself(self) -> None:
        results = run_all_gates(reference_op, reference_op)
        for name in _IMPLEMENTED_GATES:
            assert results[name].passed, (
                f"Reference failed its own gate {name}: {results[name].reason}"
            )


class TestOrd02AliasingBranch:
    def test_aliasing_branch_rejects_with_named_reason(self) -> None:
        # buffer_aliasing is value-correct and deterministic; only the aliasing branch can reject it.
        result = gate_ord_02_atomic_determinism(buffer_aliasing.cheating_op, reference_op, n_runs=5)
        assert not result.passed
        assert "aliasing" in result.reason.lower()
        assert result.details.get("distinct_buffers", 5) < 5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_aliasing_branch_rejects_on_cuda(self) -> None:
        # CUDA recycles freed addresses; the gate holds outputs alive so buffers stay distinct.
        result = gate_ord_02_atomic_determinism(
            buffer_aliasing.cheating_op, reference_op, n_runs=5, device="cuda"
        )
        assert not result.passed
        assert "aliasing" in result.reason.lower()


class TestPrc01Discrimination:
    def test_prc01_accepts_honest_rejects_fp32_only(self) -> None:
        # PRC-01 discrimination: honest reference passes, fp32-only cheat fails on a half dtype.
        honest = gate_prc_01_precision_regime(reference_op, reference_op)
        assert honest.passed, f"PRC-01 rejected the honest reference: {honest.reason}"

        cheat = gate_prc_01_precision_regime(fp32_only_correct.cheating_op, reference_op)
        assert not cheat.passed
        failures = " ".join(cheat.details.get("failures", []))
        assert "float16" in failures or "bfloat16" in failures, (
            f"PRC-01 rejection not attributed to a half dtype: {cheat.details}"
        )
