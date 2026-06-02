"""Adversarial test: every cheating kernel must be rejected by the verifier.

For each deliberately-broken kernel in ``tests/cheating_kernels/``:

1. Run all 12 Kernel-Contract gates against the canonical reference.
2. Assert at least one *implemented* gate returns ``passed=False``.
3. Where the cheat targets a specific gate, assert that gate rejects it.

If a cheating kernel slips past every implemented gate, the verifier has
a hole — the test fails, the gate is hardened, the suite re-runs until
every cheat is caught.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from flash_mamba_rl.verifier.contracts import GateResult, run_all_gates
from tests.cheating_kernels import (
    dropout_in_disguise,
    fp32_only_correct,
    inf_propagation_bug,
    memoizes_first_input,
    nan_on_extremes,
    no_op_via_side_channel,
    nondeterministic,
    returns_cached,
    returns_input,
    shape_specific,
    wrong_dtype_promotion,
    wrong_shape_output,
)
from tests.cheating_kernels._reference import reference_op

# Names of the gates that are actually implemented — stubbed gates report
# ``passed=False`` with reason ``"not_implemented"``, which does not count
# as a real verifier rejection.
_IMPLEMENTED_GATES: tuple[str, ...] = (
    "gate_cmp_01_input_variation",
    "gate_cmp_03_shape_polymorphism",
    "gate_ord_01_reduction_order_tolerance",
    "gate_ord_02_atomic_determinism",
    "gate_prc_01_precision_regime",
    "gate_exc_01_exceptional_values",
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


# ---------------------------------------------------------------------------
# Individual cheating kernels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sanity: the reference itself must pass every implemented gate.
# ---------------------------------------------------------------------------


class TestReferencePassesAllGates:
    def test_reference_against_itself(self) -> None:
        results = run_all_gates(reference_op, reference_op)
        for name in _IMPLEMENTED_GATES:
            assert results[name].passed, (
                f"Reference failed its own gate {name}: {results[name].reason}"
            )
