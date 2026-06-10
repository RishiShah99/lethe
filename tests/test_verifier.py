"""Tests for the flash_mamba_rl verifier scaffolding.

All tests are CPU-safe: no CUDA, no Triton, no GPU required.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch

from flash_mamba_rl.verifier.compile import ErrorClass, compile_kernel
from flash_mamba_rl.verifier.contracts import (
    gate_cmp_01_input_variation,
    gate_cmp_02_gradient_correctness,
    gate_cmp_03_shape_polymorphism,
    gate_exc_01_exceptional_values,
    gate_exc_02_subnormal_handling,
    gate_ord_01_reduction_order_tolerance,
    gate_ord_02_atomic_determinism,
    gate_ord_03_noncommutative_reduction,
    gate_prc_01_precision_regime,
    gate_prc_02_mixed_precision_accumulation,
    gate_res_01_memory_residency,
    gate_res_02_resource_limits,
    run_all_gates,
)
from flash_mamba_rl.verifier.reward import compute_reward
from flash_mamba_rl.verifier.sandbox import run_in_subprocess
from flash_mamba_rl.verifier.timing import TimingResult, benchmark

# ---------------------------------------------------------------------------
# compile.py
# ---------------------------------------------------------------------------


class TestCompileKernel:
    def test_ok_with_valid_source(self) -> None:
        src = "x = 1 + 1\n"
        result = compile_kernel(src)
        assert result.success is True
        assert result.error_class == ErrorClass.OK
        assert result.compile_time_s >= 0.0

    def test_classifies_syntax_error(self) -> None:
        bad_src = "def foo(:\n    pass\n"
        result = compile_kernel(bad_src)
        assert result.success is False
        assert result.error_class == ErrorClass.SYNTAX

    def test_compile_result_is_frozen(self) -> None:
        result = compile_kernel("pass\n")
        with pytest.raises((AttributeError, TypeError)):
            result.success = False  # type: ignore[misc]

    def test_detects_c7907_pattern_in_stderr(self) -> None:
        """Monkeypatch subprocess to inject a fake C7907 stderr."""
        fake_stderr = b"ptxas fatal   : Internal error: C7907 encountered during codegen\n"

        class _FakeProc:
            returncode = 1

            def communicate(
                self, input: bytes | None = None, timeout: float | None = None
            ) -> tuple[bytes, bytes]:
                return b"", fake_stderr

            def kill(self) -> None:
                pass

            def wait(self) -> None:
                pass

        with patch("subprocess.Popen", return_value=_FakeProc()):
            result = compile_kernel("x = 1\n")

        assert result.ptxas_c7907 is True
        assert result.error_class == ErrorClass.PTXAS_C7907
        assert result.blackwell_failure is True

    def test_detects_tmem_budget_in_stderr(self) -> None:
        """The triton >= 3.7 surface form of the Blackwell TMEM failure.

        Fixture is VERBATIM from our B200 reproduction of
        state-spaces/mamba#904 (mamba3_siso_bwd_kernel_dqkv, num_warps=4):
        no C7907 string appears, so C7907-only detection would miss it.
        """
        fake_stderr = (
            b"Autotuning failed with out of resource: tensor memory, "
            b"Required: 544, Hardware limit: 512. Reducing block sizes or "
            b"`num_stages` may help.\n"
        )

        class _FakeProc:
            returncode = 1

            def communicate(
                self, input: bytes | None = None, timeout: float | None = None
            ) -> tuple[bytes, bytes]:
                return b"", fake_stderr

            def kill(self) -> None:
                pass

            def wait(self) -> None:
                pass

        with patch("subprocess.Popen", return_value=_FakeProc()):
            result = compile_kernel("x = 1\n")

        assert result.tmem_budget is True
        assert result.ptxas_c7907 is False  # the new form carries no C7907 string
        assert result.blackwell_failure is True
        assert result.error_class == ErrorClass.TMEM_BUDGET

    def test_tmem_outranks_generic_oom_classification(self) -> None:
        """'out of resource: tensor memory' must not be swallowed by OOM."""
        fake_stderr = (
            b"triton.runtime.errors.OutOfResources: out of resource: tensor memory, "
            b"Required: 576, Hardware limit: 512.\n"
        )

        class _FakeProc:
            returncode = 1

            def communicate(
                self, input: bytes | None = None, timeout: float | None = None
            ) -> tuple[bytes, bytes]:
                return b"", fake_stderr

            def kill(self) -> None:
                pass

            def wait(self) -> None:
                pass

        with patch("subprocess.Popen", return_value=_FakeProc()):
            result = compile_kernel("x = 1\n")

        assert result.error_class == ErrorClass.TMEM_BUDGET

    def test_detects_internal_compiler_error_in_stderr(self) -> None:
        """'internal compiler error' phrase also triggers PTXAS_C7907."""
        fake_stderr = b"ptxas: internal compiler error during register allocation\n"

        class _FakeProc:
            returncode = 1

            def communicate(
                self, input: bytes | None = None, timeout: float | None = None
            ) -> tuple[bytes, bytes]:
                return b"", fake_stderr

            def kill(self) -> None:
                pass

            def wait(self) -> None:
                pass

        with patch("subprocess.Popen", return_value=_FakeProc()):
            result = compile_kernel("x = 1\n")

        assert result.ptxas_c7907 is True
        assert result.error_class == ErrorClass.PTXAS_C7907


# ---------------------------------------------------------------------------
# Helpers shared by contract tests
# ---------------------------------------------------------------------------


def _identity(t: torch.Tensor) -> torch.Tensor:
    return t.clone()


def _scale_by_2(t: torch.Tensor) -> torch.Tensor:
    return t * 2.0


def _return_zeros(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


def _return_wrong_shape(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1)  # always 1-D, regardless of input


def _nondeterministic(t: torch.Tensor) -> torch.Tensor:
    """Adds random noise — not deterministic."""
    return t + torch.randn_like(t) * 0.1


# ---------------------------------------------------------------------------
# contracts.py — implemented gates (pass / fail)
# ---------------------------------------------------------------------------


class TestGateCmp01InputVariation:
    def test_pass_when_candidate_equals_reference(self) -> None:
        result = gate_cmp_01_input_variation(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_returns_wrong_values(self) -> None:
        result = gate_cmp_01_input_variation(_scale_by_2, _identity)
        assert result.passed is False

    def test_fail_when_candidate_returns_zeros(self) -> None:
        result = gate_cmp_01_input_variation(_return_zeros, _identity)
        assert result.passed is False

    def test_fail_when_candidate_returns_wrong_shape(self) -> None:
        result = gate_cmp_01_input_variation(_return_wrong_shape, _identity)
        assert result.passed is False

    def test_details_populated_on_failure(self) -> None:
        result = gate_cmp_01_input_variation(_scale_by_2, _identity)
        assert "failures" in result.details
        assert len(result.details["failures"]) > 0


class TestGateCmp03ShapePolymorphism:
    def test_pass_when_candidate_equals_reference(self) -> None:
        result = gate_cmp_03_shape_polymorphism(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_returns_wrong_values(self) -> None:
        result = gate_cmp_03_shape_polymorphism(_scale_by_2, _identity)
        assert result.passed is False

    def test_fail_when_candidate_wrong_shape(self) -> None:
        result = gate_cmp_03_shape_polymorphism(_return_wrong_shape, _identity)
        assert result.passed is False


class TestGateOrd01ReductionOrderTolerance:
    def test_pass_when_candidate_equals_reference(self) -> None:
        result = gate_ord_01_reduction_order_tolerance(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_returns_zeros(self) -> None:
        result = gate_ord_01_reduction_order_tolerance(_return_zeros, _identity)
        assert result.passed is False

    def test_details_contains_atol_and_n_elements(self) -> None:
        result = gate_ord_01_reduction_order_tolerance(_identity, _identity)
        assert "atol_used" in result.details
        assert "n_elements" in result.details
        assert result.details["atol_used"] > 0.0


class TestGateOrd02AtomicDeterminism:
    def test_pass_when_candidate_is_deterministic(self) -> None:
        result = gate_ord_02_atomic_determinism(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_is_nondeterministic(self) -> None:
        result = gate_ord_02_atomic_determinism(_nondeterministic, _identity)
        assert result.passed is False

    def test_n_runs_reported_in_details(self) -> None:
        result = gate_ord_02_atomic_determinism(_identity, _identity, n_runs=3)
        assert result.details["n_runs"] == 3


class TestGatePrc01PrecisionRegime:
    def test_pass_when_candidate_equals_reference(self) -> None:
        result = gate_prc_01_precision_regime(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_doubles_values(self) -> None:
        result = gate_prc_01_precision_regime(_scale_by_2, _identity)
        assert result.passed is False


class TestGateExc01ExceptionalValues:
    def test_pass_when_candidate_equals_reference(self) -> None:
        result = gate_exc_01_exceptional_values(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_strips_nans(self) -> None:
        """A candidate that replaces all values with 0 loses the NaN mask."""

        def _strip_nans(t: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

        result = gate_exc_01_exceptional_values(_strip_nans, _identity)
        assert result.passed is False

    def test_fail_when_candidate_propagates_nans_incorrectly(self) -> None:
        """A candidate that always returns NaN disagrees with the reference."""

        def _all_nan(t: torch.Tensor) -> torch.Tensor:
            return torch.full_like(t, float("nan"))

        result = gate_exc_01_exceptional_values(_all_nan, _identity)
        assert result.passed is False


# ---------------------------------------------------------------------------
# contracts.py — stubbed gates raise NotImplementedError
# ---------------------------------------------------------------------------


class TestGateCmp02GradientCorrectness:
    def test_pass_for_identity(self) -> None:
        result = gate_cmp_02_gradient_correctness(_identity, _identity)
        assert result.passed is True

    def test_fail_when_gradient_graph_broken(self) -> None:
        """A kernel that detaches inputs has zero gradient — gradcheck fails."""

        def _detached(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone().requires_grad_(False) + 0.0 * t

        result = gate_cmp_02_gradient_correctness(_detached, _identity)
        assert result.passed is False


class TestGateOrd03NoncommutativeReduction:
    def test_pass_for_identity(self) -> None:
        result = gate_ord_03_noncommutative_reduction(_identity, _identity)
        assert result.passed is True

    def test_fail_when_reduction_order_differs(self) -> None:
        """Candidate that does reverse-order sum differs bitwise from ref."""

        def _ref(t: torch.Tensor) -> torch.Tensor:
            return t.sum(dim=-1, keepdim=True)

        def _cand(t: torch.Tensor) -> torch.Tensor:
            return t.flip(-1).sum(dim=-1, keepdim=True)

        result = gate_ord_03_noncommutative_reduction(_cand, _ref)
        assert result.passed is False


class TestGatePrc02MixedPrecisionAccumulation:
    def test_pass_for_identity(self) -> None:
        result = gate_prc_02_mixed_precision_accumulation(_identity, _identity)
        assert result.passed is True

    def test_fail_for_fp16_accumulator(self) -> None:
        """A candidate that returns a noisy fp16 reduction blows the atol."""

        def _ref(t: torch.Tensor) -> torch.Tensor:
            return t.float().sum(dim=-1, keepdim=True)

        def _cand(t: torch.Tensor) -> torch.Tensor:
            # Manual fp16 loop accumulation — known precision loss.
            acc = torch.zeros(*t.shape[:-1], 1, dtype=t.dtype, device=t.device)
            for i in range(t.shape[-1]):
                acc = acc + t[..., i : i + 1]
            return acc

        result = gate_prc_02_mixed_precision_accumulation(_cand, _ref)
        assert result.passed is False


class TestGateExc02SubnormalHandling:
    def test_pass_for_identity(self) -> None:
        result = gate_exc_02_subnormal_handling(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_flushes_subnormals(self) -> None:
        """Candidate flushes subnormals; reference preserves them."""

        def _flush(t: torch.Tensor) -> torch.Tensor:
            tiny = torch.finfo(t.dtype).tiny
            return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

        result = gate_exc_02_subnormal_handling(_flush, _identity)
        assert result.passed is False


class TestGateRes01MemoryResidency:
    def test_pass_for_identity(self) -> None:
        result = gate_res_01_memory_residency(_identity, _identity)
        assert result.passed is True

    def test_fail_when_candidate_moves_device(self) -> None:
        def _to_meta(t: torch.Tensor) -> torch.Tensor:
            return t.to("meta")

        result = gate_res_01_memory_residency(_to_meta, _identity)
        assert result.passed is False


class TestStubbedGates:
    def test_gate_res_02_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            gate_res_02_resource_limits(_identity, _identity)


# ---------------------------------------------------------------------------
# run_all_gates aggregate
# ---------------------------------------------------------------------------


class TestRunAllGates:
    def test_returns_12_entries(self) -> None:
        results = run_all_gates(_identity, _identity)
        assert len(results) == 12

    def test_implemented_gates_pass_for_identity(self) -> None:
        results = run_all_gates(_identity, _identity)
        implemented = [
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
        ]
        for name in implemented:
            assert results[name].passed is True, f"{name} should pass for identity"

    def test_stubbed_gates_recorded_as_not_implemented(self) -> None:
        results = run_all_gates(_identity, _identity)
        stubbed = [
            "gate_res_02_resource_limits",
        ]
        for name in stubbed:
            assert results[name].passed is False
            assert results[name].reason == "not_implemented"


# ---------------------------------------------------------------------------
# timing.py
# ---------------------------------------------------------------------------


class TestTimingOnCpu:
    def test_benchmark_returns_timing_result(self) -> None:
        def _add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        t = torch.randn(64, 64)
        result = benchmark(_add_one, (t,), warmup=5, trials=20)
        assert isinstance(result, TimingResult)
        assert result.n_trials == 20

    def test_median_and_bounds_are_populated(self) -> None:
        def _add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        t = torch.randn(64, 64)
        result = benchmark(_add_one, (t,), warmup=2, trials=10)
        assert result.median_ms > 0.0
        assert result.min_ms <= result.median_ms <= result.max_ms

    def test_std_is_nonnegative(self) -> None:
        def _add_one(x: torch.Tensor) -> torch.Tensor:
            return x + 1

        t = torch.randn(16, 16)
        result = benchmark(_add_one, (t,), warmup=2, trials=5)
        assert result.std_ms >= 0.0

    def test_n_trials_matches_requested(self) -> None:
        result = benchmark(lambda x: x, (torch.tensor(1.0),), warmup=1, trials=7)
        assert result.n_trials == 7


# ---------------------------------------------------------------------------
# sandbox.py
# ---------------------------------------------------------------------------


class TestSandboxTimeout:
    @pytest.mark.slow
    def test_timeout_is_detected(self) -> None:
        """Spawn a subprocess that sleeps 10s; expect TIMEOUT within 2s."""
        import tempfile
        from pathlib import Path

        # Write a tiny helper module that sleeps
        sleep_src = "import time\ndef sleeper(): time.sleep(10)\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="fmrl_sleep_", delete=False
        ) as f:
            f.write(sleep_src)
            sleep_module_path = f.name

        try:
            result = run_in_subprocess(
                sleep_module_path,
                "sleeper",
                (),
                timeout_s=1.0,
            )
        finally:
            Path(sleep_module_path).unlink(missing_ok=True)

        assert result.success is False
        assert result.error_class == ErrorClass.TIMEOUT


# ---------------------------------------------------------------------------
# reward.py
# ---------------------------------------------------------------------------


class TestRewardBranches:
    def test_not_compiled_returns_zero(self) -> None:
        r = compute_reward(
            compiled=False,
            contracts_passed=False,
            speedup_vs_handwritten=None,
        )
        assert r == 0.0

    def test_compiled_but_failed_contracts(self) -> None:
        r = compute_reward(
            compiled=True,
            contracts_passed=False,
            speedup_vs_handwritten=2.0,
        )
        assert r == pytest.approx(0.1)

    def test_compiled_passed_no_speedup(self) -> None:
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=0.8,
        )
        assert r == pytest.approx(0.5)

    def test_compiled_passed_no_speedup_none(self) -> None:
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=None,
        )
        assert r == pytest.approx(0.5)

    def test_compiled_passed_faster(self) -> None:
        speedup = 2.0
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=speedup,
        )
        expected = 1.0 + math.log(speedup)
        assert r == pytest.approx(expected)

    def test_compiled_passed_faster_with_bug_routing(self) -> None:
        speedup = 2.0
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=speedup,
            bug_routing=True,
        )
        expected = 2.0 + math.log(speedup)
        assert r == pytest.approx(expected)

    def test_log_speedup_clipped_at_3(self) -> None:
        # e^4 > e^3, so clip should cap at 3.0
        speedup = math.exp(4.0)
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=speedup,
        )
        assert r == pytest.approx(1.0 + 3.0)

    def test_bug_routing_log_speedup_clipped_at_3(self) -> None:
        speedup = math.exp(10.0)
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=speedup,
            bug_routing=True,
        )
        assert r == pytest.approx(2.0 + 3.0)

    def test_speedup_exactly_one_gives_half(self) -> None:
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=1.0,
        )
        assert r == pytest.approx(0.5)
