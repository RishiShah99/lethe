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

    def test_tmem_detected_on_success_autotune_masked_form(self) -> None:
        """rc=0 with TMEM strings on stderr = the autotune-masked #904 form.

        The reference kernel 'succeeds' on crippled num_warps=2 survivors
        while the overflowing configs print TMEM failures — the flag must be
        set even though success=True, or the perf-cliff form is invisible.
        """
        fake_stderr = (
            b"Autotuning failed with out of resource: tensor memory, "
            b"Required: 544, Hardware limit: 512.\n"
        )

        class _FakeProc:
            returncode = 0

            def communicate(
                self, input: bytes | None = None, timeout: float | None = None
            ) -> tuple[bytes, bytes]:
                return b"OK", fake_stderr

            def kill(self) -> None:
                pass

            def wait(self) -> None:
                pass

        with patch("subprocess.Popen", return_value=_FakeProc()):
            result = compile_kernel("x = 1\n")

        assert result.success is True
        assert result.error_class == ErrorClass.OK
        assert result.tmem_budget is True
        assert result.blackwell_failure is True

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


def _alias_input(t: torch.Tensor) -> torch.Tensor:
    """Returns its input storage verbatim — a view, never a fresh output buffer."""
    return t


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

    def test_default_reduction_elements_uses_sequence_extent(self) -> None:
        # (4, 512, 32) is (batch, seq, width); the scan reduction extent is the
        # seq dim (512), not the row width (32). The old shape[-1] default made
        # atol ~sqrt(512/32)=4x too tight and could false-reject honest scans.
        # Seed identically so both draw the same input → same output scale → the
        # atol ratio is exactly the sqrt(N) ratio, sqrt(512/32) = 4.
        torch.manual_seed(0)
        default = gate_ord_01_reduction_order_tolerance(_identity, _identity)
        torch.manual_seed(0)
        narrow = gate_ord_01_reduction_order_tolerance(_identity, _identity, reduction_elements=32)
        assert default.details["n_elements"] == 512
        assert default.details["atol_used"] == pytest.approx(narrow.details["atol_used"] * 4.0)


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

    def test_fail_when_output_aliases_input(self) -> None:
        # A candidate that returns a view of its input scores a fresh data_ptr
        # per call (each call gets a distinct input clone) so it slips the
        # cross-call distinctness check — the input-aliasing check catches it.
        result = gate_ord_02_atomic_determinism(_alias_input, _identity)
        assert result.passed is False
        assert "input" in result.reason


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

    def test_fail_when_candidate_mints_nan_matching_zero_mask(self) -> None:
        """Regression: NaN minted where the reference is non-zero used to pass.

        The zero-mask still agrees (NaN != 0) and ``max_err`` becomes NaN, which
        slips past ``NaN > atol == False``. The non-finite mask-agreement check
        must reject it.
        """

        def _nan_mint(t: torch.Tensor) -> torch.Tensor:
            return torch.where(t == 0, t, torch.full_like(t, float("nan")))

        result = gate_exc_02_subnormal_handling(_nan_mint, _identity)
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


class TestGateRes02ResourceLimits:
    def test_not_applicable_without_metadata(self) -> None:
        result = gate_res_02_resource_limits(_identity, _identity)
        assert result.passed is True
        assert result.details["applicable"] is False

    def test_within_limits_passes(self) -> None:
        meta = {"n_regs": 128, "shared_bytes": 100_000, "tmem_elems": 256}
        result = gate_res_02_resource_limits(_identity, _identity, resource_meta=meta)
        assert result.passed is True
        assert result.details["applicable"] is True
        assert len(result.details["checked"]) == 3

    def test_register_overflow_fails(self) -> None:
        result = gate_res_02_resource_limits(_identity, _identity, resource_meta={"n_regs": 300})
        assert result.passed is False
        assert "n_regs: used 300 > limit 255" in result.details["violations"]

    def test_tmem_budget_overflow_fails(self) -> None:
        # The #904 number: 544 required vs the 512-element sm_100 budget.
        result = gate_res_02_resource_limits(
            _identity, _identity, resource_meta={"tmem_elems": 544}
        )
        assert result.passed is False
        assert "tmem_elems: used 544 > limit 512" in result.details["violations"]

    def test_shared_memory_overflow_fails(self) -> None:
        result = gate_res_02_resource_limits(
            _identity, _identity, resource_meta={"shared_bytes": 300 * 1024}
        )
        assert result.passed is False

    def test_limit_override_respected(self) -> None:
        meta = {"shared_bytes": 100 * 1024}
        result = gate_res_02_resource_limits(
            _identity,
            _identity,
            resource_meta=meta,
            resource_limits={"shared_bytes": 64 * 1024},
        )
        assert result.passed is False

    def test_spill_warns_but_does_not_fail(self) -> None:
        # The #904 num_warps=2 survivors spill 42-50 KB: legal but crippled.
        result = gate_res_02_resource_limits(
            _identity,
            _identity,
            resource_meta={"n_regs": 32, "spill_bytes": 48 * 1024},
        )
        assert result.passed is True
        assert "warning" in result.details
        assert result.details["spill_bytes"] == 48 * 1024


# ---------------------------------------------------------------------------
# run_all_gates aggregate
# ---------------------------------------------------------------------------


class TestRunAllGates:
    def test_returns_12_entries(self) -> None:
        results = run_all_gates(_identity, _identity)
        assert len(results) == 12

    def test_all_12_gates_pass_for_identity(self) -> None:
        results = run_all_gates(_identity, _identity)
        for name, result in results.items():
            assert result.passed is True, f"{name} should pass for identity"

    def test_seeded_inputs_make_verdicts_deterministic(self) -> None:
        """Default per-gate seeding pins input-dependent gate details across
        independent runs, and the caller's global RNG state is restored."""

        def _wrong(x: torch.Tensor) -> torch.Tensor:
            return x + 0.1  # fails CMP-01's value comparison vs identity

        torch.manual_seed(0)
        before = torch.get_rng_state()
        first = run_all_gates(_wrong, _identity)
        after = torch.get_rng_state()
        second = run_all_gates(_wrong, _identity)

        assert torch.equal(before, after), "global RNG state must be restored"
        cmp01 = "gate_cmp_01_input_variation"
        assert not first[cmp01].passed
        assert first[cmp01].details == second[cmp01].details, "seeded draws must repeat"

    def test_seed_none_keeps_legacy_unseeded_draws(self) -> None:
        """seed=None leaves the global RNG advancing (no reseed / no restore) —
        the closed audit path's behavior is unchanged."""
        torch.manual_seed(0)
        before = torch.get_rng_state()
        run_all_gates(_identity, _identity, seed=None)
        after = torch.get_rng_state()
        assert not torch.equal(before, after), "unseeded gates advance the global RNG"


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


class TestSandboxStdoutShield:
    def test_candidate_stdout_does_not_corrupt_result(self) -> None:
        # A kernel that prints (Python and raw fd-1 writes) must not poison the
        # pickle channel — the result rides a private fd, fd 1 goes to stderr.
        result = run_in_subprocess(
            "tests._sandbox_helpers", "noisy_identity", (21,), timeout_s=30.0
        )
        assert result.success is True, result.stderr
        assert result.output == 42
        assert "raw fd-1 bytes" in result.stderr  # the noise was rerouted to stderr


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

    def test_compiled_passed_below_parity_ramps(self) -> None:
        # M2: the sub-parity band is a continuous ramp, not a flat 0.5.
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=0.8,
        )
        assert r == pytest.approx(1.0 + math.log(0.8))

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

    def test_speedup_exactly_one_gives_parity_anchor(self) -> None:
        # M2: parity scores 1.0 (the cliff at s=1 is gone), not 0.5.
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=1.0,
        )
        assert r == pytest.approx(1.0)

    def test_sub_parity_continuous_and_monotone(self) -> None:
        rs = [
            compute_reward(compiled=True, contracts_passed=True, speedup_vs_handwritten=s)
            for s in (0.7, 0.8, 0.9, 0.95, 0.999)
        ]
        assert rs == sorted(rs)  # monotone increasing toward parity
        assert all(0.5 < r < 1.0 for r in rs)  # above the floor, below parity
        # Continuous across parity: no 0.5-wide cliff at s=1.
        below = compute_reward(compiled=True, contracts_passed=True, speedup_vs_handwritten=0.999)
        above = compute_reward(compiled=True, contracts_passed=True, speedup_vs_handwritten=1.001)
        assert above - below == pytest.approx(math.log(1.001) - math.log(0.999), abs=1e-6)

    def test_very_slow_hits_floor(self) -> None:
        # Below e**-0.5 ≈ 0.61 the ramp saturates at the 0.5 floor — a correct
        # kernel always clears the 0.1 contract-fail level by a clear margin.
        for s in (0.6, 0.1, 0.007, 1e-9):
            r = compute_reward(compiled=True, contracts_passed=True, speedup_vs_handwritten=s)
            assert r == pytest.approx(0.5)

    def test_bug_routing_below_parity_pays_no_bonus(self) -> None:
        # The routing bonus is a genuine-beat signal: it only fires when faster.
        r = compute_reward(
            compiled=True,
            contracts_passed=True,
            speedup_vs_handwritten=0.9,
            bug_routing=True,
        )
        assert r == pytest.approx(1.0 + math.log(0.9))
