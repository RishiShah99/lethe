"""Bench-case construction and speedup measurement pins (CPU, tiny shapes)."""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from flash_mamba_rl.verifier.candidate_scoring import scoreable_ops
from flash_mamba_rl.verifier.op_bench import (
    BLACKWELL_BROKEN_OFFICIAL,
    bug_routing_active,
    build_bench_case,
    correct_at_bench_shape,
    measure_speedup,
)
from flash_mamba_rl.verifier.timing import benchmark

TINY = {"batch": 1, "seq_len": 64, "width": 8}


class TestBenchCase:
    @pytest.mark.parametrize("op", scoreable_ops())
    def test_baseline_runs_on_bench_inputs(self, op: str) -> None:
        case = build_bench_case(op, "cpu", **TINY)
        out = case.baseline(*case.args, **case.kwargs)
        if isinstance(out, torch.Tensor):
            assert torch.isfinite(out).all()
        else:
            assert len(out) > 1
            assert all(torch.isfinite(g).all() for g in out)

    def test_inputs_deterministic_across_builds(self) -> None:
        a = build_bench_case("forward_chunked_scan", "cpu", **TINY)
        b = build_bench_case("forward_chunked_scan", "cpu", **TINY)
        for ta, tb in zip(a.args, b.args, strict=True):
            assert torch.equal(ta, tb)

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(KeyError):
            build_bench_case("nonexistent_op", "cpu")


class TestMeasureSpeedup:
    def test_identity_candidate_near_parity(self) -> None:
        case = build_bench_case("elementwise_silu", "cpu", **TINY)
        result = measure_speedup(
            case.baseline, "elementwise_silu", "cpu", warmup=1, trials=5, **TINY
        )
        assert 0.1 < result["speedup"] < 10.0
        assert result["t_candidate_ms"] > 0.0
        assert result["t_baseline_ms"] > 0.0

    def test_slow_candidate_scores_below_one(self) -> None:
        def slow(x: torch.Tensor) -> torch.Tensor:
            time.sleep(0.01)
            return x * torch.sigmoid(x)

        result = measure_speedup(slow, "elementwise_silu", "cpu", warmup=1, trials=3, **TINY)
        assert result["speedup"] < 1.0


class TestPerTrialInputs:
    """C2: timing rebuilds inputs every trial so memoization can't fake speed."""

    def test_benchmark_rebuilds_inputs_per_trial(self) -> None:
        seen: list[float] = []

        def factory(i: int) -> tuple[torch.Tensor, ...]:
            return (torch.full((2, 2), float(i)),)

        def record(x: torch.Tensor) -> torch.Tensor:
            seen.append(float(x[0, 0]))
            return x

        benchmark(record, warmup=2, trials=4, inputs_factory=factory)
        assert seen[2:] == [0.0, 1.0, 2.0, 3.0]  # trials sweep distinct content
        assert len(set(seen)) == 6  # warm-up uses negative indices, disjoint from trials

    def test_benchmark_fixed_inputs_back_compat(self) -> None:
        calls = {"n": 0}

        def fn(x: torch.Tensor) -> torch.Tensor:
            calls["n"] += 1
            return x

        res = benchmark(fn, (torch.zeros(2),), warmup=1, trials=3)
        assert calls["n"] == 4  # 1 warm-up + 3 trials, fixed inputs
        assert res.n_trials == 3

    def test_content_memoizer_recomputes_under_varying_inputs(self) -> None:
        # A candidate that caches by an O(1) content fingerprint stays correct
        # under the gates (varying inputs) AND now under timing — it can never
        # serve a cache hit, because every trial's input content is new.
        cache: dict[float, torch.Tensor] = {}

        def memoizer(x: torch.Tensor) -> torch.Tensor:
            key = float(x.sum())
            if key not in cache:
                cache[key] = x * torch.sigmoid(x)
            return cache[key]

        measure_speedup(memoizer, "elementwise_silu", "cpu", warmup=1, trials=5, **TINY)
        assert len(cache) >= 5  # one fresh entry per trial — no reuse


class TestBenchShapeCorrectness:
    """C3: speedup is gated on candidate correctness at the bench width."""

    def test_shape_specific_noop_is_caught(self) -> None:
        def shape_cheat(x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] <= 8:  # correct at gate-scale widths
                return x * torch.sigmoid(x)
            return x  # no-op at the bench width

        r = measure_speedup(
            shape_cheat, "elementwise_silu", "cpu", warmup=1, trials=3, batch=1, seq_len=8, width=32
        )
        assert r["correct_at_bench"] is False
        assert r["speedup"] == 0.0

    def test_honest_candidate_passes_and_times(self) -> None:
        r = measure_speedup(
            lambda x: x * torch.sigmoid(x),
            "elementwise_silu",
            "cpu",
            warmup=1,
            trials=3,
            batch=1,
            seq_len=8,
            width=32,
        )
        assert r["correct_at_bench"] is True
        assert r["speedup"] > 0.0

    def test_correct_at_bench_shape_helper(self) -> None:
        assert correct_at_bench_shape(
            lambda x: x * torch.sigmoid(x), "elementwise_silu", "cpu", **TINY
        )
        assert not correct_at_bench_shape(lambda x: x, "elementwise_silu", "cpu", **TINY)


class TestBugRouting:
    def test_op_class_is_the_904_casualty_set(self) -> None:
        assert BLACKWELL_BROKEN_OFFICIAL == (
            "backward_selective_scan",
            "fused_block_backward",
        )

    def test_inactive_off_cuda(self) -> None:
        assert bug_routing_active("backward_selective_scan", "cpu") is False
        assert bug_routing_active("forward_chunked_scan", "cuda") is False


class TestScoringIntegration:
    """Speedup path through _score_source_body with the bench monkeypatched."""

    def test_speedup_reward_after_contracts(
        self, monkeypatch: pytest.MonkeyPatch, fake_single_op: dict[str, Any]
    ) -> None:
        import math

        from flash_mamba_rl.verifier import op_bench
        from flash_mamba_rl.verifier.candidate_scoring import _score_source_body

        monkeypatch.setattr(
            op_bench,
            "measure_speedup",
            lambda fn, op, device, **kw: {
                "t_candidate_ms": 1.0,
                "t_baseline_ms": 2.0,
                "speedup": 2.0,
            },
        )
        result = _score_source_body(
            fake_single_op["source"],
            {"op": "fake_fwd_op", "device": "cuda", "measure_speedup": True},
        )
        assert result["contracts_passed"] is True
        assert result["speedup"] == 2.0
        assert result["reward"] == pytest.approx(1.0 + math.log(2.0))
        assert result["bug_routing"] is False

    def test_bug_routing_bonus(
        self, monkeypatch: pytest.MonkeyPatch, fake_single_op: dict[str, Any]
    ) -> None:
        import math

        from flash_mamba_rl.verifier import op_bench
        from flash_mamba_rl.verifier.candidate_scoring import _score_source_body

        monkeypatch.setattr(
            op_bench,
            "measure_speedup",
            lambda fn, op, device, **kw: {
                "t_candidate_ms": 1.0,
                "t_baseline_ms": 2.0,
                "speedup": 2.0,
            },
        )
        monkeypatch.setattr(op_bench, "bug_routing_active", lambda op, device: True)
        result = _score_source_body(
            fake_single_op["source"],
            {"op": "fake_fwd_op", "device": "cuda", "measure_speedup": True},
        )
        assert result["bug_routing"] is True
        assert result["reward"] == pytest.approx(2.0 + math.log(2.0))

    def test_bench_shape_failure_demotes_to_contract_fail(
        self, monkeypatch: pytest.MonkeyPatch, fake_single_op: dict[str, Any]
    ) -> None:
        from flash_mamba_rl.verifier import op_bench
        from flash_mamba_rl.verifier.candidate_scoring import _score_source_body

        monkeypatch.setattr(
            op_bench,
            "measure_speedup",
            lambda fn, op, device, **kw: {
                "t_candidate_ms": float("nan"),
                "t_baseline_ms": float("nan"),
                "speedup": 0.0,
                "correct_at_bench": False,
            },
        )
        result = _score_source_body(
            fake_single_op["source"],
            {"op": "fake_fwd_op", "device": "cuda", "measure_speedup": True},
        )
        assert result["contracts_passed"] is False
        assert result["speedup"] is None
        assert result["reward"] == 0.1
        assert result["gates"]["bench_shape_correctness"]["passed"] is False

    def test_no_timing_off_cuda_or_unmeasured(self, fake_single_op: dict[str, Any]) -> None:
        from flash_mamba_rl.verifier.candidate_scoring import _score_source_body

        cpu = _score_source_body(
            fake_single_op["source"],
            {"op": "fake_fwd_op", "device": "cpu", "measure_speedup": True},
        )
        assert cpu["speedup"] is None
        assert cpu["reward"] == 0.5
        unmeasured = _score_source_body(
            fake_single_op["source"], {"op": "fake_fwd_op", "device": "cuda"}
        )
        assert unmeasured["speedup"] is None
        assert unmeasured["reward"] == 0.5


@pytest.fixture()
def fake_single_op(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from flash_mamba_rl.verifier import op_harness
    from flash_mamba_rl.verifier.candidate_scoring import _OP_VERIFIERS, OpSpec
    from flash_mamba_rl.verifier.contracts import GateResult

    def verify_fake(fn: Any, *, device: str = "cpu") -> dict[str, GateResult]:
        return {"gate_cmp_01_input_variation": GateResult(passed=True, reason="")}

    monkeypatch.setattr(op_harness, "verify_fake_fwd", verify_fake, raising=False)
    monkeypatch.setitem(_OP_VERIFIERS, "fake_fwd_op", OpSpec("fake_fwd", "verify_fake_fwd"))
    return {"source": "def fake_fwd(x):\n    return x\n"}
