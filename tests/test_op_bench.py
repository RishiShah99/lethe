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
    measure_speedup,
)

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
