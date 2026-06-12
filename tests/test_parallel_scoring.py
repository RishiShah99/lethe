"""Parallel scorer pins: GPU leasing, order preservation, env pinning."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from flash_mamba_rl.rl import parallel_scoring
from flash_mamba_rl.rl.parallel_scoring import ParallelScorer
from flash_mamba_rl.verifier.sandbox import run_in_subprocess


def test_sandbox_extra_env_reaches_child() -> None:
    res = run_in_subprocess(
        "tests._sandbox_helpers",
        "env_echo",
        ("FMRL_TEST_PIN",),
        timeout_s=60.0,
        memory_limit_mb=0,
        extra_env={"FMRL_TEST_PIN": "7"},
    )
    assert res.success, res.stderr
    assert res.output == "7"


def test_sandbox_extra_env_overrides_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FMRL_TEST_PIN", "0,1,2,3")
    res = run_in_subprocess(
        "tests._sandbox_helpers",
        "env_echo",
        ("FMRL_TEST_PIN",),
        timeout_s=60.0,
        memory_limit_mb=0,
        extra_env={"FMRL_TEST_PIN": "5"},
    )
    assert res.success, res.stderr
    assert res.output == "5"


class TestParallelScorer:
    @pytest.fixture()
    def recording_scorer(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        record: dict[str, Any] = {"calls": [], "active": 0, "peak": 0}
        lock = threading.Lock()

        def fake_score(source: str, **kwargs: Any) -> dict[str, Any]:
            with lock:
                record["active"] += 1
                record["peak"] = max(record["peak"], record["active"])
                record["calls"].append((source, kwargs["extra_env"]["CUDA_VISIBLE_DEVICES"]))
            time.sleep(0.05)
            with lock:
                record["active"] -= 1
            return {"status": "scored", "reward": float(len(source)), "source_echo": source}

        monkeypatch.setattr(parallel_scoring, "score_candidate_source", fake_score)
        return record

    def test_results_align_with_input_order(self, recording_scorer: dict[str, Any]) -> None:
        scorer = ParallelScorer(op="forward_chunked_scan", gpu_ids=(4, 5, 6, 7))
        sources = [f"src_{i}" * (i + 1) for i in range(8)]
        results = scorer.score_batch(sources)
        assert [r["source_echo"] for r in results] == sources

    def test_concurrency_bounded_by_gpu_slots(self, recording_scorer: dict[str, Any]) -> None:
        scorer = ParallelScorer(op="forward_chunked_scan", gpu_ids=(4, 5))
        scorer.score_batch([f"s{i}" for i in range(6)])
        assert recording_scorer["peak"] <= 2

    def test_each_call_pinned_to_pool_gpu(self, recording_scorer: dict[str, Any]) -> None:
        scorer = ParallelScorer(op="forward_chunked_scan", gpu_ids=(4, 5, 6, 7))
        results = scorer.score_batch([f"s{i}" for i in range(8)])
        pinned = {env for _, env in recording_scorer["calls"]}
        assert pinned <= {"4", "5", "6", "7"}
        assert all(r["gpu_id"] in (4, 5, 6, 7) for r in results)

    def test_empty_batch(self, recording_scorer: dict[str, Any]) -> None:
        scorer = ParallelScorer(op="forward_chunked_scan", gpu_ids=(4,))
        assert scorer.score_batch([]) == []

    def test_no_gpus_rejected(self) -> None:
        with pytest.raises(ValueError):
            ParallelScorer(op="forward_chunked_scan", gpu_ids=())


def test_train_loop_uses_batch_scorer(tmp_path: Any) -> None:
    from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig

    from .test_train_loop import BAD, GOOD, StubTrainablePolicy

    seen_batches: list[list[str]] = []

    def batch_scorer(sources: list[str]) -> list[dict[str, Any]]:
        seen_batches.append(sources)
        return [
            {
                "status": "scored",
                "reward": 1.0 if "good" in s else 0.1,
                "compiled": True,
                "contracts_passed": "good" in s,
                "gates": {},
            }
            for s in sources
        ]

    policy = StubTrainablePolicy([GOOD, BAD])
    config = TrainLoopConfig(
        n_per_prompt=4, total_steps=1, device="cpu", checkpoint_dir=str(tmp_path / "ckpt")
    )
    loop = GRPOTrainingLoop(config, policy, batch_scorer=batch_scorer)
    metrics = loop.step()
    assert len(seen_batches) == 1
    assert len(seen_batches[0]) == 4
    assert metrics.mean_reward == pytest.approx((1.0 + 0.1 + 1.0 + 0.1) / 4)
