"""Curriculum schedule + runner pins: promotion gates, cap advance, resume."""

from __future__ import annotations

import json
import os
from typing import Any

from flash_mamba_rl.rl.curriculum import (
    DEFAULT_CURRICULUM,
    CurriculumConfig,
    CurriculumRunner,
    CurriculumSchedule,
)
from flash_mamba_rl.rl.prompts import available_ops
from flash_mamba_rl.rl.train import TrainLoopConfig
from flash_mamba_rl.verifier.candidate_scoring import scoreable_ops

from .test_train_loop import GOOD, StubTrainablePolicy


def two_level_config(**overrides: Any) -> CurriculumConfig:
    defaults: dict[str, Any] = {
        "ops": ("forward_chunked_scan", "backward_selective_scan"),
        "promote_threshold": 0.4,
        "promote_window": 2,
        "max_steps_per_level": 5,
    }
    defaults.update(overrides)
    return CurriculumConfig(**defaults)


class TestSchedule:
    def test_promotes_after_consecutive_window(self) -> None:
        sched = CurriculumSchedule(two_level_config())
        assert sched.current_op == "forward_chunked_scan"
        assert sched.record_step(0.5) is False
        assert sched.record_step(0.5) is True
        assert sched.levels[0].promoted is True
        assert sched.current_op == "backward_selective_scan"

    def test_below_threshold_resets_consecutive(self) -> None:
        sched = CurriculumSchedule(two_level_config())
        sched.record_step(0.5)
        sched.record_step(0.1)
        assert sched.levels[0].consecutive_at_threshold == 0
        sched.record_step(0.5)
        assert sched.record_step(0.5) is True
        assert sched.levels[0].promoted is True

    def test_cap_advances_without_promotion(self) -> None:
        sched = CurriculumSchedule(two_level_config())
        for _ in range(4):
            assert sched.record_step(0.1) is False
        assert sched.record_step(0.1) is True
        assert sched.levels[0].promoted is False
        assert sched.levels[0].closed is True
        assert sched.level_idx == 1

    def test_done_after_last_level(self) -> None:
        sched = CurriculumSchedule(two_level_config())
        for _ in range(2):
            sched.record_step(0.5)
        for _ in range(2):
            sched.record_step(0.5)
        assert sched.done is True

    def test_state_dict_round_trip(self) -> None:
        sched = CurriculumSchedule(two_level_config())
        sched.record_step(0.5, max_reward=1.2)
        state = sched.state_dict()
        fresh = CurriculumSchedule(two_level_config())
        fresh.load_state_dict(state)
        assert fresh.level_idx == 0
        assert fresh.levels[0].steps == 1
        assert fresh.levels[0].consecutive_at_threshold == 1
        assert fresh.levels[0].best_max_reward == 1.2

    def test_default_curriculum_ops_are_wired(self) -> None:
        for op in DEFAULT_CURRICULUM:
            assert op in available_ops()
            assert op in scoreable_ops()


def stub_scorer_factory(rewards_by_op: dict[str, float]) -> Any:
    def factory(op: str) -> Any:
        reward = rewards_by_op[op]

        def scorer(source: str) -> dict[str, Any]:
            bonus = 0.4 if "good" in source else 0.0
            return {
                "status": "scored",
                "reward": reward + bonus,
                "compiled": True,
                "contracts_passed": reward + bonus >= 0.5,
                "gates": {},
            }

        return scorer

    return factory


class TestRunner:
    def make_runner(self, tmp_path: Any, rewards_by_op: dict[str, float]) -> CurriculumRunner:
        policy = StubTrainablePolicy([GOOD, "no code here"])
        base = TrainLoopConfig(
            n_per_prompt=4,
            device="cpu",
            checkpoint_dir=str(tmp_path / "curr"),
        )
        return CurriculumRunner(
            base_config=base,
            policy=policy,
            curriculum=two_level_config(),
            scorer_factory=stub_scorer_factory(rewards_by_op),
        )

    def test_runs_all_levels_and_records(self, tmp_path: Any) -> None:
        runner = self.make_runner(
            tmp_path,
            {"forward_chunked_scan": 0.5, "backward_selective_scan": 0.0},
        )
        summary = runner.run()
        assert summary[0]["promoted"] is True
        assert summary[0]["steps"] == 2
        assert summary[1]["promoted"] is False
        assert summary[1]["steps"] == 5
        assert runner.schedule.done is True

    def test_writes_level_dirs_and_state(self, tmp_path: Any) -> None:
        runner = self.make_runner(
            tmp_path,
            {"forward_chunked_scan": 0.5, "backward_selective_scan": 0.5},
        )
        runner.run()
        assert os.path.isdir(runner.level_dir(0))
        assert os.path.isfile(os.path.join(runner.level_dir(0), "trainer_state.pt"))
        with open(runner.state_path, encoding="utf-8") as f:
            state = json.load(f)
        assert state["level_idx"] == 2
        assert all(lv["closed"] for lv in state["levels"])

    def test_resume_restores_schedule(self, tmp_path: Any) -> None:
        runner = self.make_runner(
            tmp_path,
            {"forward_chunked_scan": 0.5, "backward_selective_scan": 0.5},
        )
        runner.run()
        fresh = self.make_runner(
            tmp_path,
            {"forward_chunked_scan": 0.5, "backward_selective_scan": 0.5},
        )
        assert fresh.resume() is True
        assert fresh.schedule.done is True
        assert fresh.run() == fresh.schedule.summary()
