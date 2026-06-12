"""CPU tests for the GRPO training loop with a stub trainable policy.

The stub policy carries a real torch parameter so the full update path —
advantages, loss, backward, clip, optimizer step — runs end to end and
parameter movement is observable. The scorer is stubbed to map source
content to rewards deterministically.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
import torch

from flash_mamba_rl.rl.train import (
    GRPOTrainingLoop,
    TrainLoopConfig,
    extract_code,
)

GOOD = "```python\ndef forward_chunked_scan(u):\n    return u  # good\n```"
BAD = "```python\ndef forward_chunked_scan(u):\n    return 0  # bad\n```"
NO_CODE = "I cannot write this kernel."


class StubTrainablePolicy:
    """Minimal TrainablePolicy: log-probs depend on a trainable parameter."""

    def __init__(self, completions: list[str]) -> None:
        self.theta = torch.nn.Parameter(torch.zeros(4))
        self._completions = completions
        self.saved_paths: list[str] = []
        self.eval_calls = 0

    def generate(self, prompt: str, n: int) -> list[str]:
        return [self._completions[i % len(self._completions)] for i in range(n)]

    def completion_log_probs(
        self, prompt: str, completions: list[str], *, use_adapter: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k, t = len(completions), self.theta.shape[0]
        base = -torch.nn.functional.softplus(self.theta).expand(k, t)
        if not use_adapter:
            base = base.detach() - 0.01
        mask = torch.ones(k, t, dtype=torch.bool)
        return base * mask, mask

    def trainable_parameters(self) -> Any:
        return iter([self.theta])

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.theta.detach(), os.path.join(path, "theta.pt"))
        self.saved_paths.append(path)

    def eval_mode(self) -> None:
        self.eval_calls += 1


def reward_scorer(source: str) -> dict[str, Any]:
    good = "good" in source
    return {
        "status": "scored",
        "reward": 1.0 if good else 0.1,
        "compiled": True,
        "contracts_passed": good,
        "gates": {},
    }


def make_loop(
    tmp_path: Any,
    completions: list[str],
    scorer: Any = reward_scorer,
    **overrides: Any,
) -> tuple[GRPOTrainingLoop, StubTrainablePolicy]:
    policy = StubTrainablePolicy(completions)
    config = TrainLoopConfig(
        n_per_prompt=4,
        total_steps=overrides.pop("total_steps", 3),
        checkpoint_dir=str(tmp_path / "ckpt"),
        device="cpu",
        **overrides,
    )
    return GRPOTrainingLoop(config, policy, scorer=scorer), policy


class TestExtractCode:
    def test_picks_block_with_entry_point(self) -> None:
        text = "```python\nx = 1\n```\nthen\n```python\ndef my_op(u):\n    pass\n```"
        assert extract_code(text, "my_op") == "def my_op(u):\n    pass\n"

    def test_falls_back_to_last_block(self) -> None:
        text = "```python\na = 1\n```\n```python\nb = 2\n```"
        assert extract_code(text, "my_op") == "b = 2\n"

    def test_no_block_returns_none(self) -> None:
        assert extract_code("no code here", "my_op") is None


class TestStep:
    def test_mixed_rewards_update_parameters(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [GOOD, BAD])
        before = policy.theta.detach().clone()
        metrics = loop.step()
        assert metrics.loss is not None
        assert metrics.mean_kl is not None and metrics.mean_kl >= 0.0
        assert metrics.grad_norm is not None
        assert not torch.equal(policy.theta.detach(), before)
        assert metrics.mean_reward == pytest.approx(0.55)
        assert metrics.max_reward == 1.0
        assert metrics.n_contracts_passed == 2
        assert policy.eval_calls == 1

    def test_degenerate_group_skips_update(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [BAD])
        before = policy.theta.detach().clone()
        metrics = loop.step()
        assert metrics.loss is None
        assert metrics.mean_kl is None
        assert torch.equal(policy.theta.detach(), before)
        assert loop.step_idx == 1

    def test_no_code_block_scores_zero(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [NO_CODE, GOOD])
        metrics = loop.step()
        assert metrics.n_no_code == 2
        assert metrics.mean_reward == pytest.approx((0.0 + 1.0 + 0.0 + 1.0) / 4)

    def test_rollout_and_metrics_jsonl_written(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD])
        loop.step()
        ckpt = tmp_path / "ckpt"
        rollout_rows = [json.loads(line) for line in (ckpt / "rollouts.jsonl").open()]
        metric_rows = [json.loads(line) for line in (ckpt / "metrics.jsonl").open()]
        assert len(rollout_rows) == 4
        assert all("source" in r for r in rollout_rows)
        assert len(metric_rows) == 1
        assert metric_rows[0]["step"] == 1

    def test_scorer_receives_extracted_source(self, tmp_path: Any) -> None:
        seen: list[str] = []

        def spy(source: str) -> dict[str, Any]:
            seen.append(source)
            return reward_scorer(source)

        loop, _ = make_loop(tmp_path, [GOOD], scorer=spy)
        loop.step()
        assert len(seen) == 4
        assert all(s.startswith("def forward_chunked_scan") for s in seen)


class TestRunAndCheckpoint:
    def test_run_reaches_total_steps_and_checkpoints(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [GOOD, BAD], total_steps=3)
        history = loop.run()
        assert len(history) == 3
        assert loop.step_idx == 3
        assert len(policy.saved_paths) == 3
        assert os.path.exists(tmp_path / "ckpt" / "trainer_state.pt")
        assert not os.path.exists(tmp_path / "ckpt" / "trainer_state.pt.tmp")

    def test_resume_restores_step_and_optimizer(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=5)
        loop.step()
        loop.step()
        loop.save_checkpoint()
        opt_state = loop.optimizer.state_dict()

        fresh, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=5)
        assert fresh.load_trainer_state()
        assert fresh.step_idx == 2
        restored = fresh.optimizer.state_dict()
        assert restored["state"].keys() == opt_state["state"].keys()
        for key in opt_state["state"]:
            torch.testing.assert_close(
                restored["state"][key]["exp_avg"], opt_state["state"][key]["exp_avg"]
            )

    def test_resume_without_checkpoint_returns_false(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD])
        assert not loop.load_trainer_state()

    def test_run_continues_from_restored_step(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=4)
        loop.step()
        loop.step()
        loop.save_checkpoint()

        resumed, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=4)
        resumed.load_trainer_state()
        history = resumed.run()
        assert len(history) == 2
        assert resumed.step_idx == 4
