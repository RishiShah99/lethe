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
        self.last_terminated: list[bool] = []
        self.seen_append_eos: list[Any] = []

    def generate(self, prompt: str, n: int) -> list[str]:
        out = [self._completions[i % len(self._completions)] for i in range(n)]
        self.last_terminated = [True] * n
        return out

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: Any = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_append_eos.append(append_eos)
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
        n_per_prompt=overrides.pop("n_per_prompt", 4),
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

    def test_saturated_all_equal_group_skips_despite_fp32_std(self, tmp_path: Any) -> None:
        # Eight 0.1 rewards carry std ~7e-9 from fp32 mean rounding; a
        # float-std guard would issue a spurious uniform-negative update
        # (the toy-run review blocker). The exact-equality guard must skip.
        assert torch.tensor([0.1] * 8).std(correction=0).item() > 0.0
        loop, policy = make_loop(tmp_path, [BAD], n_per_prompt=8)
        before = policy.theta.detach().clone()
        metrics = loop.step()
        assert metrics.loss is None
        assert torch.equal(policy.theta.detach(), before)

    def test_termination_flags_forwarded_as_append_eos(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [GOOD, BAD])
        policy.last_terminated = []  # overwritten by generate inside step
        loop.step()
        flags = [False, True, False, True]
        policy.generate = lambda prompt, n: (  # type: ignore[method-assign]
            setattr(policy, "last_terminated", flags),
            [GOOD, BAD, GOOD, BAD],
        )[1]
        policy.seen_append_eos.clear()
        loop.step()
        assert policy.seen_append_eos == [flags, flags]  # ref + new streams

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

    def test_latest_adapter_path_names_committed_dir(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [GOOD, BAD])
        assert GRPOTrainingLoop.latest_adapter_path(str(tmp_path / "ckpt")) is None
        loop.step()
        loop.save_checkpoint()
        path = GRPOTrainingLoop.latest_adapter_path(str(tmp_path / "ckpt"))
        assert path is not None and path.endswith("adapter_step_1")
        assert path in policy.saved_paths

    def test_uncommitted_adapter_dir_never_referenced(self, tmp_path: Any) -> None:
        # A half-written adapter dir from a preempted save must not be
        # picked up: trainer_state.pt is the commit point.
        loop, _ = make_loop(tmp_path, [GOOD, BAD])
        loop.step()
        loop.save_checkpoint()
        os.makedirs(tmp_path / "ckpt" / "adapter_step_99")  # orphan, no commit
        path = GRPOTrainingLoop.latest_adapter_path(str(tmp_path / "ckpt"))
        assert path is not None and path.endswith("adapter_step_1")

    def test_resume_refuses_pickle_gadget(self, tmp_path: Any) -> None:
        # weights_only=True must reject a __reduce__ gadget planted in a resumed
        # trainer_state.pt (pickle-RCE on resume from a foreign run dir). Both the
        # resume loader and the static latest_adapter_path reader must refuse it.
        import pickle

        from tests._sandbox_helpers import ReduceBomb

        loop, _ = make_loop(tmp_path, [GOOD])
        ckpt_dir = str(tmp_path / "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        sentinel = os.path.join(str(tmp_path), "pwned")
        torch.save(
            {
                "step": 0,
                "adapter_name": "adapter_step_0",
                "optimizer": ReduceBomb(sentinel),
                "torch_rng": torch.get_rng_state(),
            },
            os.path.join(ckpt_dir, "trainer_state.pt"),
        )
        with pytest.raises(pickle.UnpicklingError):
            loop.load_trainer_state()
        with pytest.raises(pickle.UnpicklingError):
            GRPOTrainingLoop.latest_adapter_path(ckpt_dir)
        assert not os.path.exists(sentinel)

    def test_old_adapters_pruned_keeping_two(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=4)
        loop.run()
        stamped = sorted(d for d in os.listdir(tmp_path / "ckpt") if d.startswith("adapter_step_"))
        assert stamped == ["adapter_step_3", "adapter_step_4"]

    def test_rollout_step_field_matches_metrics(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD])
        loop.step()
        ckpt = tmp_path / "ckpt"
        rollout_rows = [json.loads(line) for line in (ckpt / "rollouts.jsonl").open()]
        metric_rows = [json.loads(line) for line in (ckpt / "metrics.jsonl").open()]
        assert {r["step"] for r in rollout_rows} == {metric_rows[0]["step"]} == {1}

    def test_final_checkpoint_saved_when_total_steps_not_divisible(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, [GOOD, BAD], total_steps=5, save_every=2)
        history = loop.run()
        assert len(history) == 5
        assert loop.step_idx == 5
        path = GRPOTrainingLoop.latest_adapter_path(str(tmp_path / "ckpt"))
        assert path is not None and path.endswith("adapter_step_5")
        state = torch.load(tmp_path / "ckpt" / "trainer_state.pt", weights_only=True)
        assert state["step"] == 5

    def test_no_double_save_when_total_steps_divisible(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, [GOOD, BAD], total_steps=4, save_every=2)
        loop.run()
        assert policy.saved_paths[-1].endswith("adapter_step_4")
        save_count = sum(1 for p in policy.saved_paths if p.endswith("adapter_step_4"))
        assert save_count == 1
