"""CPU tests for the warm-start SFT loop with a stub policy."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from lethe.rl.sft import SFTConfig, SFTExample, SFTTrainingLoop, build_sft_examples
from lethe.rl.sft_targets import available_targets, target_source, target_variants
from lethe.rl.train import _OP_ENTRY_POINTS, GRPOTrainingLoop, extract_code


class StubSFTPolicy:
    """Minimal SFTPolicy: per-token log-prob is -softplus(theta)."""

    def __init__(self) -> None:
        self.theta = torch.nn.Parameter(torch.ones(4))
        self.eval_calls = 0
        self.seen_prompts: list[str] = []
        self.seen_temperatures: list[float | None] = []
        self.saved_paths: list[str] = []

    def completion_log_probs(
        self,
        prompt: str,
        completions: list[str],
        *,
        use_adapter: bool = True,
        append_eos: bool | Sequence[bool] = True,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_prompts.append(prompt)
        self.seen_temperatures.append(temperature)
        k, t = len(completions), self.theta.shape[0]
        lp = -torch.nn.functional.softplus(self.theta).expand(k, t)
        mask = torch.ones(k, t, dtype=torch.bool)
        return lp * mask, mask

    def trainable_parameters(self) -> Any:
        return iter([self.theta])

    def save_adapter(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.theta.detach(), os.path.join(path, "theta.pt"))
        self.saved_paths.append(path)

    def eval_mode(self) -> None:
        self.eval_calls += 1


def make_loop(tmp_path: Any, **overrides: Any) -> tuple[SFTTrainingLoop, StubSFTPolicy]:
    policy = StubSFTPolicy()
    examples = [
        SFTExample(op=f"op{i}", prompt=f"prompt {i}", completion=f"```python\nx = {i}\n```")
        for i in range(3)
    ]
    config = SFTConfig(
        total_steps=overrides.pop("total_steps", 6),
        checkpoint_dir=str(tmp_path / "sft"),
        save_every=overrides.pop("save_every", 2),
        **overrides,
    )
    return SFTTrainingLoop(config, policy, examples), policy


class TestBuildExamples:
    def test_one_example_per_variant_with_exact_roundtrip(self) -> None:
        examples = build_sft_examples()
        assert {e.op for e in examples} == set(available_targets())
        assert len(examples) == sum(len(target_variants(op)) for op in available_targets())
        for e in examples:
            extracted = extract_code(e.completion, _OP_ENTRY_POINTS[e.op])
            assert extracted == target_source(e.op, e.variant)
            assert f"def {e.op}(" in e.completion
            assert e.prompt  # the real op prompt rides along

    def test_subset_selection(self) -> None:
        examples = build_sft_examples(["mimo_backward"], ["eager"])
        assert [(e.op, e.variant) for e in examples] == [("mimo_backward", "eager")]

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(KeyError):
            build_sft_examples(["not_an_op"])

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(KeyError):
            build_sft_examples(["mimo_backward"], ["cuda_cpp"])


class TestLoop:
    def test_loss_decreases_and_metrics_written(self, tmp_path: Any) -> None:
        loop, policy = make_loop(tmp_path, total_steps=8)
        history = loop.run()
        assert len(history) == 8
        assert history[-1].loss < history[0].loss
        assert policy.eval_calls == 8
        with open(os.path.join(loop.config.checkpoint_dir, "metrics.jsonl")) as f:
            rows = [json.loads(line) for line in f]
        assert [r["step"] for r in rows] == list(range(1, 9))
        assert all(r["tokens"] == 4 for r in rows)

    def test_epoch_order_covers_all_examples(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, total_steps=6)
        ops = [loop._example_for_step(i).op for i in range(6)]
        assert set(ops[:3]) == {"op0", "op1", "op2"}
        assert set(ops[3:]) == {"op0", "op1", "op2"}

    def test_example_order_is_pure_function_of_step(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path)
        first = [loop._example_for_step(i).op for i in range(9)]
        torch.manual_seed(1234)  # global RNG must not matter
        second = [loop._example_for_step(i).op for i in range(9)]
        assert first == second

    def test_checkpoint_commit_point_and_resume(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, total_steps=5, save_every=2)
        loop.run()
        ckpt_dir = loop.config.checkpoint_dir
        # Final partial-interval save included; latest adapter resolves.
        latest = GRPOTrainingLoop.latest_adapter_path(ckpt_dir)
        assert latest is not None and latest.endswith("adapter_step_5")

        fresh, _ = make_loop(tmp_path, total_steps=5, save_every=2)
        assert fresh.load_trainer_state() is True
        assert fresh.step_idx == 5
        assert fresh.optimizer.state_dict()["state"]  # optimizer state restored

    def test_resume_refuses_pickle_gadget(self, tmp_path: Any) -> None:
        # weights_only=True must reject a __reduce__ gadget resumed here (same pickle-RCE as GRPO).
        import pickle

        from tests._sandbox_helpers import ReduceBomb

        loop, _ = make_loop(tmp_path)
        os.makedirs(loop.config.checkpoint_dir, exist_ok=True)
        sentinel = os.path.join(str(tmp_path), "pwned")
        torch.save(
            {
                "step": 0,
                "adapter_name": "adapter_step_0",
                "optimizer": ReduceBomb(sentinel),
                "torch_rng": torch.get_rng_state(),
            },
            os.path.join(loop.config.checkpoint_dir, "trainer_state.pt"),
        )
        with pytest.raises(pickle.UnpicklingError):
            loop.load_trainer_state()
        assert not os.path.exists(sentinel)

    def test_prune_keeps_two_adapters(self, tmp_path: Any) -> None:
        loop, _ = make_loop(tmp_path, total_steps=6, save_every=1)
        loop.run()
        stamped = [
            d for d in os.listdir(loop.config.checkpoint_dir) if d.startswith("adapter_step_")
        ]
        assert sorted(stamped) == ["adapter_step_5", "adapter_step_6"]

    def test_empty_examples_rejected(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError):
            SFTTrainingLoop(SFTConfig(checkpoint_dir=str(tmp_path)), StubSFTPolicy(), [])

    def test_scores_at_unit_temperature(self, tmp_path: Any) -> None:
        # The NLL must be exact cross-entropy: every step scores at T=1.0, not the sampling temperature.
        loop, policy = make_loop(tmp_path, total_steps=3)
        loop.run()
        assert policy.seen_temperatures == [1.0, 1.0, 1.0]

    def test_nonpositive_temperature_rejected(self, tmp_path: Any) -> None:
        examples = [SFTExample(op="o", prompt="p", completion="```python\nx = 1\n```")]
        with pytest.raises(ValueError, match="temperature"):
            SFTTrainingLoop(
                SFTConfig(checkpoint_dir=str(tmp_path), temperature=0.0),
                StubSFTPolicy(),
                examples,
            )
