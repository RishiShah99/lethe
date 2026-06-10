"""Tests for the RL scaffolding (rollout types, reward bridge, GRPO skeleton).

These are interface / data-flow tests — no actual model loading or
gradient updates. The goal is to validate that a scored rollout
flows end-to-end through the GRPOTrainer without exceptions and
produces sensible summary metrics.
"""

from __future__ import annotations

import math

import pytest
import torch

from flash_mamba_rl.rl import (
    Candidate,
    GRPOConfig,
    GRPOTrainer,
    Rollout,
    ScoredCandidate,
    StubPolicy,
    compute_grpo_loss,
    score_callable,
)

# ---------------------------------------------------------------------------
# Rollout types
# ---------------------------------------------------------------------------


class TestRolloutAdvantages:
    def _make(self, rewards: list[float]) -> Rollout:
        candidates = tuple(
            ScoredCandidate(
                candidate=Candidate(source="", target_op="foo"),
                reward=r,
                compiled=True,
                contracts_passed=True,
                speedup_vs_handwritten=None,
                bug_routing=False,
            )
            for r in rewards
        )
        return Rollout(prompt="", prompt_id="x", candidates=candidates)

    def test_advantages_sum_to_zero(self) -> None:
        rollout = self._make([1.0, 2.0, 3.0, 4.0])
        advs = rollout.advantages()
        assert math.isclose(sum(advs), 0.0, abs_tol=1e-9)

    def test_advantages_normalised(self) -> None:
        rollout = self._make([1.0, 2.0, 3.0, 4.0])
        advs = rollout.advantages()
        # Variance of advantages should be ~1.
        mean = sum(advs) / len(advs)
        var = sum((a - mean) ** 2 for a in advs) / len(advs)
        assert math.isclose(var, 1.0, abs_tol=1e-6)

    def test_single_candidate_returns_zero_advantage(self) -> None:
        rollout = self._make([5.0])
        assert rollout.advantages() == (0.0,)

    def test_all_equal_rewards_return_zero_advantages(self) -> None:
        rollout = self._make([3.0, 3.0, 3.0])
        assert rollout.advantages() == (0.0, 0.0, 0.0)

    def test_best_returns_highest_reward(self) -> None:
        rollout = self._make([1.0, 5.0, 3.0])
        assert rollout.best().reward == 5.0


# ---------------------------------------------------------------------------
# StubPolicy
# ---------------------------------------------------------------------------


class TestStubPolicy:
    def test_generate_returns_n_items(self) -> None:
        policy = StubPolicy(completions=["a", "b"])
        out = policy.generate("prompt", 4)
        assert out == ["a", "b", "a", "b"]

    def test_generate_zero_returns_empty(self) -> None:
        policy = StubPolicy()
        assert policy.generate("prompt", 0) == []

    def test_negative_n_raises(self) -> None:
        policy = StubPolicy()
        with pytest.raises(ValueError):
            policy.generate("prompt", -1)


# ---------------------------------------------------------------------------
# score_callable
# ---------------------------------------------------------------------------


def _ref(t: torch.Tensor) -> torch.Tensor:
    return t.clone()


def _identity_kernel(t: torch.Tensor) -> torch.Tensor:
    return t.clone()


def _broken_kernel(t: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(t)


class TestScoreCallable:
    def _candidate(self) -> Candidate:
        return Candidate(source="", target_op="identity")

    def test_compiled_passing_no_speedup(self) -> None:
        sc = score_callable(
            self._candidate(),
            candidate_fn=_identity_kernel,
            reference_fn=_ref,
        )
        assert sc.compiled is True
        assert sc.contracts_passed is True
        assert sc.reward == pytest.approx(0.5)

    def test_compiled_failing_contracts(self) -> None:
        sc = score_callable(
            self._candidate(),
            candidate_fn=_broken_kernel,
            reference_fn=_ref,
        )
        assert sc.compiled is True
        assert sc.contracts_passed is False
        assert sc.reward == pytest.approx(0.1)

    def test_not_compiled_short_circuits(self) -> None:
        sc = score_callable(
            self._candidate(),
            candidate_fn=_identity_kernel,
            reference_fn=_ref,
            compiled=False,
        )
        assert sc.compiled is False
        assert sc.reward == 0.0
        assert sc.gate_results == {}

    def test_speedup_and_bug_routing_compose(self) -> None:
        sc = score_callable(
            self._candidate(),
            candidate_fn=_identity_kernel,
            reference_fn=_ref,
            speedup_vs_handwritten=math.e,  # log(speedup)=1
            bug_routing=True,
        )
        assert sc.reward == pytest.approx(2.0 + 1.0)


# ---------------------------------------------------------------------------
# GRPOTrainer end-to-end on stub policy
# ---------------------------------------------------------------------------


class TestGRPOTrainerSkeleton:
    def _trainer(
        self,
        *,
        compile_returns: bool = True,
        n_per_prompt: int = 4,
    ) -> GRPOTrainer:
        policy = StubPolicy(completions=["pass\n"])
        ref_policy = StubPolicy(completions=["pass\n"])
        config = GRPOConfig(n_per_prompt=n_per_prompt)

        def candidate_factory(source: str, target_op: str, gen_id: int, idx: int) -> Candidate:
            return Candidate(
                source=source,
                target_op=target_op,
                generation_id=gen_id,
                prompt_id=f"prompt_{idx}",
            )

        def compile_fn(src: str) -> bool:
            return compile_returns

        def import_fn(c: Candidate) -> object:
            return _identity_kernel

        return GRPOTrainer(
            config=config,
            policy=policy,
            ref_policy=ref_policy,
            candidate_factory=candidate_factory,
            compile_fn=compile_fn,
            import_fn=import_fn,
            reference_fn=_ref,
        )

    def test_rollout_produces_n_candidates(self) -> None:
        trainer = self._trainer(n_per_prompt=3)
        rollout = trainer.rollout(prompt="write a kernel", target_op="identity", prompt_id="p1")
        assert len(rollout.candidates) == 3
        assert all(c.compiled for c in rollout.candidates)
        assert all(c.contracts_passed for c in rollout.candidates)

    def test_rollout_zero_reward_when_compile_fails(self) -> None:
        trainer = self._trainer(compile_returns=False)
        rollout = trainer.rollout(prompt="x", target_op="identity")
        assert all(not c.compiled for c in rollout.candidates)
        assert all(c.reward == 0.0 for c in rollout.candidates)

    def test_summarise_metrics(self) -> None:
        trainer = self._trainer(n_per_prompt=4)
        rollout = trainer.rollout(prompt="x", target_op="identity")
        metrics = GRPOTrainer.summarise(rollout)
        assert metrics.n_compiled == 4
        assert metrics.n_contracts_passed == 4
        assert metrics.mean_reward == pytest.approx(0.5)
        assert metrics.max_reward == pytest.approx(0.5)
        assert metrics.loss is None


# ---------------------------------------------------------------------------
# Loss smoke test (stub has been replaced — full coverage in test_grpo_loss.py)
# ---------------------------------------------------------------------------


class TestComputeGrpoLoss:
    def test_returns_scalar_tensor(self) -> None:
        # K=4 sequences, T=3 tokens each.
        K, T = 4, 3
        adv = torch.zeros(K)
        lp = torch.zeros(K, T)
        loss = compute_grpo_loss(adv, lp, lp, lp, clip_eps=0.2, kl_coef=0.04)
        assert loss.shape == ()
        assert loss.item() == 0.0
