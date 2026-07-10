"""GRPO trainer scaffolding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from lethe.rl.policy import PolicyInterface
from lethe.rl.reward import score_callable
from lethe.rl.rollout import Candidate, Rollout, ScoredCandidate


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for one GRPO training step."""

    n_per_prompt: int = 8
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute GRPO group-relative advantages for a single prompt's reward group."""
    mean = rewards.mean()
    std = rewards.std(correction=0)
    return (rewards - mean) / (std + eps)


def compute_grpo_loss(
    advantages: torch.Tensor,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    *,
    clip_eps: float,
    kl_coef: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """GRPO policy-gradient loss with PPO clipping and KL penalty."""
    # Defensive detach: gradient must flow only through new_log_probs.
    old_lp = old_log_probs.detach()
    ref_lp = ref_log_probs.detach()

    # --- ratio and PPO surrogate ----------------------------------------
    log_ratio = new_log_probs - old_lp

    # ratio shape: (K, T)
    ratio = torch.exp(log_ratio)

    # advantages broadcast: (K,) -> (K, 1) for element-wise multiply with (K, T)
    adv = advantages.unsqueeze(1)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)

    # --- Schulman k3 KL estimator ----------------------------------------
    d = ref_lp - new_log_probs
    kl = torch.exp(d) - d - 1.0

    # --- per-token objective ---------------------------------------------
    per_token_obj = surrogate - kl_coef * kl

    # --- masked mean over tokens per sequence ----------------------------
    if mask is None:
        float_mask = torch.ones_like(per_token_obj)
    else:
        float_mask = mask.to(dtype=per_token_obj.dtype)

    token_sums = (per_token_obj * float_mask).sum(dim=1)  # (K,)
    token_counts = float_mask.sum(dim=1).clamp(min=1.0)  # (K,)
    seq_means = token_sums / token_counts  # (K,)

    # --- mean over sequences and negate for minimisation ----------------
    loss = -seq_means.mean()
    return loss


@dataclass
class StepMetrics:
    """Per-step metrics returned by ``GRPOTrainer.step``."""

    mean_reward: float
    max_reward: float
    n_compiled: int
    n_contracts_passed: int
    n_bug_routing: int
    loss: float | None = None  # None until compute_grpo_loss is implemented


class GRPOTrainer:
    """Coordinator for one GRPO update step."""

    def __init__(
        self,
        config: GRPOConfig,
        policy: PolicyInterface,
        ref_policy: PolicyInterface,
        *,
        candidate_factory: Callable[[str, str, int, int], Candidate],
        compile_fn: Callable[[str], bool],
        import_fn: Callable[[Candidate], Callable[..., torch.Tensor]],
        reference_fn: Callable[..., torch.Tensor],
        benchmark_fn: Callable[[Callable[..., torch.Tensor]], float | None] | None = None,
        gate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Construct a trainer."""
        self.config = config
        self.policy = policy
        self.ref_policy = ref_policy
        self._candidate_factory = candidate_factory
        self._compile_fn = compile_fn
        self._import_fn = import_fn
        self._reference_fn = reference_fn
        self._benchmark_fn = benchmark_fn
        self._gate_kwargs = gate_kwargs

    def rollout(
        self,
        prompt: str,
        target_op: str,
        *,
        prompt_id: str = "",
        generation_id: int = 0,
    ) -> Rollout:
        """Generate, compile, score, and bundle ``n_per_prompt`` candidates."""
        sources = self.policy.generate(prompt, self.config.n_per_prompt)
        scored: list[ScoredCandidate] = []
        for idx, source in enumerate(sources):
            candidate = self._candidate_factory(source, target_op, generation_id, idx)
            compiled = self._compile_fn(candidate.source)
            if not compiled:
                scored.append(
                    score_callable(
                        candidate,
                        candidate_fn=lambda x: x,  # never called when compiled=False
                        reference_fn=self._reference_fn,
                        compiled=False,
                    )
                )
                continue

            candidate_fn = self._import_fn(candidate)
            speedup = self._benchmark_fn(candidate_fn) if self._benchmark_fn else None
            scored.append(
                score_callable(
                    candidate,
                    candidate_fn=candidate_fn,
                    reference_fn=self._reference_fn,
                    compiled=True,
                    speedup_vs_handwritten=speedup,
                    # set externally on CompileResult.blackwell_failure (C7907/TMEM budget)
                    bug_routing=False,
                    gate_kwargs=self._gate_kwargs,
                )
            )
        return Rollout(prompt=prompt, prompt_id=prompt_id, candidates=tuple(scored))

    @staticmethod
    def summarise(rollout: Rollout) -> StepMetrics:
        """Compute summary metrics over a scored rollout."""
        if not rollout.candidates:
            return StepMetrics(
                mean_reward=0.0,
                max_reward=0.0,
                n_compiled=0,
                n_contracts_passed=0,
                n_bug_routing=0,
            )
        rewards = rollout.rewards
        return StepMetrics(
            mean_reward=sum(rewards) / len(rewards),
            max_reward=max(rewards),
            n_compiled=sum(1 for c in rollout.candidates if c.compiled),
            n_contracts_passed=sum(1 for c in rollout.candidates if c.contracts_passed),
            n_bug_routing=sum(1 for c in rollout.candidates if c.bug_routing),
        )
