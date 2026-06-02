"""GRPO trainer scaffolding.

This module defines the *interface* of the GRPO update step plus a
minimal trainer that wires the policy → rollout → reward → advantages
pipeline together. The actual gradient update is left for a
later phase — the loss computation lives in ``compute_grpo_loss``,
unimplemented and clearly marked, so the call site in ``step`` is
already correct and a future PR only needs to fill in the math.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from flash_mamba_rl.rl.policy import PolicyInterface
from flash_mamba_rl.rl.reward import score_callable
from flash_mamba_rl.rl.rollout import Candidate, Rollout, ScoredCandidate


@dataclass(frozen=True)
class GRPOConfig:
    """Hyperparameters for one GRPO training step.

    Defaults track the DeepSeek-Math GRPO paper unless noted.
    """

    n_per_prompt: int = 8
    kl_coef: float = 0.04
    clip_eps: float = 0.2
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0


def compute_grpo_loss(
    advantages: torch.Tensor,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    *,
    clip_eps: float,
    kl_coef: float,
) -> torch.Tensor:
    """GRPO policy-gradient loss with KL penalty against the reference policy.

    Not yet implemented — left as a stub so the trainer's call site is
    typed correctly and the math can land in a focused follow-up commit.
    """
    raise NotImplementedError("compute_grpo_loss not yet implemented")


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
    """Coordinator for one GRPO update step.

    Owns the policy and reference policy, drives candidate generation,
    scoring, and the (future) gradient update. The actual training
    optimiser is supplied by the caller so this class stays
    framework-agnostic.
    """

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
        """Construct a trainer.

        Parameters
        ----------
        config:
            Hyperparameters for the update step.
        policy:
            The policy being trained.
        ref_policy:
            Frozen reference policy used for the KL penalty.
        candidate_factory:
            ``(source, target_op, generation_id, candidate_idx) -> Candidate``.
            Wraps a raw generated string into a typed ``Candidate``.
        compile_fn:
            Boolean-returning compile hook (callers wire to
            ``flash_mamba_rl.verifier.compile.compile_kernel``).
        import_fn:
            Materialises a candidate into a callable. Wire to
            ``flash_mamba_rl.kernels.loader.import_candidate`` after
            writing the source to disk.
        reference_fn:
            The reference op the candidates are scored against.
        benchmark_fn:
            Optional timing hook; returns ``t_reference / t_candidate``
            (so >1 is "candidate is faster"). ``None`` skips speedup scoring.
        gate_kwargs:
            Per-gate overrides for shape / dtype / tolerance, forwarded to
            ``run_all_gates``.
        """
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
        """Generate, compile, score, and bundle ``n_per_prompt`` candidates.

        Failed compiles short-circuit to a zero-reward ``ScoredCandidate``
        without running any contract gates.
        """
        sources = self.policy.generate(prompt, self.config.n_per_prompt)
        scored: list[ScoredCandidate] = []
        for idx, source in enumerate(sources):
            candidate = self._candidate_factory(
                source, target_op, generation_id, idx
            )
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
                    bug_routing=False,  # set externally if compile detected C7907 on ref
                    gate_kwargs=self._gate_kwargs,
                )
            )
        return Rollout(
            prompt=prompt, prompt_id=prompt_id, candidates=tuple(scored)
        )

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
            n_contracts_passed=sum(
                1 for c in rollout.candidates if c.contracts_passed
            ),
            n_bug_routing=sum(1 for c in rollout.candidates if c.bug_routing),
        )
