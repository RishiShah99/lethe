"""Rollout data structures for GRPO over kernel-generation policies.

A *rollout* is a batch of candidates generated from one prompt, each
scored by the verifier. GRPO computes group-relative advantages over
this batch and uses them to update the policy.

These types are deliberately pure-data dataclasses with no model
dependencies — the trainer mutates them, the verifier scores them,
the policy generates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """A single generated kernel candidate.

    Attributes
    ----------
    source:
        Generated Triton source code (as a string).
    target_op:
        Name of the reference op this candidate implements (e.g.,
        ``"forward_chunked_scan"``).
    generation_id:
        RL step (or evaluation cycle) that produced this candidate.
    prompt_id:
        Identifier of the prompt that produced this candidate. All
        candidates in a Rollout share the same ``prompt_id``.
    """

    source: str
    target_op: str
    generation_id: int = 0
    prompt_id: str = ""


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate paired with its verifier scoring outcome."""

    candidate: Candidate
    reward: float
    compiled: bool
    contracts_passed: bool
    speedup_vs_handwritten: float | None
    bug_routing: bool
    gate_results: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Rollout:
    """A batch of scored candidates from a single prompt.

    Attributes
    ----------
    prompt:
        The exact prompt string fed to the policy.
    prompt_id:
        Stable identifier for the prompt across runs.
    candidates:
        Scored candidates sampled for this prompt.
    """

    prompt: str
    prompt_id: str
    candidates: tuple[ScoredCandidate, ...]

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(c.reward for c in self.candidates)

    def advantages(self) -> tuple[float, ...]:
        """Group-relative advantages via the single ground-truth estimator.

        Delegates to :func:`flash_mamba_rl.rl.grpo.compute_group_advantages` — the
        estimator the live optimizer step uses (``eps=1e-4``). A local
        reimplementation here previously carried a divergent ``eps=1e-8`` floor: a
        latent foot-gun that would resurrect behaviour the trainer tuned away for
        any future caller. Fewer than two candidates carry no relative signal, so
        advantages are zeroed. The import is deferred to keep this module
        dependency-free at import time and to avoid a rollout<->grpo cycle.
        """
        rewards = self.rewards
        if len(rewards) < 2:
            return tuple(0.0 for _ in rewards)
        import torch

        from flash_mamba_rl.rl.grpo import compute_group_advantages

        advs = compute_group_advantages(torch.tensor(rewards, dtype=torch.float64))
        return tuple(advs.tolist())

    def best(self) -> ScoredCandidate:
        """Return the candidate with the highest reward (first wins on ties)."""
        if not self.candidates:
            raise ValueError("empty rollout has no best candidate")
        return max(self.candidates, key=lambda c: c.reward)
