"""Rollout data structures for GRPO over kernel-generation policies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """A single generated kernel candidate."""

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
    """A batch of scored candidates from a single prompt."""

    prompt: str
    prompt_id: str
    candidates: tuple[ScoredCandidate, ...]

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(c.reward for c in self.candidates)

    def advantages(self) -> tuple[float, ...]:
        """Group-relative advantages via the single ground-truth estimator."""
        rewards = self.rewards
        if len(rewards) < 2:
            return tuple(0.0 for _ in rewards)
        import torch

        from lethe.rl.grpo import compute_group_advantages

        advs = compute_group_advantages(torch.tensor(rewards, dtype=torch.float64))
        return tuple(advs.tolist())

    def best(self) -> ScoredCandidate:
        """Return the candidate with the highest reward (first wins on ties)."""
        if not self.candidates:
            raise ValueError("empty rollout has no best candidate")
        return max(self.candidates, key=lambda c: c.reward)
