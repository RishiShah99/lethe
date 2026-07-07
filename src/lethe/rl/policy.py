"""Policy interface + a deterministic stub for tests.

``PolicyInterface`` is the contract a real GRPO policy implementation
(an HF causal LM with a LoRA adapter, in our case) must satisfy.
``StubPolicy`` is a non-trainable canned-response policy used in
tests and scaffolding so the trainer's data flow can be exercised
without loading a real model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PolicyInterface(Protocol):
    """The interface every GRPO-trainable policy must satisfy.

    The verifier-driven trainer treats the policy as a black box that
    can generate candidate completions and report per-token log-probs.
    The actual model — HF base + LoRA adapter, possibly quantised —
    lives behind this interface.
    """

    def generate(self, prompt: str, n: int) -> list[str]:
        """Sample ``n`` candidate completions for ``prompt``."""
        ...

    def log_probs(self, prompt: str, completion: str) -> list[float]:
        """Per-token log-probabilities of ``completion`` under this policy.

        Returns one float per token in the completion (model-tokenised).
        """
        ...


class StubPolicy:
    """A canned-response policy. Use only in tests / scaffolding.

    The same fixed completions cycle through every ``generate`` call.
    Log-probs are all zero (placeholder).
    """

    def __init__(self, completions: Sequence[str] | None = None) -> None:
        self._completions: tuple[str, ...] = tuple(
            completions if completions is not None else ("def kernel(x):\n    return x\n",)
        )

    def generate(self, prompt: str, n: int) -> list[str]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return [self._completions[i % len(self._completions)] for i in range(n)]

    def log_probs(self, prompt: str, completion: str) -> list[float]:
        # One log-prob per whitespace-separated token, all zero.
        return [0.0 for _ in completion.split()]
