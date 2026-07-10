"""Policy interface + a deterministic stub for tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PolicyInterface(Protocol):
    """The interface every GRPO-trainable policy must satisfy."""

    def generate(self, prompt: str, n: int) -> list[str]:
        """Sample ``n`` candidate completions for ``prompt``."""
        ...

    def log_probs(self, prompt: str, completion: str) -> list[float]:
        """Per-token log-probabilities of ``completion`` under this policy."""
        ...


class StubPolicy:
    """A canned-response policy."""

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
