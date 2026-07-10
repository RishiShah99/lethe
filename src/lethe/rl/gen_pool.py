"""Data-parallel generation pool: K candidates across N policy replicas."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import torch


class _Replica(Protocol):
    """The replica surface the pool drives (real HFPolicy or a stub)."""

    last_terminated: list[bool]

    def generate(self, prompt: str, n: int) -> list[str]: ...

    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None: ...

    def eval_mode(self) -> None: ...


class _TrainerPolicy(Protocol):
    def adapter_state_dict(self) -> dict[str, torch.Tensor]: ...


def split_counts(n: int, parts: int) -> list[int]:
    """Distribute ``n`` items across ``parts`` as evenly as possible."""
    if parts <= 0:
        raise ValueError("parts must be positive")
    base, extra = divmod(n, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


class GenerationPool:
    """Fan K completions across N replicas; gather in stable prompt order."""

    def __init__(self, replicas: Sequence[_Replica]) -> None:
        if not replicas:
            raise ValueError("GenerationPool needs at least one replica")
        self._replicas = list(replicas)
        self.last_terminated: list[bool] = []

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        gen_devices: Sequence[int],
        *,
        sampling: Any = None,
        **policy_kwargs: Any,
    ) -> GenerationPool:
        """Load one inference HFPolicy replica per GPU id in ``gen_devices``."""
        from lethe.rl.hf_policy import HFPolicy

        replicas = [
            HFPolicy.from_pretrained(
                model_name,
                device_map={"": f"cuda:{dev}"},
                sampling=sampling,
                **policy_kwargs,
            )
            for dev in gen_devices
        ]
        for r in replicas:
            r.eval_mode()
        return cls(replicas)

    def refresh_from(self, trainer: _TrainerPolicy) -> None:
        """Copy the trainer's current LoRA tensors into every replica."""
        state = trainer.adapter_state_dict()
        for replica in self._replicas:
            replica.load_adapter_state_dict(state)

    def generate(self, prompt: str, n: int) -> list[str]:
        """Sample ``n`` completions, splitting across replicas concurrently."""
        if n <= 0:
            self.last_terminated = []
            return []
        counts = split_counts(n, len(self._replicas))
        active = [(rep, c) for rep, c in zip(self._replicas, counts, strict=True) if c > 0]

        def run(item: tuple[_Replica, int]) -> tuple[list[str], list[bool]]:
            replica, count = item
            out = replica.generate(prompt, count)
            term = list(getattr(replica, "last_terminated", [True] * len(out)))
            return out, term

        with ThreadPoolExecutor(max_workers=len(active)) as pool:
            results = list(pool.map(run, active))

        completions: list[str] = []
        terminated: list[bool] = []
        for out, term in results:
            completions.extend(out)
            terminated.extend(term)
        self.last_terminated = terminated
        return completions

    def eval_mode(self) -> None:
        for replica in self._replicas:
            replica.eval_mode()

    def __len__(self) -> int:
        return len(self._replicas)
