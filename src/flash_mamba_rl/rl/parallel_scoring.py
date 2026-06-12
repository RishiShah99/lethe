"""GPU-parallel candidate scoring: one pinned sandbox per candidate.

A GRPO step produces K candidate sources that are independent to score;
this farms them across a set of scoring GPUs. Each worker thread leases
one GPU id from a queue, runs the regular sandboxed
``score_candidate_source`` with ``CUDA_VISIBLE_DEVICES`` overridden to
that id (absolute — the override replaces any parent mask), and returns
the id to the pool. Inside the sandbox the device is therefore always
``cuda:0`` regardless of which physical GPU it landed on, so ``device``
stays ``"cuda"``.

Threads only marshal subprocess I/O — the GIL is irrelevant; concurrency
equals ``len(gpu_ids) * workers_per_gpu``. ``workers_per_gpu`` defaults
to 1: a scoring sandbox owns its GPU's memory while a candidate runs,
and two candidates sharing a device would perturb each other's timings
in the speedup stage.
"""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from flash_mamba_rl.verifier.candidate_scoring import (
    DEFAULT_EXCLUDE_GATES,
    score_candidate_source,
)


@dataclass
class ParallelScorer:
    """Callable batch scorer over a fixed set of scoring GPUs."""

    op: str
    gpu_ids: tuple[int, ...]
    device: str = "cuda"
    timeout_s: float = 420.0
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES
    fail_fast: bool = True
    reward_shaping: str = "none"
    measure_speedup: bool = False
    workers_per_gpu: int = 1
    _slots: queue.Queue[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError("ParallelScorer needs at least one GPU id")
        self._slots = queue.Queue()
        for _ in range(self.workers_per_gpu):
            for gpu in self.gpu_ids:
                self._slots.put(gpu)

    def _score_one(self, source: str) -> dict[str, Any]:
        gpu = self._slots.get()
        try:
            result = score_candidate_source(
                source,
                op=self.op,
                device=self.device,
                timeout_s=self.timeout_s,
                exclude_gates=self.exclude_gates,
                fail_fast=self.fail_fast,
                reward_shaping=self.reward_shaping,
                measure_speedup=self.measure_speedup,
                extra_env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        finally:
            self._slots.put(gpu)
        result["gpu_id"] = gpu
        return result

    def score_batch(self, sources: list[str]) -> list[dict[str, Any]]:
        """Score *sources* concurrently; results align with the input order."""
        if not sources:
            return []
        n_workers = len(self.gpu_ids) * self.workers_per_gpu
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(self._score_one, sources))

    def __call__(self, sources: list[str]) -> list[dict[str, Any]]:
        return self.score_batch(sources)
