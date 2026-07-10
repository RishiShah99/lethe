"""GPU-parallel candidate scoring: one pinned sandbox per candidate."""

from __future__ import annotations

import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from lethe.kernels.autotune import ShapeSpec
from lethe.rl.config_grpo import score_config_candidate
from lethe.rl.edit_rl import score_edit_candidate
from lethe.verifier.candidate_scoring import (
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


@dataclass
class ParallelConfigScorer:
    """Batch scorer over scoring GPUs for the config-emission track."""

    op: str
    gpu_ids: tuple[int, ...]
    shape: ShapeSpec | None = None
    device: str = "cuda"
    timeout_s: float = 300.0
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES
    reward_shaping: str = "none"
    measure_speedup: bool = True
    workers_per_gpu: int = 1
    _slots: queue.Queue[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError("ParallelConfigScorer needs at least one GPU id")
        self._slots = queue.Queue()
        for _ in range(self.workers_per_gpu):
            for gpu in self.gpu_ids:
                self._slots.put(gpu)

    def _score_one(self, text: str) -> dict[str, Any]:
        gpu = self._slots.get()
        try:
            result = score_config_candidate(
                text,
                op=self.op,
                device=self.device,
                shape=self.shape,
                timeout_s=self.timeout_s,
                exclude_gates=self.exclude_gates,
                reward_shaping=self.reward_shaping,
                measure_speedup=self.measure_speedup,
                extra_env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        finally:
            self._slots.put(gpu)
        result["gpu_id"] = gpu
        return result

    def score_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Score *texts* concurrently; results align with the input order."""
        if not texts:
            return []
        n_workers = len(self.gpu_ids) * self.workers_per_gpu
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(self._score_one, texts))

    def __call__(self, texts: list[str]) -> list[dict[str, Any]]:
        return self.score_batch(texts)


@dataclass
class ParallelEditScorer:
    """Batch scorer over scoring GPUs for the edit-emission track."""

    op: str
    gpu_ids: tuple[int, ...]
    shape: ShapeSpec | None = None
    base_variant: str = "triton"
    device: str = "cuda"
    timeout_s: float = 300.0
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES
    reward_shaping: str = "none"
    measure_speedup: bool = True
    workers_per_gpu: int = 1
    _slots: queue.Queue[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.gpu_ids:
            raise ValueError("ParallelEditScorer needs at least one GPU id")
        self._slots = queue.Queue()
        for _ in range(self.workers_per_gpu):
            for gpu in self.gpu_ids:
                self._slots.put(gpu)

    def _score_one(self, text: str) -> dict[str, Any]:
        gpu = self._slots.get()
        try:
            result = score_edit_candidate(
                text,
                op=self.op,
                base_variant=self.base_variant,
                device=self.device,
                shape=self.shape,
                timeout_s=self.timeout_s,
                exclude_gates=self.exclude_gates,
                reward_shaping=self.reward_shaping,
                measure_speedup=self.measure_speedup,
                extra_env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        finally:
            self._slots.put(gpu)
        result["gpu_id"] = gpu
        return result

    def score_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Score *texts* concurrently; results align with the input order."""
        if not texts:
            return []
        n_workers = len(self.gpu_ids) * self.workers_per_gpu
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(self._score_one, texts))

    def __call__(self, texts: list[str]) -> list[dict[str, Any]]:
        return self.score_batch(texts)
