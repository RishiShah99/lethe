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
    """Batch scorer over scoring GPUs for the E2.c config track.

    Mirrors :class:`ParallelScorer`, but each candidate is a ``KernelConfig``
    JSON applied to the trusted kernel (:func:`score_config_candidate`) and
    timed at a fixed target ``shape`` — the field the source path lacks. Config
    generation is tiny, so scoring (a full gate battery + the speedup bench per
    candidate) is the step's bottleneck; this farms the K configs across GPUs.
    """

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
    """Batch scorer over scoring GPUs for the E2.f edit track.

    Mirrors :class:`ParallelConfigScorer`, but each candidate is a SEARCH/REPLACE
    edit applied to the ``base_variant`` kernel source and graded through the
    untrusted path (:func:`score_edit_candidate`) at a fixed target ``shape``.
    Generation is larger than a config but far smaller than full-source-gen;
    scoring (the gate battery + speedup bench per candidate) is still the
    bottleneck, so the K edits are farmed across GPUs.
    """

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
