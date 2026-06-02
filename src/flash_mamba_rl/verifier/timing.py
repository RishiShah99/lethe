"""Kernel benchmarking utilities: wall-clock on CPU, CUDA Events on GPU."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimingResult:
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_trials: int


def benchmark(
    callable: Callable[..., Any],
    inputs: tuple[Any, ...],
    *,
    warmup: int = 10,
    trials: int = 100,
) -> TimingResult:
    """Benchmark *callable* with positional *inputs*.

    Uses ``torch.cuda.Event`` for GPU timing when a CUDA device is available,
    otherwise falls back to ``time.perf_counter_ns`` for CPU-only hosts.

    Parameters
    ----------
    callable:
        The function to benchmark.  Called as ``callable(*inputs)``.
    inputs:
        Positional arguments forwarded to *callable* on every call.
    warmup:
        Number of warm-up calls before timing begins.
    trials:
        Number of timed calls.  Median (not best-of) is reported.

    Returns
    -------
    TimingResult
        Frozen dataclass with median, std, min, max (all in milliseconds).
    """
    import torch  # local import keeps the module importable without torch on PATH

    use_cuda = torch.cuda.is_available()

    # ---- Warm-up ----
    for _ in range(warmup):
        callable(*inputs)

    # ---- Timed trials ----
    times_ms: list[float] = []

    if use_cuda:
        torch.cuda.synchronize()
        for _ in range(trials):
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start_event.record()
            callable(*inputs)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(trials):
            t0 = time.perf_counter_ns()
            callable(*inputs)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1_000_000.0)

    median_ms = statistics.median(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    return TimingResult(
        median_ms=median_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        n_trials=trials,
    )
