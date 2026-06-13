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
    inputs: tuple[Any, ...] = (),
    *,
    warmup: int = 10,
    trials: int = 100,
    inputs_factory: Callable[[int], tuple[Any, ...]] | None = None,
) -> TimingResult:
    """Benchmark *callable* with positional *inputs*.

    Uses ``torch.cuda.Event`` for GPU timing when a CUDA device is available,
    otherwise falls back to ``time.perf_counter_ns`` for CPU-only hosts.

    Parameters
    ----------
    callable:
        The function to benchmark.  Called as ``callable(*inputs)``.
    inputs:
        Positional arguments forwarded to *callable* on every call. Ignored
        when ``inputs_factory`` is given.
    warmup:
        Number of warm-up calls before timing begins.
    trials:
        Number of timed calls.  Median (not best-of) is reported.
    inputs_factory:
        When supplied, fresh inputs are built per call via
        ``inputs_factory(i)`` *outside* the timed window — distinct content
        per trial. A fixed input tuple lets a candidate that memoizes on an
        input fingerprint, or mutates a shared buffer in place, fabricate a
        speedup by serving cache hits after the first trial; rebuilding with
        varying content forces an honest recompute every trial. Warm-up uses
        negative indices so it never primes a cache the timed trials hit.

    Returns
    -------
    TimingResult
        Frozen dataclass with median, std, min, max (all in milliseconds).
    """
    import torch  # local import keeps the module importable without torch on PATH

    use_cuda = torch.cuda.is_available()

    def call_inputs(i: int) -> tuple[Any, ...]:
        return inputs_factory(i) if inputs_factory is not None else inputs

    # ---- Warm-up ----
    for w in range(warmup):
        callable(*call_inputs(-1 - w))

    # ---- Timed trials ----
    times_ms: list[float] = []

    if use_cuda:
        torch.cuda.synchronize()
        for i in range(trials):
            args = call_inputs(i)  # built outside the timed window
            start_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end_event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start_event.record()
            callable(*args)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
    else:
        for i in range(trials):
            args = call_inputs(i)
            t0 = time.perf_counter_ns()
            callable(*args)
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
