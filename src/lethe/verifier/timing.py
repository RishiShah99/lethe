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
    output_sink: list[Any] | None = None,
) -> TimingResult:
    """Benchmark *callable* with positional *inputs*."""
    import torch  # local import keeps the module importable without torch on PATH

    def call_inputs(i: int) -> tuple[Any, ...]:
        return inputs_factory(i) if inputs_factory is not None else inputs

    # Choose the CUDA-event timer from tensors actually fed, not the host, or a CPU candidate times ~0.
    probe = call_inputs(-1)
    tensor_args = [a for a in probe if isinstance(a, torch.Tensor)]
    use_cuda = torch.cuda.is_available() and (
        not tensor_args or any(t.is_cuda for t in tensor_args)
    )

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
            out = callable(*args)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))
            if output_sink is not None:
                output_sink.append(out)
    else:
        for i in range(trials):
            args = call_inputs(i)
            t0 = time.perf_counter_ns()
            out = callable(*args)
            t1 = time.perf_counter_ns()
            times_ms.append((t1 - t0) / 1_000_000.0)
            if output_sink is not None:
                output_sink.append(out)

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
