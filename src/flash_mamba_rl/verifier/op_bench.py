"""Post-contract speedup measurement: candidate vs the hand-written op.

The reward table's speedup term (``t_handwritten / t_candidate``) is
measured here, only after every contract gate has passed — the staged
reward never pays performance for incorrect kernels. Inputs come from
the op-harness aux builders (deterministic, saturation-free — saturated
softplus entries probe value correctness, not throughput) at a fixed
training-ish bench shape; the baseline is the op's public hand-written
entry point, which dispatches to the Phase C Triton kernel on CUDA.

Inputs are rebuilt between the candidate and baseline timings: the gate
battery regenerates aux per call, so an input-mutating candidate could
pass every gate and then poison a shared bench buffer.

Bug routing: the official Mamba-3 Triton backward is the #904 casualty
on sm_100 (TMEM overflow at num_warps >= 4), so the backward-scan op
class carries the routing bonus on Blackwell — a contract-passing
candidate that also beats the hand-written kernel there demonstrated a
working route around the broken upstream op (reward 2.0 + log speedup,
per ``reward.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from flash_mamba_rl.verifier.timing import benchmark

_BENCH_SEED = 7331

BLACKWELL_BROKEN_OFFICIAL: tuple[str, ...] = (
    "backward_selective_scan",
    "fused_block_backward",
)

DEFAULT_BATCH = 2
DEFAULT_SEQ_LEN = 2048
DEFAULT_WIDTH = 1024


@dataclass(frozen=True)
class BenchCase:
    """Positional args + kwargs for one op call, and its hand-written baseline."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    baseline: Callable[..., Any]


def _primary(shape: tuple[int, ...], device: str | torch.device) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_BENCH_SEED)
    return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)


def build_bench_case(
    op: str,
    device: str | torch.device,
    *,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    width: int = DEFAULT_WIDTH,
) -> BenchCase:
    """Deterministic fp32 bench inputs for *op* at a [batch, seq_len, width] primary.

    ``width`` is d_model; the head-structured ops view it as
    (nheads, headdim) exactly as their gate views do.
    """
    from flash_mamba_rl.kernels import ops as hand_ops
    from flash_mamba_rl.verifier import op_harness as oh

    dev = torch.device(device)
    fp32 = torch.float32

    if op == "elementwise_silu":
        return BenchCase(
            args=(_primary((batch, seq_len, width), dev),),
            kwargs={},
            baseline=lambda x: x * torch.sigmoid(x),
        )
    if op == "forward_chunked_scan":
        u = _primary((batch, seq_len, width), dev)
        aux = oh._scan_aux(batch, seq_len, width, oh.SCAN_N_STATE, dev, fp32, saturate=False)
        return BenchCase(
            args=(u, *aux),
            kwargs={"chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.forward_chunked_scan,
        )
    if op == "backward_selective_scan":
        inputs = oh._bwd_scan_aux(batch, seq_len, width, oh.SCAN_N_STATE, dev, fp32, saturate=False)
        dy = _primary((batch, seq_len, width), dev)
        return BenchCase(
            args=(*inputs, dy),
            kwargs={"chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.backward_selective_scan,
        )
    if op == "mimo_backward":
        nheads = oh._mimo_nheads(width, oh.MIMO_HEADDIM)
        mimo_aux = oh._mimo_bwd_aux(
            batch, seq_len, nheads, oh.MIMO_HEADDIM, oh.MIMO_RANK, oh.MIMO_N_STATE, dev, fp32
        )
        dy = _primary((batch, seq_len, nheads, oh.MIMO_HEADDIM), dev)
        return BenchCase(args=(*mimo_aux, dy), kwargs={}, baseline=hand_ops.mimo_backward)
    if op == "complex_scan_rope":
        nheads = oh._mimo_nheads(width, oh.ROPE_HEADDIM)
        rope_aux = oh._rope_aux(
            batch, seq_len, nheads, oh.ROPE_N_STATE, oh.ROPE_NUM_ANGLES, dev, fp32
        )
        x = _primary((batch, seq_len, nheads, oh.ROPE_HEADDIM), dev)
        return BenchCase(args=(x, *rope_aux), kwargs={}, baseline=hand_ops.complex_scan_rope)
    if op == "fused_block_forward":
        fused_aux = oh._fused_aux(
            batch, seq_len, width, oh.SCAN_N_STATE, oh.FUSED_CONV_K, dev, fp32, saturate=False
        )
        x = _primary((batch, seq_len, width), dev)
        x_pad = torch.nn.functional.pad(x, (0, 0, oh.FUSED_CONV_K - 1, 0))
        return BenchCase(
            args=(x_pad, *fused_aux),
            kwargs={"conv_kernel_size": oh.FUSED_CONV_K, "chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.fused_block_forward,
        )
    if op == "fused_block_backward":
        x, *fused_bwd_aux = oh._fused_bwd_aux(
            batch, seq_len, width, oh.SCAN_N_STATE, oh.FUSED_CONV_K, dev, fp32, saturate=False
        )
        x_pad = torch.nn.functional.pad(x, (0, 0, oh.FUSED_CONV_K - 1, 0))
        dy = _primary((batch, seq_len, width), dev)
        return BenchCase(
            args=(x_pad, *fused_bwd_aux, dy),
            kwargs={"conv_kernel_size": oh.FUSED_CONV_K, "chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.fused_block_backward,
        )
    raise KeyError(f"no bench case for op {op!r}")


def measure_speedup(
    candidate: Callable[..., Any],
    op: str,
    device: str | torch.device,
    *,
    warmup: int = 3,
    trials: int = 20,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    width: int = DEFAULT_WIDTH,
) -> dict[str, float]:
    """Median-timing speedup ``t_baseline / t_candidate`` at the bench shape."""

    def timed(fn: Callable[..., Any], case: BenchCase) -> float:
        call = (lambda *a: fn(*a, **case.kwargs)) if case.kwargs else fn
        return benchmark(call, case.args, warmup=warmup, trials=trials).median_ms

    t_candidate = timed(
        candidate, build_bench_case(op, device, batch=batch, seq_len=seq_len, width=width)
    )
    fresh = build_bench_case(op, device, batch=batch, seq_len=seq_len, width=width)
    t_baseline = timed(fresh.baseline, fresh)
    return {
        "t_candidate_ms": t_candidate,
        "t_baseline_ms": t_baseline,
        "speedup": t_baseline / t_candidate,
    }


def bug_routing_active(op: str, device: str | torch.device) -> bool:
    """Bonus eligibility: a #904-class op scored on a Blackwell (sm_100) device."""
    if op not in BLACKWELL_BROKEN_OFFICIAL:
        return False
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(dev) == (10, 0)
