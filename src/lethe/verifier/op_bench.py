"""Post-contract speedup measurement: candidate vs the hand-written op."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lethe.verifier.timing import benchmark

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


def _primary(shape: tuple[int, ...], device: str | torch.device, seed: int = _BENCH_SEED) -> Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device=device, dtype=torch.float32)


def build_bench_case(
    op: str,
    device: str | torch.device,
    *,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    width: int = DEFAULT_WIDTH,
    seed: int = _BENCH_SEED,
) -> BenchCase:
    """Deterministic fp32 bench inputs for *op* at a [batch, seq_len, width] primary."""
    from lethe.kernels import ops as hand_ops
    from lethe.verifier import op_harness as oh

    dev = torch.device(device)
    fp32 = torch.float32

    if op == "elementwise_silu":
        return BenchCase(
            args=(_primary((batch, seq_len, width), dev, seed),),
            kwargs={},
            baseline=lambda x: x * torch.sigmoid(x),
        )
    if op == "forward_chunked_scan":
        u = _primary((batch, seq_len, width), dev, seed)
        aux = oh._scan_aux(batch, seq_len, width, oh.SCAN_N_STATE, dev, fp32, saturate=False)
        return BenchCase(
            args=(u, *aux),
            kwargs={"chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.forward_chunked_scan,
        )
    if op == "backward_selective_scan":
        inputs = oh._bwd_scan_aux(batch, seq_len, width, oh.SCAN_N_STATE, dev, fp32, saturate=False)
        dy = _primary((batch, seq_len, width), dev, seed)
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
        dy = _primary((batch, seq_len, nheads, oh.MIMO_HEADDIM), dev, seed)
        return BenchCase(args=(*mimo_aux, dy), kwargs={}, baseline=hand_ops.mimo_backward)
    if op == "complex_scan_rope":
        nheads = oh._mimo_nheads(width, oh.ROPE_HEADDIM)
        rope_aux = oh._rope_aux(
            batch, seq_len, nheads, oh.ROPE_N_STATE, oh.ROPE_NUM_ANGLES, dev, fp32
        )
        x = _primary((batch, seq_len, nheads, oh.ROPE_HEADDIM), dev, seed)
        return BenchCase(args=(x, *rope_aux), kwargs={}, baseline=hand_ops.complex_scan_rope)
    if op == "fused_block_forward":
        fused_aux = oh._fused_aux(
            batch, seq_len, width, oh.SCAN_N_STATE, oh.FUSED_CONV_K, dev, fp32, saturate=False
        )
        x = _primary((batch, seq_len, width), dev, seed)
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
        dy = _primary((batch, seq_len, width), dev, seed)
        return BenchCase(
            args=(x_pad, *fused_bwd_aux, dy),
            kwargs={"conv_kernel_size": oh.FUSED_CONV_K, "chunk_size": oh.SCAN_CHUNK_SIZE},
            baseline=hand_ops.fused_block_backward,
        )
    raise KeyError(f"no bench case for op {op!r}")


_CORRECTNESS_SEED_BASE = _BENCH_SEED + 90_001


def _tensor_close(out: Any, ref: Tensor, *, atol: float, rtol: float) -> bool:
    if not isinstance(out, Tensor) or out.shape != ref.shape:
        return False
    ref32 = ref.float()
    finite = ref32[torch.isfinite(ref32)]
    scale = max(1.0, finite.abs().max().item()) if finite.numel() else 1.0
    return bool(torch.allclose(out.float(), ref32, atol=atol * scale, rtol=rtol, equal_nan=True))


def _outputs_close(out: Any, ref: Any, *, atol: float, rtol: float) -> bool:
    """Scale-aware equality for a tensor output or a gradient tuple."""
    if isinstance(ref, Tensor):
        return _tensor_close(out, ref, atol=atol, rtol=rtol)
    try:
        ref_seq = list(ref)
        out_seq = list(out)
    except TypeError:
        return False
    if len(out_seq) != len(ref_seq):
        return False
    return all(
        _tensor_close(o, r, atol=atol, rtol=rtol) for o, r in zip(out_seq, ref_seq, strict=True)
    )


def correct_at_bench_shape(
    candidate: Callable[..., Any],
    op: str,
    device: str | torch.device,
    *,
    batch: int = DEFAULT_BATCH,
    seq_len: int = DEFAULT_SEQ_LEN,
    width: int = DEFAULT_WIDTH,
    n_checks: int = 2,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Candidate matches the hand-written baseline at the bench shape."""
    for i in range(n_checks):
        case = build_bench_case(
            op, device, batch=batch, seq_len=seq_len, width=width, seed=_CORRECTNESS_SEED_BASE + i
        )
        try:
            ref = case.baseline(*case.args, **case.kwargs)
            out = candidate(*case.args, **case.kwargs)
        except Exception:
            return False
        if not _outputs_close(out, ref, atol=atol, rtol=rtol):
            return False
    return True


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
) -> dict[str, float | bool]:
    """Median-timing speedup ``t_baseline / t_candidate`` at the bench shape."""
    failed: dict[str, float | bool] = {
        "t_candidate_ms": float("nan"),
        "t_baseline_ms": float("nan"),
        "speedup": 0.0,
        "correct_at_bench": False,
    }
    if not correct_at_bench_shape(candidate, op, device, batch=batch, seq_len=seq_len, width=width):
        return failed

    template = build_bench_case(op, device, batch=batch, seq_len=seq_len, width=width)
    kwargs = template.kwargs
    baseline = template.baseline

    def factory(i: int) -> tuple[Any, ...]:
        return build_bench_case(
            op, device, batch=batch, seq_len=seq_len, width=width, seed=_BENCH_SEED + i
        ).args

    def timed(fn: Callable[..., Any], sink: list[Any] | None = None) -> float:
        call = (lambda *a: fn(*a, **kwargs)) if kwargs else fn
        return benchmark(
            call, warmup=warmup, trials=trials, inputs_factory=factory, output_sink=sink
        ).median_ms

    cand_outputs: list[Any] = []
    try:
        t_candidate = timed(candidate, cand_outputs)
    except Exception:
        return failed

    # Verify the candidate's actual timed outputs against the baseline on the identical per-trial inputs.
    for i, out in enumerate(cand_outputs):
        try:
            ref = baseline(*factory(i), **kwargs)
        except Exception:
            return failed
        if not _outputs_close(out, ref, atol=1e-2, rtol=1e-2):
            return failed

    t_baseline = timed(baseline)
    return {
        "t_candidate_ms": t_candidate,
        "t_baseline_ms": t_baseline,
        "speedup": t_baseline / t_candidate,
        "correct_at_bench": True,
    }


def bug_routing_active(op: str, device: str | torch.device) -> bool:
    """Bonus eligibility: a #904-class op scored on a Blackwell (sm_100) device."""
    if op not in BLACKWELL_BROKEN_OFFICIAL:
        return False
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(dev) == (10, 0)
