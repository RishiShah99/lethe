"""Post-contract speedup measurement: candidate vs the hand-written op.

The reward table's speedup term (``t_handwritten / t_candidate``) is
measured here, only after every contract gate has passed; the staged
reward never pays performance for incorrect kernels. Inputs come from
the op-harness aux builders (deterministic, saturation-free; saturated
softplus entries probe value correctness, not throughput) at a fixed
training-ish bench shape; the baseline is the op's public hand-written
entry point, which dispatches to the Phase C Triton kernel on CUDA.

Inputs are rebuilt with varying content on every timed trial (see
``measure_speedup``): a fixed bench input lets a candidate that memoizes
on an input fingerprint or mutates a shared buffer in place serve cache
hits after the first trial and fabricate a speedup; per-trial rebuilds
force an honest recompute each trial.

Correctness is re-gated at the bench shape before any timing. The contract
gates top out at d_model=64; a candidate correct there but no-op / wrong
at the bench width (1024) would otherwise earn an unbounded fake speedup.
``measure_speedup`` first checks candidate-vs-baseline output equality at
the bench shape (scale-aware tolerance) and refuses to pay speedup on a
mismatch; the scoring bridge demotes such a candidate to the
contract-failure reward.

Bug routing: the official Mamba-3 Triton backward is the #904 casualty
on sm_100 (TMEM overflow at num_warps >= 4), so the backward-scan op
class carries the routing bonus on Blackwell; a contract-passing
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
    """Deterministic fp32 bench inputs for *op* at a [batch, seq_len, width] primary.

    ``width`` is d_model; the head-structured ops view it as
    (nheads, headdim) exactly as their gate views do. ``seed`` varies the
    primary tensor's content (every build still allocates fresh buffers for
    primary and aux): the timing loop sweeps it per trial so a memoizing or
    in-place-mutating candidate cannot serve cache hits across trials.
    """
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
    """Candidate matches the hand-written baseline at the bench shape.

    The contract gates cap d_model at 64; this re-checks value correctness
    at the bench width (1024) the speedup is measured at, so a shape-keyed
    no-op cannot bank a fake speedup. The tolerance is loose on purpose:
    fine numerics are the gates' job at small shapes; this only has to
    separate "actually computing the op" (error ~ eps*sqrt(L)*scale) from a
    gross divergence (no-op / wrong scale ~ full output magnitude), so an
    honest kernel clears it by orders of magnitude.
    """
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
    """Median-timing speedup ``t_baseline / t_candidate`` at the bench shape.

    Gates candidate correctness at the bench shape first (``correct_at_bench``
    in the result): on a mismatch the speedup is not measured and the scoring
    bridge withholds the speedup reward. Otherwise each timed trial draws
    fresh-content inputs (``build_bench_case(seed=...)``) so memoization or
    in-place mutation cannot fabricate the ratio; candidate and baseline see
    the identical per-trial inputs for a fair comparison. The candidate's
    output on every timed trial is captured (``output_sink``) and value-checked
    against the baseline afterwards, outside the timed window; the actual
    timed calls are verified, not a separate probe, so a candidate cannot no-op
    the trials the ratio is built from while computing correctly only for the
    inputs it can tell are being inspected (whether it fingerprints them by
    seed content or by call count). A candidate correct only on inputs the
    bench does not time is not paid for outputs it never actually produced.
    """
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

    # Verify the candidate's actual timed outputs against the baseline on the
    # identical per-trial inputs. Any timed trial the candidate no-op'd (to
    # collapse the median) produced a captured output that fails here, there
    # is no probe it can be correct at while cheating the trials that are timed.
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
