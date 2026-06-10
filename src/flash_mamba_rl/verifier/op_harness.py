"""Adapters bridging multi-argument kernel ops to the single-tensor gates.

The 12 Kernel-Contract gates (``contracts.py``) drive candidates through a
single-tensor interface ``candidate(t) -> Tensor``; real ops take full
argument sets (``u, delta, A, B, C, D``). Per op this module provides:

- a **candidate adapter**: closes a scan-signature callable over auxiliary
  inputs derived deterministically from the primary tensor's shape. Aux
  tensors are generated on CPU from a fixed seed and then cast to the
  primary tensor's device/dtype, so the candidate and the reference consume
  bit-identical auxiliaries on any device, and repeated calls are identical
  (ORD-02 relies on this).
- a **reference adapter**: feeds the same auxiliaries to the fp32 reference
  oracle. Non-fp32 inputs are upcast to fp32 for the oracle and the output
  is rounded back once — this *defines* the mixed-precision contract the
  PRC gates measure candidates against: compute internally in fp32, round
  only at the output. (fp64 never reaches the reference adapter: the only
  fp64 gate, CMP-02 gradcheck, calls the candidate alone.)
- **gate overrides**: per-gate kwarg overrides tuning gate defaults to the
  op (e.g. PRC-02's accumulation stress belongs on the scan length L, not
  on the elementwise D axis its default shape implies).

Backward ops return gradient *tuples*, so they get one gate view per
gradient output (``BWD_GRAD_FIELDS``): the primary tensor the gates drive
is the upstream gradient ``dy`` — same [B, L, D] shape conventions, and
the gates' non-finite injections flow through ``dy`` exactly as the
forward gates flow them through ``u`` — while ``u`` joins the
deterministic auxiliaries. Each view runs the full 12-gate suite against
the autograd oracle's corresponding output, with per-view tolerance
overrides (``SCAN_BWD_GATE_OVERRIDES``): the accumulation extents differ
per output — grad_A sums over batch*L of L-long carry chains where
grad_u sees one chain — so one tolerance table cannot serve all six.

This is the same wiring the RL reward path needs to score generated
kernels per op; Phase D consumes it via ``score_candidate(gate_kwargs=...)``.

The verify drivers share one generic core (``_verify_op_views``): adapters
plus override table plus an optional saturation-free PRC-02 re-run. The
scan ops re-run PRC-02 because their aux plants softplus-saturation
entries; the MIMO backward takes ``dt``/``alpha`` precomputed — no
softplus exists inside the op, so its aux has no saturation analog and a
single PRC-02 run already measures the accumulator. MIMO's 4D primary
rides the gates' 3D [batch, seq, d_model] tensors by viewing d_model as
(nheads, MIMO_HEADDIM) — every gate d_model is divisible by 4. The rope
scan (C4) reuses the same 4D viewing for its forward primary ``x`` and,
like MIMO, runs PRC-02 once (no softplus in the op).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from flash_mamba_rl.kernels.references import (
    reference_backward_selective_scan,
    reference_forward_chunked_scan,
)
from flash_mamba_rl.kernels.references.complex_scan_rope import reference_complex_scan_rope
from flash_mamba_rl.kernels.references.mimo_backward import reference_mimo_backward
from flash_mamba_rl.verifier.contracts import (
    GateResult,
    gate_prc_02_mixed_precision_accumulation,
    run_all_gates,
)

ScanCallable = Callable[..., Tensor]
# Backward-scan signature: (u, delta, A, B, C, D, dy, *, chunk_size) -> the
# six gradients as an indexable sequence (SelectiveScanGrads or plain tuple).
BwdScanCallable = Callable[..., Any]

SCAN_N_STATE = 16
# Must divide every sequence length the gates use (CMP-02's default L=8 is
# the smallest); the scan result itself does not depend on chunking.
SCAN_CHUNK_SIZE = 8
_SCAN_AUX_SEED = 23117
_BWD_AUX_SEED = 9241

# Gradient outputs of the backward scan, in SelectiveScanGrads field order;
# each gets its own full gate run (one single-tensor view per output).
BWD_GRAD_FIELDS: tuple[str, ...] = ("grad_u", "grad_delta", "grad_A", "grad_B", "grad_C", "grad_D")

# Per-gate overrides appropriate for the scan op.
SCAN_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    # The gate's default (2, 32, 1024) puts the long axis on D, which the
    # scan treats elementwise. The op's accumulation axes are the sequence
    # (h carried over L) and the N-dot; a missing fp32 accumulator shows up
    # on a long scan, so stress L instead. At this shape with the Mamba-
    # realistic delta below (saturation off — see verify_scan_op), the
    # honest fp32-accumulator floor is ~6e-3 and an fp16 accumulator lands
    # at ~1.5e-1, so the gate's default atol=2e-2 separates them with
    # margin (pinned by a discriminative test).
    "gate_prc_02_mixed_precision_accumulation": {"shape": (2, 1024, 32)},
    # The scan's accumulation chain is the sequence, not the trailing dim:
    # at the gate's (4, 512, 32) shape the reduction extent is L=512.
    "gate_ord_01_reduction_order_tolerance": {"reduction_elements": 512},
    # Bitwise identity vs a torch eager reference is unachievable for a
    # tree-reducing, FMA-contracting hardware kernel (C1 measures ~2e-6 on
    # B200), so run with a tolerance — but at a length where unstable
    # orderings have actually diverged. The cumprod-ratio scan trick is
    # algebraically exact and stays within ~3e-5 of the oracle up to
    # L=4096 with this aux distribution; at L=8192 its decay products
    # underflow and it NaNs out, while honest reorder noise follows
    # eps*sqrt(L)*scale ~ 1e-4. atol=1e-3 sits 9x above honest noise and
    # rejects the collapse (pinned by a discriminative test).
    "gate_ord_03_noncommutative_reduction": {
        "shape": (1, 8192, 16),
        "atol": 1e-3,
        "rtol": 1e-3,
    },
}

# Official Mamba dt initialisation range (state-spaces/mamba,
# modules/mamba_simple.py): dt log-uniform in [dt_min, dt_max].
_DT_MIN = 1e-3
_DT_MAX = 1e-1


def _verify_op_views(
    candidate_factory: Callable[[bool], Callable[[Tensor], Tensor]],
    reference_factory: Callable[[bool], Callable[[Tensor], Tensor]],
    *,
    base_overrides: dict[str, dict[str, Any]],
    device: str | torch.device,
    resource_meta: dict[str, int] | None,
    saturation_rerun: bool,
) -> dict[str, GateResult]:
    """Generic gate run for one adapted view: overrides + optional PRC-02 re-run.

    Factories take ``saturate`` and return the gates' single-tensor callable;
    the re-run swaps in saturation-free aux (see ``verify_scan_op`` for why).
    """
    overrides: dict[str, dict[str, Any]] = {k: dict(v) for k, v in base_overrides.items()}
    if resource_meta is not None:
        overrides["gate_res_02_resource_limits"] = {"resource_meta": resource_meta}
    results = run_all_gates(
        candidate_factory(True),
        reference_factory(True),
        device=device,
        gate_overrides=overrides,
    )
    if not saturation_rerun:
        return results
    prc02_kwargs: dict[str, Any] = {
        "device": device,
        **base_overrides.get("gate_prc_02_mixed_precision_accumulation", {}),
    }
    try:
        results["gate_prc_02_mixed_precision_accumulation"] = (
            gate_prc_02_mixed_precision_accumulation(
                candidate_factory(False),
                reference_factory(False),
                **prc02_kwargs,
            )
        )
    except Exception as exc:  # mirror run_all_gates' crash containment
        results["gate_prc_02_mixed_precision_accumulation"] = GateResult(
            passed=False,
            reason=f"gate crashed: {type(exc).__name__}: {exc}",
            details={"exception_type": type(exc).__name__},
        )
    return results


def _aux_from_gen(
    gen: torch.Generator,
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    saturate: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU-side (delta, A, B, C, D_skip) drawn from ``gen`` — shared aux core.

    ``delta`` is drawn so that ``softplus(delta)`` is log-uniform in
    [1e-3, 1e-1] — the official Mamba dt-init distribution. This matters
    for gate power: small dt makes the scan a near-integrator with long
    memory, which is both the deployment-realistic regime and the one
    where low-precision accumulators actually lose mass (PRC-02). A
    unit-scale delta would make the scan strongly contractive and mask
    accumulator cheats.

    With ``saturate=True``, a sparse deterministic subset of ``delta``
    entries is set into the softplus saturation regime (12 / 25 / 95):
    an unguarded softplus (``log1p(exp(x))`` with no large-x branch)
    overflows to Inf at x ~ 89 in fp32 and would otherwise pass every
    gate, since the bulk distribution never leaves negative territory.
    PRC-02 runs with ``saturate=False``: the saturated channels amplify
    local magnitudes ~100x, so the candidate's irreducible fp16
    *input-rounding* error there swamps the accumulator signal the gate
    exists to measure (value correctness at saturation is CMP-01's job,
    on same-bits fp32 inputs). Aux tensors stay finite by construction;
    non-finite *aux* behaviour is outside the gate contract (the gates
    inject non-finites through the primary only) and is covered by the
    kernels' direct tests.
    """
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, d_model, generator=gen) * (log_hi - log_lo) + log_lo)
    delta = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
    if saturate:
        flat = delta.view(-1)
        saturation = torch.tensor([12.0, 25.0, 95.0])
        idx = torch.arange(0, flat.numel(), 257)
        flat[idx] = saturation.repeat((idx.numel() + 2) // 3)[: idx.numel()]
    a = -torch.rand(d_model, n_state, generator=gen)  # negative for stability
    b_proj = torch.randn(batch, seq_len, n_state, generator=gen)
    c_proj = torch.randn(batch, seq_len, n_state, generator=gen)
    d_skip = torch.randn(d_model, generator=gen)
    return delta, a, b_proj, c_proj, d_skip


def _scan_aux(
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
    saturate: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (delta, A, B, C, D_skip) for a [batch, seq_len, d_model] primary.

    Distribution rationale lives on ``_aux_from_gen``. Draw order under
    ``_SCAN_AUX_SEED`` is pinned: the measured cheat-vs-honest separations
    in the discriminative tests depend on these exact values.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SCAN_AUX_SEED)
    delta, a, b_proj, c_proj, d_skip = _aux_from_gen(
        gen, batch, seq_len, d_model, n_state, saturate
    )

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(delta), cast(a), cast(b_proj), cast(c_proj), cast(d_skip)


def _bwd_scan_aux(
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
    saturate: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (u, delta, A, B, C, D_skip) for a [batch, seq_len, d_model] ``dy``.

    The backward op's primary is the upstream gradient ``dy``, so the
    forward input ``u`` joins the auxiliaries: unit-normal, like the
    primaries the gates drive the forward op with. Same distribution
    rationale as ``_aux_from_gen``; a distinct seed keeps the backward
    aux decorrelated from any forward-harness primary.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_BWD_AUX_SEED)
    u = torch.randn(batch, seq_len, d_model, generator=gen)
    delta, a, b_proj, c_proj, d_skip = _aux_from_gen(
        gen, batch, seq_len, d_model, n_state, saturate
    )

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(u), cast(delta), cast(a), cast(b_proj), cast(c_proj), cast(d_skip)


def scan_candidate_adapter(
    scan_fn: ScanCallable,
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Wrap a scan-signature callable into the gates' single-tensor interface."""

    def adapted(u: Tensor) -> Tensor:
        batch, seq_len, d_model = u.shape
        aux = _scan_aux(batch, seq_len, d_model, n_state, u.device, u.dtype, saturate=saturate)
        return scan_fn(u, *aux, chunk_size=chunk_size)

    return adapted


def scan_reference_adapter(
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The reference oracle behind the same single-tensor interface."""

    def adapted(u: Tensor) -> Tensor:
        batch, seq_len, d_model = u.shape
        aux = _scan_aux(batch, seq_len, d_model, n_state, u.device, u.dtype, saturate=saturate)
        if u.dtype == torch.float32:
            return reference_forward_chunked_scan(u, *aux, chunk_size=chunk_size)
        # Mixed-precision contract: oracle computes in fp32 from the same
        # (already rounded) operand bits, rounds once at the output.
        y = reference_forward_chunked_scan(
            u.to(torch.float32),
            *(t.to(torch.float32) for t in aux),
            chunk_size=chunk_size,
        )
        return y.to(u.dtype)

    return adapted


def verify_scan_op(
    scan_fn: ScanCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over a scan-signature callable.

    ``resource_meta`` (from compile-stage extraction or the Triton kernel
    cache) activates RES-02's evidence-based check; None leaves it
    not-applicable.

    PRC-02 is re-run with saturation-free auxiliaries: the saturated
    delta entries probe softplus value-correctness (CMP-01's same-bits
    domain) but their ~100x amplification drowns PRC-02's accumulator
    signal in fp16 input-rounding noise (see ``_scan_aux``).
    """
    return _verify_op_views(
        lambda saturate: scan_candidate_adapter(
            scan_fn, n_state=n_state, chunk_size=chunk_size, saturate=saturate
        ),
        lambda saturate: scan_reference_adapter(
            n_state=n_state, chunk_size=chunk_size, saturate=saturate
        ),
        base_overrides=SCAN_GATE_OVERRIDES,
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=True,
    )


# ---------------------------------------------------------------------------
# Backward scan: one gate view per gradient output
# ---------------------------------------------------------------------------


def _bwd_view_overrides(
    *,
    ord01_reduction_elements: int,
    ord03_atol: float,
    ord03_rtol: float,
    prc02: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-view gate overrides sharing the backward op's fixed shapes.

    PRC-02 keeps the forward's L-axis stress shape and adds rtol=1e-3:
    gradient magnitudes span decades across outputs, so the fp16
    *input-rounding* floor is relative to local output scale exactly as
    the gate docstring anticipates — while the fp16-accumulator signal
    sits ~30x above that rtol at this shape's sqrt(L)*eps16 error.
    ``prc02`` replaces those kwargs for views where the flat model is
    wrong (grad_A, below).

    ORD-01's ``reduction_elements`` is the view's honest accumulation
    extent under the eps*sqrt(chain)*scale random-walk model (C1's
    calibration approach). ORD-03 keeps the forward's L=8192 collapse
    length. ORD tolerances are theory-seeded; the B200 suite passes them
    as-is (C1 measured within 1.1x of the model).
    """
    return {
        "gate_prc_02_mixed_precision_accumulation": (
            prc02 if prc02 is not None else {"shape": (2, 1024, 32), "rtol": 1e-3}
        ),
        "gate_ord_01_reduction_order_tolerance": {"reduction_elements": ord01_reduction_elements},
        "gate_ord_03_noncommutative_reduction": {
            "shape": (1, 8192, 16),
            "atol": ord03_atol,
            "rtol": ord03_rtol,
        },
    }


# Accumulation extents at ORD-01's (4, 512, 32) gate shape:
# grad_u / grad_delta see one reverse carry chain (~L); grad_B / grad_C sum
# D=32 chain-carrying terms (~D*L/2); grad_A sums batch*L terms each with an
# ~L/2 chain (~B*L^2/2); grad_D is a flat batch*L product sum (no chains).
SCAN_BWD_GATE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_u": _bwd_view_overrides(ord01_reduction_elements=512, ord03_atol=1e-3, ord03_rtol=1e-3),
    "grad_delta": _bwd_view_overrides(
        ord01_reduction_elements=512, ord03_atol=1e-3, ord03_rtol=1e-3
    ),
    "grad_A": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        # grad_A sums batch*L large near-integrator terms: the fp16
        # input-rounding error carries the magnitude of those intermediates
        # even at elements that cancel — flat atol and elementwise rtol both
        # misread it. B200-measured floors at this shape (scale-normalised):
        # honest 4.9e-4 (Triton) / 9.0e-4 (eager), fp16-carry cheat 1.25e-2;
        # unit atol 3e-3 holds >3x margin both ways (pinned by a
        # discriminative test).
        prc02={
            "shape": (2, 1024, 32),
            "atol": 3e-3,
            "rtol": 0.0,
            "scale_atol_by_ref_inf": True,
        },
    ),
    "grad_B": _bwd_view_overrides(
        ord01_reduction_elements=32 * 512 // 2, ord03_atol=3e-3, ord03_rtol=3e-3
    ),
    "grad_C": _bwd_view_overrides(
        ord01_reduction_elements=32 * 512 // 2, ord03_atol=3e-3, ord03_rtol=3e-3
    ),
    "grad_D": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512, ord03_atol=1e-3, ord03_rtol=1e-3
    ),
}


def bwd_scan_candidate_adapter(
    bwd_fn: BwdScanCallable,
    grad_field: str,
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one gradient output: ``dy -> bwd(...)[field]``."""
    idx = BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        inputs = _bwd_scan_aux(
            batch, seq_len, d_model, n_state, dy.device, dy.dtype, saturate=saturate
        )
        grads = bwd_fn(*inputs, dy, chunk_size=chunk_size)
        out: Tensor = grads[idx]
        return out

    return adapted


def bwd_scan_reference_adapter(
    grad_field: str,
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The autograd oracle behind the same per-gradient single-tensor interface."""
    idx = BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        inputs = _bwd_scan_aux(
            batch, seq_len, d_model, n_state, dy.device, dy.dtype, saturate=saturate
        )
        if dy.dtype == torch.float32:
            return reference_backward_selective_scan(*inputs, dy, chunk_size=chunk_size)[idx]
        # Mixed-precision contract: oracle computes in fp32 from the same
        # (already rounded) operand bits, rounds once at the output.
        u32, delta32, a32, b32, c32, d32 = (t.to(torch.float32) for t in inputs)
        grads = reference_backward_selective_scan(
            u32, delta32, a32, b32, c32, d32, dy.to(torch.float32), chunk_size=chunk_size
        )
        return grads[idx].to(dy.dtype)

    return adapted


def verify_bwd_scan_op(
    bwd_fn: BwdScanCallable,
    *,
    grad_field: str,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over one gradient-output view of a backward op.

    Mirrors ``verify_scan_op`` per view, including the saturation-free
    PRC-02 re-run (same rationale: the saturated channels' ~100x local
    amplification drowns the accumulator signal in fp16 input-rounding
    noise; saturation value-correctness is CMP-01's job on this view).
    """
    return _verify_op_views(
        lambda saturate: bwd_scan_candidate_adapter(
            bwd_fn, grad_field, n_state=n_state, chunk_size=chunk_size, saturate=saturate
        ),
        lambda saturate: bwd_scan_reference_adapter(
            grad_field, n_state=n_state, chunk_size=chunk_size, saturate=saturate
        ),
        base_overrides=SCAN_BWD_GATE_OVERRIDES.get(grad_field, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=True,
    )


def verify_bwd_scan_op_all_grads(
    bwd_fn: BwdScanCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all six gradient views — the backward op's full verdict."""
    return {
        grad_field: verify_bwd_scan_op(
            bwd_fn,
            grad_field=grad_field,
            device=device,
            resource_meta=resource_meta,
            n_state=n_state,
            chunk_size=chunk_size,
        )
        for grad_field in BWD_GRAD_FIELDS
    }


# ---------------------------------------------------------------------------
# MIMO backward (C3): one gate view per gradient output
# ---------------------------------------------------------------------------

# MIMO-backward signature: (x, B, C, dt, alpha, mimo_x, mimo_o, dy) -> the
# seven gradients as an indexable sequence (MimoGrads or plain tuple).
MimoBwdCallable = Callable[..., Any]

# The gates drive 3D [batch, seq, d_model] primaries; the MIMO dy views
# d_model as (nheads, MIMO_HEADDIM). 4 divides every gate d_model.
MIMO_HEADDIM = 4
MIMO_RANK = 4
MIMO_N_STATE = 16
_MIMO_BWD_AUX_SEED = 15331

# Gradient outputs of the MIMO backward, in MimoGrads field order.
MIMO_BWD_GRAD_FIELDS: tuple[str, ...] = (
    "grad_x",
    "grad_B",
    "grad_C",
    "grad_dt",
    "grad_alpha",
    "grad_mimo_x",
    "grad_mimo_o",
)


def _mimo_nheads(d_model: int, headdim: int) -> int:
    if d_model % headdim != 0:
        raise ValueError(f"d_model={d_model} not divisible by MIMO headdim {headdim}")
    return d_model // headdim


def _mimo_bwd_aux(
    batch: int,
    seq_len: int,
    nheads: int,
    headdim: int,
    rank: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (x, B, C, dt, alpha, mimo_x, mimo_o) for a 4D-viewed ``dy``.

    ``dt`` is log-uniform in the official Mamba dt-init range and ``alpha``
    is exp(dt * A) with per-head negative A (Mamba-3's A is a per-head
    scalar) — the near-integrator regime where low-precision accumulators
    actually lose mass, as in the scan aux. ``mimo_x``/``mimo_o`` are the
    official ones/R init plus seeded jitter: exact ones/R is
    rank-degenerate, and a rank-collapsing cheat would be invisible against
    it. No saturation variant exists: the op takes dt/alpha precomputed,
    there is no softplus inside it. Draw order under ``_MIMO_BWD_AUX_SEED``
    is pinned.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_MIMO_BWD_AUX_SEED)
    x = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    b_proj = torch.randn(batch, seq_len, rank, nheads, n_state, generator=gen)
    c_proj = torch.randn(batch, seq_len, rank, nheads, n_state, generator=gen)
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, nheads, generator=gen) * (log_hi - log_lo) + log_lo)
    a_head = -torch.rand(nheads, generator=gen)
    alpha = torch.exp(dt * a_head)
    jitter = 0.25 / rank
    mimo_x = 1.0 / rank + torch.randn(nheads, rank, headdim, generator=gen) * jitter
    mimo_o = 1.0 / rank + torch.randn(nheads, rank, headdim, generator=gen) * jitter

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(x), cast(b_proj), cast(c_proj), cast(dt), cast(alpha), cast(mimo_x), cast(mimo_o)


def _mimo_prc02(atol: float) -> dict[str, Any]:
    """Scale-aware PRC-02 config for a MIMO view at the L=4096 stress shape.

    Every MIMO gradient is a near-integrator sum whose fp16 input-rounding
    error carries the magnitude of large cancelling intermediates (the
    grad_A class) — flat atol and elementwise rtol both misread it, so all
    seven views run scale-aware. The L=1024 forward shape has no power: an
    fp16 state-carry cheat sits at 1.3-2.5x the honest floor there, inside
    draw variance. At L=4096 the cheat's per-step re-rounding compounds
    while the honest floor stays put — measured (CPU eager, 5 draws,
    scale-normalised): honest <= 2.1e-3..4.9e-3 per view vs cheat >=
    7.8e-3..1.4e-2; per-view unit atols sit between with the margin biased
    to the honest side. Probes: scratch/c3_prc02_floor.py (CPU calibration),
    scratch/c3_b200_floor.py (B200 confirmation against the Triton kernel,
    3 draws: kernel floor 1.2-2.6x under atol, cheat 1.3-2.7x over, every
    view discriminates both ways; grad_dt is the thin view at 1.2x/1.3x —
    recalibrate there first if it ever flakes).
    """
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Accumulation extents at ORD-01's (4, 512, 32) gate shape (nheads=8,
# headdim=4, R=4, N=16), theory-seeded under the eps*sqrt(chain)*scale
# model — B200-validated like the scan tables. grad_x sees one reverse
# carry chain (~L); grad_B / grad_C contract headdim chain-carrying terms
# (~P*L/2); grad_dt sums R*N of them (~R*N*L/2) and grad_alpha P*N
# (~P*N*L/2); grad_mimo_x / grad_mimo_o are batch*L sums of chain-carrying
# products (~B*L^2/2). PRC-02 entries: see _mimo_prc02.
MIMO_BWD_GATE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_x": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=1e-3,
        ord03_rtol=1e-3,
        prc02=_mimo_prc02(6e-3),
    ),
    "grad_B": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 // 2,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_mimo_prc02(6e-3),
    ),
    "grad_C": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 // 2,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_mimo_prc02(6e-3),
    ),
    "grad_dt": _bwd_view_overrides(
        ord01_reduction_elements=4 * 16 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_mimo_prc02(7e-3),
    ),
    "grad_alpha": _bwd_view_overrides(
        ord01_reduction_elements=4 * 16 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_mimo_prc02(8e-3),
    ),
    "grad_mimo_x": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_mimo_prc02(4e-3),
    ),
    "grad_mimo_o": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_mimo_prc02(4.5e-3),
    ),
}


def mimo_bwd_candidate_adapter(
    bwd_fn: MimoBwdCallable,
    grad_field: str,
    *,
    rank: int = MIMO_RANK,
    n_state: int = MIMO_N_STATE,
    headdim: int = MIMO_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one gradient output: ``dy -> bwd(...)[field]``.

    ``saturate`` is accepted for interface parity with the scan adapters
    and ignored — the MIMO aux has no saturation variant.
    """
    idx = MIMO_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        nheads = _mimo_nheads(d_model, headdim)
        aux = _mimo_bwd_aux(batch, seq_len, nheads, headdim, rank, n_state, dy.device, dy.dtype)
        grads = bwd_fn(*aux, dy.reshape(batch, seq_len, nheads, headdim))
        out: Tensor = grads[idx]
        return out

    return adapted


def mimo_bwd_reference_adapter(
    grad_field: str,
    *,
    rank: int = MIMO_RANK,
    n_state: int = MIMO_N_STATE,
    headdim: int = MIMO_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The autograd oracle behind the same per-gradient single-tensor interface."""
    idx = MIMO_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        nheads = _mimo_nheads(d_model, headdim)
        aux = _mimo_bwd_aux(batch, seq_len, nheads, headdim, rank, n_state, dy.device, dy.dtype)
        dy4 = dy.reshape(batch, seq_len, nheads, headdim)
        if dy.dtype == torch.float32:
            return reference_mimo_backward(*aux, dy4)[idx]
        # Mixed-precision contract: oracle computes in fp32 from the same
        # (already rounded) operand bits, rounds once at the output.
        x32, b32, c32, dt32, alpha32, mx32, mo32 = (t.to(torch.float32) for t in aux)
        grads = reference_mimo_backward(
            x32, b32, c32, dt32, alpha32, mx32, mo32, dy4.to(torch.float32)
        )
        return grads[idx].to(dy.dtype)

    return adapted


def verify_mimo_bwd_op(
    bwd_fn: MimoBwdCallable,
    *,
    grad_field: str,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    rank: int = MIMO_RANK,
    n_state: int = MIMO_N_STATE,
    headdim: int = MIMO_HEADDIM,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over one gradient-output view of the MIMO backward."""
    return _verify_op_views(
        lambda saturate: mimo_bwd_candidate_adapter(
            bwd_fn, grad_field, rank=rank, n_state=n_state, headdim=headdim, saturate=saturate
        ),
        lambda saturate: mimo_bwd_reference_adapter(
            grad_field, rank=rank, n_state=n_state, headdim=headdim, saturate=saturate
        ),
        base_overrides=MIMO_BWD_GATE_OVERRIDES.get(grad_field, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )


def verify_mimo_bwd_op_all_grads(
    bwd_fn: MimoBwdCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    rank: int = MIMO_RANK,
    n_state: int = MIMO_N_STATE,
    headdim: int = MIMO_HEADDIM,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all seven gradient views — the MIMO backward's full verdict."""
    return {
        grad_field: verify_mimo_bwd_op(
            bwd_fn,
            grad_field=grad_field,
            device=device,
            resource_meta=resource_meta,
            rank=rank,
            n_state=n_state,
            headdim=headdim,
        )
        for grad_field in MIMO_BWD_GRAD_FIELDS
    }


# ---------------------------------------------------------------------------
# Complex-RoPE scan (C4): forward op, one gate view
# ---------------------------------------------------------------------------

# Rope-scan signature: (x, B, C, dt, A, angle_proj) -> y.
RopeCallable = Callable[..., Tensor]

ROPE_HEADDIM = 4
ROPE_N_STATE = 16
# rotary_dim = 2*6 = 12 < N = 16: every gate run exercises the rotated
# lanes and the identity tail simultaneously.
ROPE_NUM_ANGLES = 6
_ROPE_AUX_SEED = 27791


def _rope_aux(
    batch: int,
    seq_len: int,
    nheads: int,
    n_state: int,
    num_angles: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (B, C, dt, A, angle_proj) for a 4D-viewed ``x``.

    ``dt`` is log-uniform in the official dt-init range and ``A`` per-head
    negative — the near-integrator regime, as in the scan aux. ``angle_proj``
    is unit-normal: tanh maps it across both its linear and saturated
    regions, and the rotation magnitude |tanh * dt * pi| stays well inside
    one period per step. No saturation variant exists (no softplus in the
    op). Draw order under ``_ROPE_AUX_SEED`` is pinned.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_ROPE_AUX_SEED)
    b_proj = torch.randn(batch, seq_len, nheads, n_state, generator=gen)
    c_proj = torch.randn(batch, seq_len, nheads, n_state, generator=gen)
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, nheads, generator=gen) * (log_hi - log_lo) + log_lo)
    a_head = -torch.rand(nheads, generator=gen)
    angle = torch.randn(batch, seq_len, nheads, num_angles, generator=gen)

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(b_proj), cast(c_proj), cast(dt), cast(a_head), cast(angle)


# The accumulation chain is the sequence (h carried over L), as for the
# forward scan: extent L=512 at ORD-01's gate shape. ORD-03 keeps the
# scan's L=8192 collapse length but with 3x the scan's tolerance: the
# kernel accumulates theta with a per-step remainder while the reference
# applies one mod after an fp32 cumsum whose magnitude reaches O(L*dt) —
# the reference's own rounding there contributes ~1e-3 rad of honest
# angle noise at L=8192, which the value tolerance must absorb on top of
# reorder noise. PRC-02 runs the scan's stress shape and default atol:
# the rotation factors are bounded by 1 and leave the accumulator-error
# structure of the decay scan unchanged (pinned by a discriminative test).
ROPE_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "gate_prc_02_mixed_precision_accumulation": {"shape": (2, 1024, 32)},
    "gate_ord_01_reduction_order_tolerance": {"reduction_elements": 512},
    "gate_ord_03_noncommutative_reduction": {
        "shape": (1, 8192, 16),
        "atol": 3e-3,
        "rtol": 3e-3,
    },
}


def rope_candidate_adapter(
    rope_fn: RopeCallable,
    *,
    n_state: int = ROPE_N_STATE,
    num_angles: int = ROPE_NUM_ANGLES,
    headdim: int = ROPE_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of the rope scan: ``x -> y``, d_model as (nheads, headdim).

    ``saturate`` is accepted for interface parity with the scan adapters
    and ignored — the rope aux has no saturation variant.
    """

    def adapted(x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        nheads = _mimo_nheads(d_model, headdim)
        aux = _rope_aux(batch, seq_len, nheads, n_state, num_angles, x.device, x.dtype)
        y = rope_fn(x.reshape(batch, seq_len, nheads, headdim), *aux)
        return y.reshape(batch, seq_len, d_model)

    return adapted


def rope_reference_adapter(
    *,
    n_state: int = ROPE_N_STATE,
    num_angles: int = ROPE_NUM_ANGLES,
    headdim: int = ROPE_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The reference oracle behind the same single-tensor interface."""

    def adapted(x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        nheads = _mimo_nheads(d_model, headdim)
        aux = _rope_aux(batch, seq_len, nheads, n_state, num_angles, x.device, x.dtype)
        x4 = x.reshape(batch, seq_len, nheads, headdim)
        if x.dtype == torch.float32:
            return reference_complex_scan_rope(x4, *aux).reshape(batch, seq_len, d_model)
        # Mixed-precision contract: oracle computes in fp32 from the same
        # (already rounded) operand bits, rounds once at the output.
        y = reference_complex_scan_rope(x4.to(torch.float32), *(t.to(torch.float32) for t in aux))
        return y.to(x.dtype).reshape(batch, seq_len, d_model)

    return adapted


def verify_rope_op(
    rope_fn: RopeCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = ROPE_N_STATE,
    num_angles: int = ROPE_NUM_ANGLES,
    headdim: int = ROPE_HEADDIM,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over a rope-scan-signature callable."""
    return _verify_op_views(
        lambda saturate: rope_candidate_adapter(
            rope_fn,
            n_state=n_state,
            num_angles=num_angles,
            headdim=headdim,
            saturate=saturate,
        ),
        lambda saturate: rope_reference_adapter(
            n_state=n_state, num_angles=num_angles, headdim=headdim, saturate=saturate
        ),
        base_overrides=ROPE_GATE_OVERRIDES,
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )
