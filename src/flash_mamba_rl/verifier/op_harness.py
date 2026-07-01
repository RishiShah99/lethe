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

from flash_mamba_rl.kernels.cute.gdn2_assemble import (
    assembled_channelwise_gdn2_backward,
    assembled_scalar_gdn2_backward,
)
from flash_mamba_rl.kernels.references import (
    reference_backward_selective_scan,
    reference_forward_chunked_scan,
)
from flash_mamba_rl.kernels.references.complex_scan_rope import reference_complex_scan_rope
from flash_mamba_rl.kernels.references.fused_block_backward import reference_fused_block_backward
from flash_mamba_rl.kernels.references.fused_block_forward import reference_fused_block_forward
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward
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
    cmp03_atol: float | None = None,
    cmp01_atol: float | None = None,
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

    ``cmp03_atol`` widens CMP-03's unit atol for views whose cross-impl
    reorder noise compounds past the 1e-5 default (B200-calibrated, the
    C1 lesson applied per view — see FUSED_BWD_GATE_OVERRIDES).
    ``cmp01_atol`` is the same lesson at CMP-01's shapes — its long_seq
    variation carries a longer worst chain than any CMP-03 shape.
    """
    overrides: dict[str, dict[str, Any]] = {
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
    if cmp03_atol is not None:
        overrides["gate_cmp_03_shape_polymorphism"] = {"atol": cmp03_atol}
    if cmp01_atol is not None:
        overrides["gate_cmp_01_input_variation"] = {"atol": cmp01_atol}
    return overrides


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
        ord01_reduction_elements=4 * 512,
        ord03_atol=1e-3,
        ord03_rtol=1e-3,
        # grad_D is a flat batch*L product sum: |out| ~ sqrt(B*L) ~ 100 at
        # the gate shape, so the honest fp16 *input-rounding* floor and one
        # fp16 output ULP both exceed the flat 2e-2 default — the gate could
        # only pass by draw luck (surfaced by the SFT-target validation; the
        # hand-written kernel itself failed). B200-measured over 8 draws
        # (scratch/c2_gradd_floor.py, scale-normalised): honest kernel/eager
        # <= 5.9e-4 vs fp16-sequential-accumulation cheat >= 3.8e-3; unit
        # atol 1.5e-3 holds ~2.5x margin both ways.
        prc02={
            "shape": (2, 1024, 32),
            "atol": 1.5e-3,
            "rtol": 0.0,
            "scale_atol_by_ref_inf": True,
        },
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
# GDN-2 backward: one gate view per gradient output
# ---------------------------------------------------------------------------

# GDN-2-backward signature: (q, k, v, g, b, w, do) -> the six gradients as an
# indexable sequence (Gdn2Grads, whose 7th field grad_initial_state is None and
# never viewed). The gates drive 3D [batch, seq, d_model] primaries; the GDN-2
# ``do`` views d_model as (nheads, GDN2_HEADDIM). The crown target is d_k=d_v=128;
# the harness uses GDN2_HEADDIM=4 (d_k=d_v) so every gate d_model factors cleanly.
Gdn2BwdCallable = Callable[..., Any]

GDN2_HEADDIM = 4  # d_k == d_v at the gate shapes (no GVA; crown is H == HV)
_GDN2_BWD_AUX_SEED = 41213

# Gradient outputs of the GDN-2 backward, in Gdn2Grads field order (the 6 the
# kernel produces; grad_initial_state is excluded — no initial state is fed).
GDN2_BWD_GRAD_FIELDS: tuple[str, ...] = (
    "grad_q",
    "grad_k",
    "grad_v",
    "grad_g",
    "grad_b",
    "grad_w",
)


def _gdn2_nheads(d_model: int, headdim: int) -> int:
    if d_model % headdim != 0:
        raise ValueError(f"d_model={d_model} not divisible by GDN-2 headdim {headdim}")
    return d_model // headdim


def _gdn2_bwd_aux(
    batch: int,
    seq_len: int,
    nheads: int,
    headdim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (q, k, v, g, b, w) for a 4D-viewed ``do``.

    ``g`` is channel-wise log-decay built from the official Mamba dt-init range
    times a per-head negative rate (so exp(g) stays in the near-integrator regime
    where low-precision accumulators lose mass), plus a small per-channel jitter
    kept <= 0 for stability. ``b`` (erase, key axis) and ``w`` (write, value axis)
    are sigmoid-squashed into (0, 1) — the gate domains. No saturation variant
    exists: the op takes g/b/w precomputed, there is no softplus inside it. Draw
    order under ``_GDN2_BWD_AUX_SEED`` is pinned.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_GDN2_BWD_AUX_SEED)
    q = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    k = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    v = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, nheads, generator=gen) * (log_hi - log_lo) + log_lo)
    a_head = -torch.rand(nheads, generator=gen)
    g_head = dt * a_head  # (batch, seq, nheads), <= 0
    jitter = -torch.rand(batch, seq_len, nheads, headdim, generator=gen) * 0.02
    g = g_head.unsqueeze(-1) + jitter  # channel-wise, <= 0
    b = torch.randn(batch, seq_len, nheads, headdim, generator=gen).sigmoid()
    w = torch.randn(batch, seq_len, nheads, headdim, generator=gen).sigmoid()

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(q), cast(k), cast(v), cast(g), cast(b), cast(w)


def _gdn2_prc02(atol: float) -> dict[str, Any]:
    """Scale-aware PRC-02 config for a GDN-2 view at the L=4096 stress shape.

    GDN-2 gradients are reverse-state-recurrence sums whose fp16 input-rounding
    error carries the magnitude of large cancelling intermediates (the delta-rule
    erase term), so a flat atol misreads them — every view runs scale-aware
    (``scale_atol_by_ref_inf``), as the MIMO backward does.

    Calibrated on CPU at (1, 4096, 32), 6 views x several draws
    (``scratch/gdn2_prc02_floor.py``): the honest fp16 input-rounding floor is
    5.5e-4..7.6e-4 of output scale across views. Crucially, an fp16-*state* cheat
    sits only ~2x over that floor — GDN-2's decay-limited memory (exp(g) ~ 0.9, so
    only the last ~tens of tokens contribute) caps how much an fp16 accumulator can
    drift, so it is NOT separable at a safe atol. The robust adversary is a bf16
    (or coarser) state accumulator at 4.3e-3..9.4e-3 of scale (5.9x..14.9x over the
    floor). The unit atol below (2e-3) sits ~3x above the honest floor and >=2.1x
    under the bf16 cheat. This is a DESK floor that catches coarse accumulators; the
    tight fp16-vs-fp32 discrimination floor is re-pinned against the native kernel
    on B200 in Phase 2 (as the MIMO/scan tables were).
    """
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Per-view overrides. ORD reduction extents at ORD-01's (4, 512, 32) gate shape
# (nheads=8, d_k=d_v=4) are theory-seeded under the eps*sqrt(chain)*scale model:
# every gradient's dominant chain is the reverse-time state carry (~L). ORD/CMP
# tolerances are DESK-SEEDED (no kernel yet) — re-pin on B200 in Phase 2, like the
# MIMO/scan tables were. PRC-02 floor is calibrated on CPU (see _gdn2_prc02).
_GDN2_PRC02_ATOL = 2e-3
GDN2_BWD_GATE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_q": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=1e-3,
        ord03_rtol=1e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
    "grad_k": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
    "grad_v": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=1e-3,
        ord03_rtol=1e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
    "grad_g": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
    "grad_b": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
    "grad_w": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
}


def gdn2_bwd_candidate_adapter(
    bwd_fn: Gdn2BwdCallable,
    grad_field: str,
    *,
    headdim: int = GDN2_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one gradient output: ``do -> bwd(...)[field]``.

    ``saturate`` is accepted for interface parity with the scan adapters and
    ignored — the GDN-2 aux has no saturation variant.
    """
    idx = GDN2_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, headdim)
        aux = _gdn2_bwd_aux(batch, seq_len, nheads, headdim, do.device, do.dtype)
        grads = bwd_fn(*aux, do.reshape(batch, seq_len, nheads, headdim))
        out: Tensor = grads[idx]
        return out

    return adapted


def gdn2_bwd_reference_adapter(
    grad_field: str,
    *,
    headdim: int = GDN2_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The autograd oracle behind the same per-gradient single-tensor interface."""
    idx = GDN2_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, headdim)
        q, k, v, g, b, w = _gdn2_bwd_aux(batch, seq_len, nheads, headdim, do.device, do.dtype)
        do4 = do.reshape(batch, seq_len, nheads, headdim)
        if do.dtype == torch.float32:
            out = reference_gdn2_backward(q, k, v, g, b, w, do4)[idx]
        else:
            # Mixed-precision contract: oracle computes in fp32 from the same
            # (already rounded) operand bits, rounds once at the output.
            q32, k32, v32, g32, b32, w32 = (t.to(torch.float32) for t in (q, k, v, g, b, w))
            out = reference_gdn2_backward(q32, k32, v32, g32, b32, w32, do4.to(torch.float32))[idx]
            out = out.to(do.dtype) if out is not None else None
        assert out is not None  # idx is always a grad field, never grad_initial_state
        return out

    return adapted


def verify_gdn2_bwd_op(
    bwd_fn: Gdn2BwdCallable,
    *,
    grad_field: str,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    headdim: int = GDN2_HEADDIM,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over one gradient-output view of the GDN-2 backward."""
    return _verify_op_views(
        lambda saturate: gdn2_bwd_candidate_adapter(
            bwd_fn, grad_field, headdim=headdim, saturate=saturate
        ),
        lambda saturate: gdn2_bwd_reference_adapter(grad_field, headdim=headdim, saturate=saturate),
        base_overrides=GDN2_BWD_GATE_OVERRIDES.get(grad_field, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )


def verify_gdn2_bwd_op_all_grads(
    bwd_fn: Gdn2BwdCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    headdim: int = GDN2_HEADDIM,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all six gradient views — the GDN-2 backward's full verdict."""
    return {
        grad_field: verify_gdn2_bwd_op(
            bwd_fn,
            grad_field=grad_field,
            device=device,
            resource_meta=resource_meta,
            headdim=headdim,
        )
        for grad_field in GDN2_BWD_GRAD_FIELDS
    }


# ---------------------------------------------------------------------------
# GDN-2 scalar reduction gate: the Phase-2 native-assembly integration credential
# ---------------------------------------------------------------------------

# The Phase-2 native kernels are SCALAR-GDN (g scalar per token, b = w = beta). The
# assembly (kernels.cute.gdn2_assemble) is graded here in that regime against the
# pure-torch refs assembly (the kernels' readable contracts; same wiring, K#1/K#2 on
# their torch reference paths). This mirrors the stock GDN-2 gate's discipline
# (candidate vs same-algorithm reference): on a CPU desk run the candidate IS the refs
# assembly, so all 12 gates verify the assembly's contract compliance (differentiable,
# deterministic, dtype/exceptional/subnormal-faithful); on a Blackwell box the candidate
# is the tcgen05 native path, so the gates become a real kernel-vs-reference cross-check
# (tolerances re-pinned there). Independent VALUE correctness is carried separately by
# the fp64 oracle test (assembly vs the token-serial oracle, bit-exact) and the chunkwise
# tests — keeping the reference here structurally matched avoids false EXC/subnormal
# divergence between two algebraically-equal but op-order-distinct backwards.
#
# Views: grad_q/grad_k/grad_v compare channel-wise; grad_g and grad_beta (the combined
# erase+write gate grad) compare as scalars — the only quantities a scalar kernel
# recovers. Distinct from the stock channel-wise GDN-2 gate above (eager fallback;
# the channel-wise crown is Phase 3).
_GDN2_SCALAR_AUX_SEED = 51217
GDN2_REDUCTION_VIEWS: tuple[str, ...] = ("grad_q", "grad_k", "grad_v", "grad_g", "grad_beta")


def _gdn2_bwd_aux_scalar(
    batch: int,
    seq_len: int,
    nheads: int,
    d_k: int,
    d_v: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Scalar-reducible (q, k, v, g, b, w): g channel-constant, ``b = w = beta·1``.

    Same near-integrator decay distribution as ``_gdn2_bwd_aux`` (official Mamba
    dt-init x per-head negative rate) but with NO per-channel jitter, so g is a single
    log-decay per token broadcast across d_k; beta is sigmoid-squashed into (0, 1) and
    broadcast as both the erase (b, key axis) and write (w, value axis) gate. Draw
    order under ``_GDN2_SCALAR_AUX_SEED`` is pinned.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_GDN2_SCALAR_AUX_SEED)
    q = torch.randn(batch, seq_len, nheads, d_k, generator=gen)
    k = torch.randn(batch, seq_len, nheads, d_k, generator=gen)
    v = torch.randn(batch, seq_len, nheads, d_v, generator=gen)
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, nheads, generator=gen) * (log_hi - log_lo) + log_lo)
    a_head = -torch.rand(nheads, generator=gen)
    g_scalar = dt * a_head  # (batch, seq, nheads), <= 0
    beta = torch.randn(batch, seq_len, nheads, generator=gen).sigmoid()
    g = g_scalar.unsqueeze(-1).expand(batch, seq_len, nheads, d_k)
    b = beta.unsqueeze(-1).expand(batch, seq_len, nheads, d_k)
    w = beta.unsqueeze(-1).expand(batch, seq_len, nheads, d_v)

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype).contiguous()

    return cast(q), cast(k), cast(v), cast(g), cast(b), cast(w)


def _gdn2_reduce_candidate(view: str, grads: Any) -> Tensor:
    """One reduced view from a candidate's six-grad bundle (channel-summed for g/beta)."""
    if view == "grad_q":
        out: Tensor = grads[0]
    elif view == "grad_k":
        out = grads[1]
    elif view == "grad_v":
        out = grads[2]
    elif view == "grad_g":
        out = grads[3].sum(-1)
    elif view == "grad_beta":  # combined erase (grad_b) + write (grad_w)
        out = grads[4].sum(-1) + grads[5].sum(-1)
    else:
        raise ValueError(view)
    return out


def gdn2_reduction_candidate_adapter(
    bwd_fn: Gdn2BwdCallable,
    view: str,
    *,
    d_k: int,
    d_v: int,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one (possibly channel-summed) gradient of the assembly."""

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, d_v)
        aux = _gdn2_bwd_aux_scalar(batch, seq_len, nheads, d_k, d_v, do.device, do.dtype)
        grads = bwd_fn(*aux, do.reshape(batch, seq_len, nheads, d_v))
        return _gdn2_reduce_candidate(view, grads)

    return adapted


def gdn2_reduction_reference_adapter(
    view: str,
    *,
    d_k: int,
    d_v: int,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The pure-torch refs assembly behind the same reduced single-tensor interface."""

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, d_v)
        q, k, v, g, b, w = _gdn2_bwd_aux_scalar(
            batch, seq_len, nheads, d_k, d_v, do.device, do.dtype
        )
        do4 = do.reshape(batch, seq_len, nheads, d_v)
        grads = assembled_scalar_gdn2_backward(q, k, v, g, b, w, do4)
        return _gdn2_reduce_candidate(view, grads)

    return adapted


_GDN2_REDUCTION_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    **{f: GDN2_BWD_GATE_OVERRIDES[f] for f in ("grad_q", "grad_k", "grad_v", "grad_g")},
    "grad_beta": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_gdn2_prc02(_GDN2_PRC02_ATOL),
    ),
}


def verify_gdn2_reduction_op(
    bwd_fn: Gdn2BwdCallable,
    *,
    view: str,
    d_k: int = GDN2_HEADDIM,
    d_v: int = GDN2_HEADDIM,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
) -> dict[str, GateResult]:
    """All 12 gates over one reduced view of the native scalar-GDN assembly."""
    return _verify_op_views(
        lambda saturate: gdn2_reduction_candidate_adapter(
            bwd_fn, view, d_k=d_k, d_v=d_v, saturate=saturate
        ),
        lambda saturate: gdn2_reduction_reference_adapter(
            view, d_k=d_k, d_v=d_v, saturate=saturate
        ),
        base_overrides=_GDN2_REDUCTION_OVERRIDES.get(view, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )


def verify_gdn2_reduction_op_all_grads(
    bwd_fn: Gdn2BwdCallable,
    *,
    d_k: int = GDN2_HEADDIM,
    d_v: int = GDN2_HEADDIM,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all five reduced views — the assembly's full reduction verdict."""
    return {
        view: verify_gdn2_reduction_op(
            bwd_fn,
            view=view,
            d_k=d_k,
            d_v=d_v,
            device=device,
            resource_meta=resource_meta,
        )
        for view in GDN2_REDUCTION_VIEWS
    }


# ---------------------------------------------------------------------------
# GDN-2 channel-wise gate: the Phase-3 crown native-assembly integration credential
# ---------------------------------------------------------------------------

# The Phase-3 native kernels are channel-wise GDN-2 (per-channel decay g; erase b on the
# key axis, write w on the value axis). The channel-wise assembly
# (kernels.cute.gdn2_assemble.assembled_channelwise_gdn2_backward) is graded here against
# the channel-wise refs assembly (the kernels' readable contracts on their torch paths) —
# the same candidate-vs-same-algorithm discipline as the scalar reduction gate, so all 12
# gates verify contract compliance (differentiable, deterministic, dtype/exceptional/
# subnormal-faithful). On a Blackwell box the candidate is the tcgen05 native path, turning
# the gates into a real channel-wise kernel-vs-reference cross-check (tolerances re-pinned).
# Independent VALUE correctness is carried by the fp64 oracle test (channel-wise assembly vs
# the token-serial GDN-2 oracle, bit-exact) and the channel-wise chunkwise tests. All six
# per-channel grads are viewed directly (no reduction) — this is the full crown credential,
# distinct from the scalar reduction gate (5 channel-summed views) above. The channel-wise
# aux (``_gdn2_bwd_aux``, per-channel g/b/w) is shared with the stock GDN-2 gate.


def gdn2_channelwise_reference_adapter(
    grad_field: str,
    *,
    headdim: int = GDN2_HEADDIM,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The channel-wise refs assembly behind the per-gradient single-tensor interface."""
    idx = GDN2_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, headdim)
        q, k, v, g, b, w = _gdn2_bwd_aux(batch, seq_len, nheads, headdim, do.device, do.dtype)
        do4 = do.reshape(batch, seq_len, nheads, headdim)
        grads = assembled_channelwise_gdn2_backward(q, k, v, g, b, w, do4)
        out = grads[idx]
        assert out is not None  # idx is always a grad field, never grad_initial_state
        return out

    return adapted


def verify_gdn2_channelwise_op(
    bwd_fn: Gdn2BwdCallable,
    *,
    grad_field: str,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    headdim: int = GDN2_HEADDIM,
) -> dict[str, GateResult]:
    """All 12 gates over one per-channel gradient view of the channel-wise assembly."""
    return _verify_op_views(
        lambda saturate: gdn2_bwd_candidate_adapter(
            bwd_fn, grad_field, headdim=headdim, saturate=saturate
        ),
        lambda saturate: gdn2_channelwise_reference_adapter(
            grad_field, headdim=headdim, saturate=saturate
        ),
        base_overrides=GDN2_BWD_GATE_OVERRIDES.get(grad_field, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )


def verify_gdn2_channelwise_op_all_grads(
    bwd_fn: Gdn2BwdCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    headdim: int = GDN2_HEADDIM,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all six per-channel views — the channel-wise crown's full verdict."""
    return {
        grad_field: verify_gdn2_channelwise_op(
            bwd_fn,
            grad_field=grad_field,
            device=device,
            resource_meta=resource_meta,
            headdim=headdim,
        )
        for grad_field in GDN2_BWD_GRAD_FIELDS
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
# scan's L=8192 collapse length with 3x the scan's value tolerance as
# kernel reorder/FMA headroom over the longer decay chain (theory-seeded;
# the reference's own cumsum-vs-fp64 angle noise at L=8192 measures only
# ~1e-6 rad and is not the budget driver). The fp16-state cheat measures
# 5.2e-2 at this shape — 17x over the 3e-3 budget — so the collapse
# length keeps ORD-03 discriminative too. B200 gate run passed unchanged.
# PRC-02 runs the scan's stress shape scale-aware with a unit atol of
# 3e-3 (the C2 grad_A / C3 MIMO convention — rope outputs at the gate
# shape reach |ref|_inf ~6-9, so a flat atol mixes axes with the
# scale-relative floors). Floors, as fractions of output scale: honest
# eager <= 7.3e-4 vs fp16-state cheat >= 8.6e-3 on CPU (3 draws,
# scratch/c4_prc02_floor.py); B200 Triton kernel <= 6.3e-4 vs cheat
# >= 1.05e-2 (scratch/c4_b200_floor.py) — 4.8x under / 3.5x over the
# unit atol on-device, pinned by a discriminative test.
ROPE_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "gate_prc_02_mixed_precision_accumulation": {
        "shape": (2, 1024, 32),
        "atol": 3e-3,
        "scale_atol_by_ref_inf": True,
    },
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


# ---------------------------------------------------------------------------
# Fused block (C5): forward op, one gate view
# ---------------------------------------------------------------------------

# Fused-block signature: (x, conv_weight, conv_bias, delta, A, B, C, D,
# norm_weight, *, conv_kernel_size, eps, chunk_size) -> y.
FusedCallable = Callable[..., Tensor]

FUSED_CONV_K = 4
_FUSED_AUX_SEED = 40427


def _fused_aux(
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    conv_k: int,
    device: torch.device,
    dtype: torch.dtype,
    saturate: bool = True,
) -> tuple[Tensor, ...]:
    """Deterministic (conv_w, conv_b, delta, A, B, C, D_skip, norm_w) for a [B, L, D] primary.

    The scan core (delta/A/B/C/D_skip) comes from ``_aux_from_gen`` — same
    distribution rationale, including the saturation variant (the op has a
    softplus inside its scan). ``conv_weight`` is unit-normal scaled
    1/sqrt(K) so the conv output stays near input scale; ``conv_bias`` is
    half-scale, shifting the SiLU operating point across its nonlinear
    range without drowning the signal; ``norm_weight`` is 1 + 0.25*jitter —
    exact ones is gain-degenerate (per-channel gain would go untested).
    Draw order under ``_FUSED_AUX_SEED`` is pinned.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_FUSED_AUX_SEED)
    delta, a, b_proj, c_proj, d_skip = _aux_from_gen(
        gen, batch, seq_len, d_model, n_state, saturate
    )
    conv_w = torch.randn(d_model, 1, conv_k, generator=gen) / math.sqrt(conv_k)
    conv_b = 0.5 * torch.randn(d_model, generator=gen)
    norm_w = 1.0 + 0.25 * torch.randn(d_model, generator=gen)

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return (
        cast(conv_w),
        cast(conv_b),
        cast(delta),
        cast(a),
        cast(b_proj),
        cast(c_proj),
        cast(d_skip),
        cast(norm_w),
    )


# The dominant accumulation chain is still the scan over L (extent 512 at
# ORD-01's gate shape); the RMSNorm adds a D-extent reduction (32) that is
# negligible under the sqrt-N model. ORD-03 keeps the scan's L=8192
# collapse length with 3x the scan's value tolerance as kernel reorder/FMA
# headroom across the conv window, the scan chain and the cross-D norm
# reduction (theory-seeded, the C4 convention; B200 must confirm).
# PRC-02 runs at L=4096, the C3 lesson: at L=1024 the fp16-state cheat's
# per-step re-rounding has only compounded to 7.8x the honest floor; at
# L=4096 the corridor is 15.6x. Scale-aware (the C2 grad_A / C3 / C4
# convention — outputs here reach |ref|_inf ~7). Floors as fractions of
# output scale, CPU over 3 draws (scratch/c5_prc02_floor.py): honest
# eager <= 1.26e-3 vs fp16-state cheat >= 1.95e-2 — unit atol 5e-3 sits
# 4.0x over honest / 3.9x under cheat, pinned by a discriminative test.
# B200 (scratch/c5_b200_floor.py, 3 draws): Triton kernel <= 9.75e-4 vs
# cheat >= 2.08e-2 — 21x corridor, 5.1x under / 4.2x over the unit atol.
# PRC-02's discrimination here is the L-chain scan state only: a fp16
# norm-ssq cheat measures 1.2-1.6e-3 of scale (C5.g review) — at the
# honest floor, because the D=32 ssq chain is too short for fp16
# re-rounding to compound above input-rounding noise; no tolerance can
# separate it at the gate's d_model. The norm's fp32 accumulator is
# pinned structurally instead (kernel-replica test + byte-identical
# determinism).
FUSED_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "gate_prc_02_mixed_precision_accumulation": {
        "shape": (1, 4096, 32),
        "atol": 5e-3,
        "scale_atol_by_ref_inf": True,
    },
    # The conv bias makes this op affine in x — the first op in the suite
    # where subnormal inputs do NOT produce subnormal-scale outputs (the
    # bias path dominates at O(1), where C1-C4 are multiplicative in the
    # primary). EXC-02's value branch therefore measures plain kernel
    # reorder noise at output scale (B200 kernel: 8.0e-6), not subnormal
    # handling; the zero-mask parity branch keeps exact semantics. 1e-4 is
    # ~12x the measured noise and far below any real numeric divergence.
    # Input-subnormal preservation is fundamentally unobservable for this
    # op (a DAZ load changes the output by ~1e-38 under an O(1) bias term)
    # — that is the op's structure, not the override. Internal subnormal
    # behaviour (a_bar underflow at saturated delta, the C1 ex2.approx
    # lesson) stays covered by EXC-01/CMP-01's saturated-delta aux.
    "gate_exc_02_subnormal_handling": {"atol": 1e-4},
    "gate_ord_01_reduction_order_tolerance": {"reduction_elements": 512},
    "gate_ord_03_noncommutative_reduction": {
        "shape": (1, 8192, 16),
        "atol": 3e-3,
        "rtol": 3e-3,
    },
}


def fused_candidate_adapter(
    fused_fn: FusedCallable,
    *,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of the fused block: ``x -> y``.

    The primary is the *unpadded* input; the adapter applies the causal
    left-padding of K-1 zeros itself, so the op's output length equals the
    gate primary's length and non-finite injections at step t smear
    causally into outputs t..t+K-1, identically for candidate and
    reference.
    """

    def adapted(x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        aux = _fused_aux(
            batch, seq_len, d_model, n_state, conv_k, x.device, x.dtype, saturate=saturate
        )
        x_pad = torch.nn.functional.pad(x, (0, 0, conv_k - 1, 0))
        return fused_fn(x_pad, *aux, conv_kernel_size=conv_k, chunk_size=chunk_size)

    return adapted


def fused_reference_adapter(
    *,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The reference oracle behind the same single-tensor interface."""

    def adapted(x: Tensor) -> Tensor:
        batch, seq_len, d_model = x.shape
        aux = _fused_aux(
            batch, seq_len, d_model, n_state, conv_k, x.device, x.dtype, saturate=saturate
        )
        x_pad = torch.nn.functional.pad(x, (0, 0, conv_k - 1, 0))
        if x.dtype == torch.float32:
            return reference_fused_block_forward(
                x_pad, *aux, conv_kernel_size=conv_k, chunk_size=chunk_size
            )
        # Mixed-precision contract: oracle computes in fp32 from the same
        # (already rounded) operand bits, rounds once at the output.
        y = reference_fused_block_forward(
            x_pad.to(torch.float32),
            *(t.to(torch.float32) for t in aux),
            conv_kernel_size=conv_k,
            chunk_size=chunk_size,
        )
        return y.to(x.dtype)

    return adapted


def verify_fused_block_op(
    fused_fn: FusedCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over a fused-block-signature callable.

    PRC-02 is re-run with saturation-free auxiliaries, exactly as for the
    forward scan: the saturated delta entries probe softplus value
    correctness (CMP-01's same-bits domain) but their amplification drowns
    PRC-02's accumulator signal in fp16 input-rounding noise.
    """
    return _verify_op_views(
        lambda saturate: fused_candidate_adapter(
            fused_fn, n_state=n_state, conv_k=conv_k, chunk_size=chunk_size, saturate=saturate
        ),
        lambda saturate: fused_reference_adapter(
            n_state=n_state, conv_k=conv_k, chunk_size=chunk_size, saturate=saturate
        ),
        base_overrides=FUSED_GATE_OVERRIDES,
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=True,
    )


# ---------------------------------------------------------------------------
# Fused block backward (C6): one gate view per gradient output
# ---------------------------------------------------------------------------

# Fused-backward signature: (x, conv_weight, conv_bias, delta, A, B, C, D,
# norm_weight, dy, *, conv_kernel_size, eps, chunk_size) -> the nine
# gradients as an indexable sequence (FusedBlockGrads or plain tuple).
FusedBwdCallable = Callable[..., Any]

_FUSED_BWD_AUX_SEED = 52361

# Gradient outputs of the fused-block backward, in FusedBlockGrads order.
FUSED_BWD_GRAD_FIELDS: tuple[str, ...] = (
    "grad_x",
    "grad_conv_weight",
    "grad_conv_bias",
    "grad_delta",
    "grad_A",
    "grad_B",
    "grad_C",
    "grad_D",
    "grad_norm_weight",
)


def _fused_bwd_aux(
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    conv_k: int,
    device: torch.device,
    dtype: torch.dtype,
    saturate: bool = True,
) -> tuple[Tensor, ...]:
    """Deterministic (x, conv_w, conv_b, delta, A, B, C, D_skip, norm_w) for a [B, L, D] ``dy``.

    The backward op's primary is the upstream gradient ``dy``, so the
    forward input ``x`` joins the auxiliaries — unit-normal and *unpadded*
    (the adapters apply the causal K-1 left-padding, mirroring the forward
    harness). Remaining draws follow ``_fused_aux``'s distribution
    rationale; a distinct seed keeps the backward aux decorrelated from
    the forward harness's primaries.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_FUSED_BWD_AUX_SEED)
    x = torch.randn(batch, seq_len, d_model, generator=gen)
    delta, a, b_proj, c_proj, d_skip = _aux_from_gen(
        gen, batch, seq_len, d_model, n_state, saturate
    )
    conv_w = torch.randn(d_model, 1, conv_k, generator=gen) / math.sqrt(conv_k)
    conv_b = 0.5 * torch.randn(d_model, generator=gen)
    norm_w = 1.0 + 0.25 * torch.randn(d_model, generator=gen)

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return (
        cast(x),
        cast(conv_w),
        cast(conv_b),
        cast(delta),
        cast(a),
        cast(b_proj),
        cast(c_proj),
        cast(d_skip),
        cast(norm_w),
    )


def _fused_bwd_prc02(atol: float) -> dict[str, Any]:
    """Scale-aware PRC-02 config for a fused-backward view at (1, 4096, 32).

    Same axes as the forward view (L=4096 needed for cheat compounding;
    scale-aware because gradient errors carry the magnitude of large
    cancelling intermediates — the C2 grad_A / C3 convention). Floors
    CPU-measured per view over 3 draws (scratch/c6_prc02_floor.py),
    honest eager vs autograd through the fp16-scan-state forward cheat,
    scale-normalised: honest <= 7.0e-4..1.6e-3 per view; cheat corridors
    range 44x (grad_A, 4.2e-2) down to 3.4x (grad_conv_bias, 5.3e-3) and
    5.0x (grad_D, 6.4e-3) — the conv-bias and D-skip paths touch the
    fp16 state only through dconv/dys, so the cheat's compounding is
    diluted there; those two views carry the thinnest margins (the C3
    grad_dt situation — re-measure first on B200 if either flakes).
    Per-view unit atols sit between the floors, margin biased honest-side.

    B200 re-measurement (scratch/c6_b200_floor.py, Triton pipeline vs the
    same cheat, 3 draws): kernel honest <= 1.28e-3 of scale (worst view
    grad_B), every view discriminates both ways; margins vs the unit
    atols 3.1-6.2x under / 1.8-6.3x over, thinnest grad_D (1.8x over)
    and grad_conv_bias (2.5x over) — matching the CPU prediction.
    """
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Accumulation extents at ORD-01's (4, 512, 32) gate shape, theory-seeded
# under the eps*sqrt(chain)*scale model (the scan-table convention; B200
# confirms). grad_x / grad_delta see one reverse carry chain (~L);
# grad_B / grad_C contract D=32 chain-carrying terms (~D*L/2); grad_A,
# grad_conv_weight, grad_conv_bias and grad_norm_weight sum batch*L terms
# that each carry an ~L/2 chain (~B*L^2/2); grad_D is a batch*L sum of
# chain-carrying dys*z products (~B*L). ORD-03 keeps the scan's L=8192
# collapse length with the fused forward's 3x headroom for the conv
# window and cross-D norm reduction; the long-chain-sum views (grad_A
# class) take 1e-2, the C2 grad_A precedent.
#
# CMP-03 unit atols are B200-calibrated (scratch/c6_cmp03_probe.py, the
# gate's 5 shapes x 5 draws): the kernel's honest cross-impl reorder
# noise rides the forward chain state h (recomputed in-kernel vs torch's
# chunked scan, both fp32 — divergence compounds over L), and the views
# that contract against h or the reverse carry g measure up to 1.4e-5 of
# output scale (grad_A/B/C; 1.9e-5 observed in a live gate draw) — past
# the 1e-5 default. grad_A/B/C take 1e-4 (~5-7x over measured); the five
# other overridden views take 5e-5 (measured <= 8.4e-6);
# grad_norm_weight keeps the default (measured 1.5e-6 — the rms division
# cancels h's compounding and the norm backward is local in t). The
# atols stay >= 50x under the PRC-02 cheat floors (floors measured at
# L=4096, a conservative basis for CMP-03's L<=256 shapes).
#
# CMP-01 runs the same kernel-vs-oracle comparison and its long_seq
# variation (4, 256, 32) carries a longer worst chain (B*L^2/2 = 131072)
# than any CMP-03 shape, so the same default cannot be assumed safe.
# B200-measured at CMP-01's base + long_seq shapes, 8 draws (the gate's
# n_random; same probe): the h-contracting views reach 6.1e-6 of scale
# (grad_C; grad_delta 5.7e-6, grad_B 4.3e-6, grad_A 2.7e-6) — under the
# 1e-5 default but a 1.6x margin on an unseeded reward gate, and CMP-03's
# live draw exceeded its probe envelope by 1.36x. Those four views take
# cmp01_atol 5e-5 (~8x over measured, above the eps32*sqrt(chain) model
# bound of 4.3e-5); the rest measure <= 1.8e-6 and keep the default.
FUSED_BWD_GATE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_x": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_fused_bwd_prc02(4e-3),
        cmp03_atol=5e-5,
    ),
    "grad_conv_weight": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_fused_bwd_prc02(5e-3),
        cmp03_atol=5e-5,
    ),
    "grad_conv_bias": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_fused_bwd_prc02(3e-3),
        cmp03_atol=5e-5,
    ),
    "grad_delta": _bwd_view_overrides(
        ord01_reduction_elements=512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_fused_bwd_prc02(4e-3),
        cmp03_atol=5e-5,
        cmp01_atol=5e-5,
    ),
    "grad_A": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_fused_bwd_prc02(5e-3),
        cmp03_atol=1e-4,
        cmp01_atol=5e-5,
    ),
    "grad_B": _bwd_view_overrides(
        ord01_reduction_elements=32 * 512 // 2,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_fused_bwd_prc02(4e-3),
        cmp03_atol=1e-4,
        cmp01_atol=5e-5,
    ),
    "grad_C": _bwd_view_overrides(
        ord01_reduction_elements=32 * 512 // 2,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_fused_bwd_prc02(4e-3),
        cmp03_atol=1e-4,
        cmp01_atol=5e-5,
    ),
    "grad_D": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512,
        ord03_atol=3e-3,
        ord03_rtol=3e-3,
        prc02=_fused_bwd_prc02(3e-3),
        cmp03_atol=5e-5,
    ),
    "grad_norm_weight": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        prc02=_fused_bwd_prc02(3e-3),
    ),
}


def fused_bwd_candidate_adapter(
    bwd_fn: FusedBwdCallable,
    grad_field: str,
    *,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one gradient output: ``dy -> bwd(...)[field]``.

    The primary is the upstream gradient ``dy``; the aux ``x`` is padded
    here so non-finite injections through ``dy`` hit output rows whose
    indices match the gate primary's, identically for candidate and
    reference.
    """
    idx = FUSED_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        x, *aux = _fused_bwd_aux(
            batch, seq_len, d_model, n_state, conv_k, dy.device, dy.dtype, saturate=saturate
        )
        x_pad = torch.nn.functional.pad(x, (0, 0, conv_k - 1, 0))
        grads = bwd_fn(x_pad, *aux, dy, conv_kernel_size=conv_k, chunk_size=chunk_size)
        out: Tensor = grads[idx]
        return out

    return adapted


def fused_bwd_reference_adapter(
    grad_field: str,
    *,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The autograd oracle behind the same per-gradient single-tensor interface."""
    idx = FUSED_BWD_GRAD_FIELDS.index(grad_field)

    def adapted(dy: Tensor) -> Tensor:
        batch, seq_len, d_model = dy.shape
        x, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w = _fused_bwd_aux(
            batch, seq_len, d_model, n_state, conv_k, dy.device, dy.dtype, saturate=saturate
        )
        x_pad = torch.nn.functional.pad(x, (0, 0, conv_k - 1, 0))
        if dy.dtype != torch.float32:
            # Mixed-precision contract: oracle computes in fp32 from the
            # same (already rounded) operand bits, rounds once at the output.
            x_pad, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w = (
                t.to(torch.float32)
                for t in (x_pad, conv_w, conv_b, delta, a, b_proj, c_proj, d_skip, norm_w)
            )
        grads = reference_fused_block_backward(
            x_pad,
            conv_w,
            conv_b,
            delta,
            a,
            b_proj,
            c_proj,
            d_skip,
            norm_w,
            dy.to(torch.float32),
            conv_kernel_size=conv_k,
            chunk_size=chunk_size,
        )
        return grads[idx].to(dy.dtype)

    return adapted


def verify_fused_bwd_op(
    bwd_fn: FusedBwdCallable,
    *,
    grad_field: str,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, GateResult]:
    """Run all 12 contract gates over one gradient view of the fused backward.

    Mirrors ``verify_bwd_scan_op`` per view, including the saturation-free
    PRC-02 re-run (the op has a softplus inside its scan; the saturated
    channels' amplification drowns the accumulator signal).
    """
    return _verify_op_views(
        lambda saturate: fused_bwd_candidate_adapter(
            bwd_fn,
            grad_field,
            n_state=n_state,
            conv_k=conv_k,
            chunk_size=chunk_size,
            saturate=saturate,
        ),
        lambda saturate: fused_bwd_reference_adapter(
            grad_field, n_state=n_state, conv_k=conv_k, chunk_size=chunk_size, saturate=saturate
        ),
        base_overrides=FUSED_BWD_GATE_OVERRIDES.get(grad_field, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=True,
    )


def verify_fused_bwd_op_all_grads(
    bwd_fn: FusedBwdCallable,
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
    n_state: int = SCAN_N_STATE,
    conv_k: int = FUSED_CONV_K,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over all nine gradient views — the backward op's full verdict."""
    return {
        grad_field: verify_fused_bwd_op(
            bwd_fn,
            grad_field=grad_field,
            device=device,
            resource_meta=resource_meta,
            n_state=n_state,
            conv_k=conv_k,
            chunk_size=chunk_size,
        )
        for grad_field in FUSED_BWD_GRAD_FIELDS
    }


# ---------------------------------------------------------------------------
# Elementwise SiLU: the GRPO trainer's toy validation op
# ---------------------------------------------------------------------------

ELEMENTWISE_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    # Bitwise identity vs torch eager is unachievable for a hardware kernel
    # (libdevice exp vs torch's sigmoid differ at ULP level); a near-ULP
    # budget keeps the gate discriminative — wrong math errs > 1e-2 here.
    "gate_ord_03_noncommutative_reduction": {"atol": 1e-6, "rtol": 1e-6},
}


def elementwise_silu_reference(x: Tensor) -> Tensor:
    """fp32 eager SiLU oracle under the op-harness mixed-precision contract."""
    if x.dtype in (torch.float32, torch.float64):
        return x * torch.sigmoid(x)
    x32 = x.to(torch.float32)
    return (x32 * torch.sigmoid(x32)).to(x.dtype)


def verify_elementwise_op(
    fn: Callable[[Tensor], Tensor],
    *,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
) -> dict[str, GateResult]:
    """All 12 contract gates over an elementwise single-tensor callable.

    Candidates already match the gates' single-tensor interface, so no
    adapter closure or saturation re-run is needed and the gate defaults
    apply unchanged (no accumulation axis to re-point).
    """
    overrides: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in ELEMENTWISE_GATE_OVERRIDES.items()
    }
    if resource_meta is not None:
        overrides["gate_res_02_resource_limits"] = {"resource_meta": resource_meta}
    return run_all_gates(fn, elementwise_silu_reference, device=device, gate_overrides=overrides)
