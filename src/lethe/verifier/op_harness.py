"""Adapters bridging multi-argument kernel ops to the single-tensor gates."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_assemble import (
    assembled_channelwise_gdn2_backward,
    assembled_scalar_gdn2_backward,
)
from lethe.kernels.references import (
    reference_backward_selective_scan,
    reference_forward_chunked_scan,
)
from lethe.kernels.references.complex_scan_rope import reference_complex_scan_rope
from lethe.kernels.references.fused_block_backward import reference_fused_block_backward
from lethe.kernels.references.fused_block_forward import reference_fused_block_forward
from lethe.kernels.references.gdn_backward import reference_gdn2_backward
from lethe.kernels.references.mimo_backward import reference_mimo_backward
from lethe.verifier.contracts import (
    GateResult,
    run_all_gates,
)

ScanCallable = Callable[..., Tensor]
# Backward-scan signature returns six grads (SelectiveScanGrads or a plain tuple).
BwdScanCallable = Callable[..., Any]

SCAN_N_STATE = 16
# Must divide every gate sequence length (min is CMP-02's L=8); scan result is chunk-independent.
SCAN_CHUNK_SIZE = 8
_SCAN_AUX_SEED = 23117
_BWD_AUX_SEED = 9241

# Backward-scan gradient outputs, in SelectiveScanGrads field order; each gets its own gate run.
BWD_GRAD_FIELDS: tuple[str, ...] = ("grad_u", "grad_delta", "grad_A", "grad_B", "grad_C", "grad_D")

# Per-gate overrides appropriate for the scan op.
SCAN_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    # The gate's default (2, 32, 1024) puts the long axis on D, which the scan treats elementwise.
    "gate_prc_02_mixed_precision_accumulation": {"shape": (2, 1024, 32)},
    # The scan's accumulation chain is the sequence, not the trailing dim; at the gate shape L=512.
    "gate_ord_01_reduction_order_tolerance": {"reduction_elements": 512},
    # Bitwise match to torch eager is unachievable (~2e-6 on B200); use a length where orderings diverge.
    "gate_ord_03_noncommutative_reduction": {
        "shape": (1, 8192, 16),
        "atol": 1e-3,
        "rtol": 1e-3,
    },
}

# Official Mamba dt init (state-spaces/mamba mamba_simple.py): dt log-uniform in [dt_min, dt_max].
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
    """Generic gate run for one adapted view: overrides + optional PRC-02 re-run."""
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
    # Routes the saturation-free re-run through run_all_gates for the per-gate seed and RNG bracket.
    gate = "gate_prc_02_mixed_precision_accumulation"
    prc02_override = base_overrides.get(gate)
    rerun = run_all_gates(
        candidate_factory(False),
        reference_factory(False),
        device=device,
        gate_names=[gate],
        gate_overrides={gate: prc02_override} if prc02_override else None,
    )
    results[gate] = rerun[gate]
    return results


def _aux_from_gen(
    gen: torch.Generator,
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    saturate: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """CPU-side (delta, A, B, C, D_skip) drawn from ``gen``, shared aux core."""
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
    """Deterministic (delta, A, B, C, D_skip) for a [batch, seq_len, d_model] primary."""
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
    """Deterministic (u, delta, A, B, C, D_skip) for a [batch, seq_len, d_model] ``dy``."""
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
        # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """Run all 12 contract gates over a scan-signature callable."""
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


def _bwd_view_overrides(
    *,
    ord01_reduction_elements: int,
    ord03_atol: float,
    ord03_rtol: float,
    prc02: dict[str, Any] | None = None,
    cmp03_atol: float | None = None,
    cmp01_atol: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-view gate overrides sharing the backward op's fixed shapes."""
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


# Reduction extents scale with each field's accumulation chain: carry ~L, sums ~D*L/2 or B*L^2/2.
SCAN_BWD_GATE_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_u": _bwd_view_overrides(ord01_reduction_elements=512, ord03_atol=1e-3, ord03_rtol=1e-3),
    "grad_delta": _bwd_view_overrides(
        ord01_reduction_elements=512, ord03_atol=1e-3, ord03_rtol=1e-3
    ),
    "grad_A": _bwd_view_overrides(
        ord01_reduction_elements=4 * 512 * 512 // 2,
        ord03_atol=1e-2,
        ord03_rtol=1e-2,
        # grad_A sums near-integrator terms; fp16 rounding error carries their magnitude past cancellation.
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
        # grad_D: |out| ~ sqrt(B*L) ~ 100 at gate shape; fp16 rounding floor exceeds the flat 2e-2 default.
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
        # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """Run all 12 contract gates over one gradient-output view of a backward op."""
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
    """All 12 gates over all six gradient views: the backward op's full verdict."""
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


# MIMO-backward signature returns seven grads (MimoGrads or a plain tuple).
MimoBwdCallable = Callable[..., Any]

# Gates drive 3D [batch, seq, d_model] primaries; MIMO dy views d_model as (nheads, MIMO_HEADDIM).
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
    """Deterministic (x, B, C, dt, alpha, mimo_x, mimo_o) for a 4D-viewed ``dy``."""
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
    """Scale-aware PRC-02 config for a MIMO view at the L=4096 stress shape."""
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Extents at (4,512,32) (nheads=8, headdim=4, R=4, N=16), eps*sqrt(chain)*scale, B200-validated.
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
    """Single-tensor view of one gradient output: ``dy -> bwd(...)[field]``."""
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
        # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """All 12 gates over all seven gradient views: the MIMO backward's full verdict."""
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


# GDN-2-backward signature returns six grads (Gdn2Grads; grad_initial_state unused, always None).
Gdn2BwdCallable = Callable[..., Any]

GDN2_HEADDIM = 4  # d_k == d_v at the gate shapes (no GVA; H == HV)
_GDN2_BWD_AUX_SEED = 41213

# Gradient outputs of the GDN-2 backward, in Gdn2Grads order; grad_initial_state is excluded.
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
    """Deterministic (q, k, v, g, b, w) for a 4D-viewed ``do``."""
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
    """Scale-aware PRC-02 config for a GDN-2 view at the L=4096 stress shape."""
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Per-view overrides.
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
    """Single-tensor view of one gradient output: ``do -> bwd(...)[field]``."""
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
            # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """All 12 gates over all six gradient views: the GDN-2 backward's full verdict."""
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


# The native scalar-GDN kernels use g scalar per token, b = w = beta.
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
    """Scalar-reducible (q, k, v, g, b, w): g channel-constant, ``b = w = beta·1``."""
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
    """All 12 gates over all five reduced views: the assembly's full reduction verdict."""
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


# Channel-wise native kernels use per-channel decay g; erase b on the key axis, write w on value.


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
    """All 12 gates over all six per-channel views: the channel-wise op's full verdict."""
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


# Each family member (LA/GLA/SSD/KDA) runs all 12 gates via the gdn2_family wrapper's native mode.

FAMILY_GATE_VIEWS: dict[str, tuple[str, ...]] = {
    "gla": ("grad_q", "grad_k", "grad_v", "grad_g"),
    "la": ("grad_q", "grad_k", "grad_v"),
    "ssd": ("grad_q", "grad_k", "grad_v", "grad_g"),
    "kda": ("grad_q", "grad_k", "grad_v", "grad_g", "grad_beta"),
}

# One pinned draw seed per family (distinct streams; draw order documented per builder).
_FAMILY_AUX_SEEDS: dict[str, int] = {"gla": 61931, "la": 61933, "ssd": 61937, "kda": 61949}


def _family_bwd_aux(
    family: str,
    batch: int,
    seq_len: int,
    nheads: int,
    d_k: int,
    d_v: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, ...]:
    """Deterministic family-native aux for a 4D-viewed ``do``."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_FAMILY_AUX_SEEDS[family])
    q = torch.randn(batch, seq_len, nheads, d_k, generator=gen)
    k = torch.randn(batch, seq_len, nheads, d_k, generator=gen)
    v = torch.randn(batch, seq_len, nheads, d_v, generator=gen)

    def _g_scalar() -> Tensor:
        log_lo = math.log(_DT_MIN)
        log_hi = math.log(_DT_MAX)
        dt = torch.exp(
            torch.rand(batch, seq_len, nheads, generator=gen) * (log_hi - log_lo) + log_lo
        )
        a_head = -torch.rand(nheads, generator=gen)
        return dt * a_head  # (batch, seq, nheads), <= 0

    def _g_channelwise() -> Tensor:
        g_head = _g_scalar()
        jitter = -torch.rand(batch, seq_len, nheads, d_k, generator=gen) * 0.02
        return g_head.unsqueeze(-1) + jitter

    aux: tuple[Tensor, ...]
    if family == "gla":
        aux = (q, k, v, _g_channelwise())
    elif family == "la":
        aux = (q, k, v)
    elif family == "ssd":
        aux = (q, k, v, _g_scalar())
    elif family == "kda":
        g = _g_channelwise()
        beta = torch.randn(batch, seq_len, nheads, generator=gen).sigmoid()
        aux = (q, k, v, g, beta)
    else:
        raise ValueError(family)

    return tuple(t.to(device=device, dtype=dtype).contiguous() for t in aux)


def family_candidate_adapter(
    family: str,
    bwd_fn: Gdn2BwdCallable,
    view: str,
    *,
    d_k: int,
    d_v: int,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """Single-tensor view of one gradient of a family-mode backward."""
    idx = FAMILY_GATE_VIEWS[family].index(view)

    def adapted(do: Tensor) -> Tensor:
        batch, seq_len, d_model = do.shape
        nheads = _gdn2_nheads(d_model, d_v)
        aux = _family_bwd_aux(family, batch, seq_len, nheads, d_k, d_v, do.device, do.dtype)
        grads = bwd_fn(*aux, do.reshape(batch, seq_len, nheads, d_v))
        out: Tensor = grads[idx]
        return out

    return adapted


def _family_refs_backward(family: str) -> Gdn2BwdCallable:
    """The refs-path family wrapper (lazy import keeps the verifier import-light)."""
    from lethe.kernels.cute import gdn2_family

    fns: dict[str, Gdn2BwdCallable] = {
        "gla": gdn2_family.gla_backward,
        "la": gdn2_family.la_backward,
        "ssd": gdn2_family.ssd_backward,
        "kda": gdn2_family.kda_backward,
    }
    return fns[family]


def family_reference_adapter(
    family: str,
    view: str,
    *,
    d_k: int,
    d_v: int,
    saturate: bool = True,
) -> Callable[[Tensor], Tensor]:
    """The structurally matched refs-path wrapper behind the same interface."""
    return family_candidate_adapter(
        family, _family_refs_backward(family), view, d_k=d_k, d_v=d_v, saturate=saturate
    )


# Family views reuse GDN-2 tolerances: dominant chains match (reverse-time carry ~L) across families.
_FAMILY_VIEW_OVERRIDES: dict[str, dict[str, dict[str, Any]]] = {
    "grad_q": GDN2_BWD_GATE_OVERRIDES["grad_q"],
    "grad_k": GDN2_BWD_GATE_OVERRIDES["grad_k"],
    "grad_v": GDN2_BWD_GATE_OVERRIDES["grad_v"],
    "grad_g": GDN2_BWD_GATE_OVERRIDES["grad_g"],
    "grad_beta": _GDN2_REDUCTION_OVERRIDES["grad_beta"],
}


def verify_gdn2_family_op(
    bwd_fn: Gdn2BwdCallable,
    *,
    family: str,
    view: str,
    d_k: int = GDN2_HEADDIM,
    d_v: int = GDN2_HEADDIM,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
) -> dict[str, GateResult]:
    """All 12 gates over one gradient view of a family-mode backward."""
    if family not in FAMILY_GATE_VIEWS:
        raise ValueError(family)
    return _verify_op_views(
        lambda saturate: family_candidate_adapter(
            family, bwd_fn, view, d_k=d_k, d_v=d_v, saturate=saturate
        ),
        lambda saturate: family_reference_adapter(
            family, view, d_k=d_k, d_v=d_v, saturate=saturate
        ),
        base_overrides=_FAMILY_VIEW_OVERRIDES.get(view, {}),
        device=device,
        resource_meta=resource_meta,
        saturation_rerun=False,
    )


def verify_gdn2_family_op_all_views(
    bwd_fn: Gdn2BwdCallable,
    *,
    family: str,
    d_k: int = GDN2_HEADDIM,
    d_v: int = GDN2_HEADDIM,
    device: str | torch.device = "cpu",
    resource_meta: dict[str, int] | None = None,
) -> dict[str, dict[str, GateResult]]:
    """All 12 gates over every gradient view: one family's full verdict."""
    return {
        view: verify_gdn2_family_op(
            bwd_fn,
            family=family,
            view=view,
            d_k=d_k,
            d_v=d_v,
            device=device,
            resource_meta=resource_meta,
        )
        for view in FAMILY_GATE_VIEWS[family]
    }


# Rope-scan signature: (x, B, C, dt, A, angle_proj) -> y.
RopeCallable = Callable[..., Tensor]

ROPE_HEADDIM = 4
ROPE_N_STATE = 16
# rotary_dim = 2*6 = 12 < N = 16: every gate run exercises both rotated lanes and identity tail.
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
    """Deterministic (B, C, dt, A, angle_proj) for a 4D-viewed ``x``."""
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


# Accumulation chain is the sequence (h carried over L); extent L=512 at ORD-01's gate shape.
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
    """Single-tensor view of the rope scan: ``x -> y``, d_model as (nheads, headdim)."""

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
        # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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


# Fused-block signature: (x, conv_w, conv_b, delta, A, B, C, D, norm_w, kwargs) -> y.
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
    """Deterministic (conv_w, conv_b, delta, A, B, C, D_skip, norm_w) for a [B, L, D] primary."""
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


# Dominant chain is the scan over L (512 at gate shape); RMSNorm's D=32 reduction is negligible.
FUSED_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    "gate_prc_02_mixed_precision_accumulation": {
        "shape": (1, 4096, 32),
        "atol": 5e-3,
        "scale_atol_by_ref_inf": True,
    },
    # Conv bias makes this op affine in x, so subnormal inputs don't yield subnormal-scale outputs.
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
    """Single-tensor view of the fused block: ``x -> y``."""

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
        # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """Run all 12 contract gates over a fused-block-signature callable."""
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


# Fused-backward signature: adds dy to the forward args; returns nine grads (FusedBlockGrads).
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
    """Deterministic (x, conv_w, conv_b, delta, A, B, C, D_skip, norm_w) for a [B, L, D] ``dy``."""
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
    """Scale-aware PRC-02 config for a fused-backward view at (1, 4096, 32)."""
    return {
        "shape": (1, 4096, 32),
        "atol": atol,
        "rtol": 0.0,
        "scale_atol_by_ref_inf": True,
    }


# Extents at (4,512,32), theory-seeded under eps*sqrt(chain)*scale, B200-confirmed.
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
    """Single-tensor view of one gradient output: ``dy -> bwd(...)[field]``."""
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
            # Mixed-precision: oracle computes in fp32 from the same rounded bits, rounds once at output.
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
    """Run all 12 contract gates over one gradient view of the fused backward."""
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
    """All 12 gates over all nine gradient views: the backward op's full verdict."""
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


ELEMENTWISE_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    # Bitwise match is unachievable (libdevice exp vs sigmoid differ at ULP); near-ULP tol, errs > 1e-2.
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
    """All 12 contract gates over an elementwise single-tensor callable."""
    overrides: dict[str, dict[str, Any]] = {
        k: dict(v) for k, v in ELEMENTWISE_GATE_OVERRIDES.items()
    }
    if resource_meta is not None:
        overrides["gate_res_02_resource_limits"] = {"resource_meta": resource_meta}
    return run_all_gates(fn, elementwise_silu_reference, device=device, gate_overrides=overrides)
