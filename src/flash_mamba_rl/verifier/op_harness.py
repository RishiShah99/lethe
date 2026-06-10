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

This is the same wiring the RL reward path needs to score generated
kernels per op; Phase D consumes it via ``score_candidate(gate_kwargs=...)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from flash_mamba_rl.kernels.references import reference_forward_chunked_scan
from flash_mamba_rl.verifier.contracts import GateResult, run_all_gates

ScanCallable = Callable[..., Tensor]

SCAN_N_STATE = 16
# Must divide every sequence length the gates use (CMP-02's default L=8 is
# the smallest); the scan result itself does not depend on chunking.
SCAN_CHUNK_SIZE = 8
_SCAN_AUX_SEED = 23117

# Per-gate overrides appropriate for the scan op.
SCAN_GATE_OVERRIDES: dict[str, dict[str, Any]] = {
    # The gate's default (2, 32, 1024) puts the long axis on D, which the
    # scan treats elementwise. The op's accumulation axes are the sequence
    # (h carried over L) and the N-dot; a missing fp32 accumulator shows up
    # on a long scan, so stress L instead. At this shape with the Mamba-
    # realistic delta below, the honest fp32-accumulator floor is ~6e-3 and
    # an fp16 accumulator lands at ~1.5e-1, so the gate's default atol=2e-2
    # separates them with margin (pinned by a discriminative test).
    "gate_prc_02_mixed_precision_accumulation": {"shape": (2, 1024, 32)},
}

# Official Mamba dt initialisation range (state-spaces/mamba,
# modules/mamba_simple.py): dt log-uniform in [dt_min, dt_max].
_DT_MIN = 1e-3
_DT_MAX = 1e-1


def _scan_aux(
    batch: int,
    seq_len: int,
    d_model: int,
    n_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic (delta, A, B, C, D_skip) for a [batch, seq_len, d_model] primary.

    ``delta`` is drawn so that ``softplus(delta)`` is log-uniform in
    [1e-3, 1e-1] — the official Mamba dt-init distribution. This matters
    for gate power: small dt makes the scan a near-integrator with long
    memory, which is both the deployment-realistic regime and the one
    where low-precision accumulators actually lose mass (PRC-02). A
    unit-scale delta would make the scan strongly contractive and mask
    accumulator cheats.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SCAN_AUX_SEED)
    log_lo = math.log(_DT_MIN)
    log_hi = math.log(_DT_MAX)
    dt = torch.exp(torch.rand(batch, seq_len, d_model, generator=gen) * (log_hi - log_lo) + log_lo)
    delta = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
    a = -torch.rand(d_model, n_state, generator=gen)  # negative for stability
    b_proj = torch.randn(batch, seq_len, n_state, generator=gen)
    c_proj = torch.randn(batch, seq_len, n_state, generator=gen)
    d_skip = torch.randn(d_model, generator=gen)

    def cast(t: Tensor) -> Tensor:
        return t.to(device=device, dtype=dtype)

    return cast(delta), cast(a), cast(b_proj), cast(c_proj), cast(d_skip)


def scan_candidate_adapter(
    scan_fn: ScanCallable,
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> Callable[[Tensor], Tensor]:
    """Wrap a scan-signature callable into the gates' single-tensor interface."""

    def adapted(u: Tensor) -> Tensor:
        batch, seq_len, d_model = u.shape
        aux = _scan_aux(batch, seq_len, d_model, n_state, u.device, u.dtype)
        return scan_fn(u, *aux, chunk_size=chunk_size)

    return adapted


def scan_reference_adapter(
    *,
    n_state: int = SCAN_N_STATE,
    chunk_size: int = SCAN_CHUNK_SIZE,
) -> Callable[[Tensor], Tensor]:
    """The reference oracle behind the same single-tensor interface."""

    def adapted(u: Tensor) -> Tensor:
        batch, seq_len, d_model = u.shape
        aux = _scan_aux(batch, seq_len, d_model, n_state, u.device, u.dtype)
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
    """
    overrides: dict[str, dict[str, Any]] = {k: dict(v) for k, v in SCAN_GATE_OVERRIDES.items()}
    if resource_meta is not None:
        overrides["gate_res_02_resource_limits"] = {"resource_meta": resource_meta}
    return run_all_gates(
        scan_candidate_adapter(scan_fn, n_state=n_state, chunk_size=chunk_size),
        scan_reference_adapter(n_state=n_state, chunk_size=chunk_size),
        device=device,
        gate_overrides=overrides,
    )
