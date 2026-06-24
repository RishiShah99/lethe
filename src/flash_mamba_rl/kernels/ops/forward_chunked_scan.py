"""C1 — hand-written SISO selective-scan forward.

Drop-in for ``reference_forward_chunked_scan`` with the same signature and
semantics, widened to more dtypes and devices:

- CUDA + {fp32, fp16, bf16} with triton installed -> the Triton kernel
  (``_triton_fwd_scan``), wrapped in an autograd.Function whose backward
  recomputes through the eager path. The backward is provisional until C2
  lands the hand-written Triton backward kernel.
- everything else (CPU, fp64, missing triton) -> ``_scan_eager``, a
  differentiable eager-torch path replicating the reference op-for-op.
  For fp32 on the same device it is bitwise-equal to the reference.
  fp64 is a verification-only dtype (gradcheck); it deliberately routes
  to the eager path rather than the Triton kernel.

Deviations from the reference: the reference rejects non-fp32 inputs; this
op defines the mixed-precision contract instead (compute in fp32, round
once on store). ``chunk_size`` is validated identically but is a blocking
hint only — the scan result does not depend on it.
"""

from __future__ import annotations

from importlib import util as _importlib_util
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig

_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_TRITON: bool | None = None


def _triton_usable() -> bool:
    global _HAS_TRITON
    if _HAS_TRITON is None:
        _HAS_TRITON = _importlib_util.find_spec("triton") is not None
    return _HAS_TRITON


_CHUNK_PARALLEL_CAP = 512


def _auto_chunk_len(seq_len: int, requested: int | None) -> int:
    """Largest divisor of ``seq_len`` not exceeding ``requested`` (or the cap).

    The chunk-parallel launcher requires ``chunk_len | seq_len``; a fixed value
    cannot divide every shape the gate battery probes (CMP-03/CMP-01), and the
    serial carry is O(seq_len / chunk_len), so a tiny chunk_len is slow. Choosing
    the largest divisor ≤ target makes chunk_parallel a contract-clean drop-in
    (never raises, identical math) that still picks a coarse, fast granularity at
    long L. ``requested`` is the RL/autotuner's preferred granularity; ``None``
    falls back to the cap.
    """
    target = min(requested if requested is not None else _CHUNK_PARALLEL_CAP, seq_len)
    for k in range(target, 0, -1):
        if seq_len % k == 0:
            return k
    return 1


def _default_scan_mode(seq_len: int, batch: int, width: int, *, is_forward: bool) -> str:
    """Launch-default scan mode for a shape, consulted only when scan_mode is unset.

    Calibrated to the broad boundary sweep (``results/scan_mode_boundary.json``,
    178 shapes x 3 SISO ops, best-tuned-vs-best-tuned through the verifier):
    chunk_parallel is the global optimum (geomean ~2.2x over the serial default)
    EXCEPT in the saturated short-sequence corner, where the O(seq_len/chunk_len)
    carry cannot amortise against an already SM-saturated device. Serial there,
    chunk_parallel otherwise; an explicit ``config.scan_mode`` overrides this so
    the autotuner/config-RL still drives the choice. ``is_forward`` adds the
    forward op's extra serial preference at every short L (the forward kernel is
    cheap enough per step that the carry rarely pays below L=512). The thresholds
    are the sweep's measured boundary, slightly biased toward chunk_parallel (the
    dominant winner): the rule trails the per-shape oracle by <0.1% in geomean
    with its worst single-shape regression ~5%.
    """
    if is_forward and seq_len <= 512:
        return "serial"
    if batch >= 8 and width >= 4096 and seq_len <= 4096:
        return "serial"
    if batch >= 8 and width >= 2048 and seq_len <= 512:
        return "serial"
    return "chunk_parallel"


def _resolve_scan_mode(
    config: KernelConfig | None, seq_len: int, batch: int, width: int, *, is_forward: bool
) -> str:
    """The effective scan mode: an explicit config knob, else the shape default."""
    if config is not None and config.scan_mode is not None:
        return config.scan_mode
    return _default_scan_mode(seq_len, batch, width, is_forward=is_forward)


def _scan_eager(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """Reference math, replicated op-for-op, in fp32 (fp64 stays fp64)."""
    compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    uc = u.to(compute_dtype)
    dc = delta.to(compute_dtype)
    ac = A.to(compute_dtype)
    bc = B.to(compute_dtype)
    cc = C.to(compute_dtype)
    dsc = D.to(compute_dtype)

    batch, seq_len, d_model = uc.shape
    n_state = ac.shape[1]

    delta_bar = F.softplus(dc)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * ac.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * bc.unsqueeze(2)

    h = torch.zeros(batch, d_model, n_state, dtype=compute_dtype, device=uc.device)
    ys: list[Tensor] = []
    for t in range(seq_len):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * uc[:, t, :].unsqueeze(-1)
        ys.append((h * cc[:, t, :].unsqueeze(1)).sum(-1) + dsc * uc[:, t, :])
    return torch.stack(ys, dim=1).to(u.dtype)


class _ForwardScanCuda(torch.autograd.Function):
    """Triton forward; backward is the hand-written Triton kernel (C2).

    First-order only: the Triton backward returns plain tensors with no
    graph, so double-backward (HVPs, gradient penalties) through the CUDA
    path is unsupported — route through the eager path (CPU or fp64) when
    higher-order gradients are needed.
    """

    @staticmethod
    def forward(
        ctx: Any,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        config: KernelConfig | None = None,
    ) -> Tensor:
        ctx.save_for_backward(u, delta, A, B, C, D)
        ctx.config = config
        batch, seq_len, width = u.shape
        if _resolve_scan_mode(config, seq_len, batch, width, is_forward=True) == "chunk_parallel":
            from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_fwd

            k = _auto_chunk_len(seq_len, config.chunk_len if config is not None else None)
            return _triton_chunk_parallel_fwd.launch_chunk_parallel_scan(
                u, delta, A, B, C, D, chunk_len=k, config=config
            )
        from flash_mamba_rl.kernels.ops import _triton_fwd_scan

        return _triton_fwd_scan.launch_forward_scan(u, delta, A, B, C, D, config=config)

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        u, delta, a, b, c, d_skip = ctx.saved_tensors
        config = ctx.config
        # Both scan modes are correct for any tiling; chunk_parallel reassociates
        # the backward the same way it does the forward. The backward is its own
        # regime (the C2 adjoint, not the C1 forward), so it resolves the mode
        # with is_forward=False rather than inheriting the forward pass's choice.
        batch, seq_len, width = u.shape
        if _resolve_scan_mode(config, seq_len, batch, width, is_forward=False) == "chunk_parallel":
            from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_bwd

            k = _auto_chunk_len(seq_len, config.chunk_len if config is not None else None)
            grads = _triton_chunk_parallel_bwd.launch_backward_chunk_parallel_scan(
                u, delta, a, b, c, d_skip, grad_y, chunk_len=k, config=config
            )
        else:
            from flash_mamba_rl.kernels.ops import _triton_bwd_scan

            grads = _triton_bwd_scan.launch_backward_scan(
                u, delta, a, b, c, d_skip, grad_y, config=config
            )
        # +1 None for the non-tensor forward arg (config).
        return (*grads, None)


def forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
    config: KernelConfig | None = None,
) -> Tensor:
    """Selective state-space scan forward (SISO, Mamba-1 recurrence).

    Args/semantics mirror ``reference_forward_chunked_scan``:
    ``u``/``delta`` [B, L, D], ``A`` [D, N], ``B``/``C`` [B, L, N],
    ``D`` [D]; returns ``y`` [B, L, D] in ``u``'s dtype.

    Raises:
        ValueError: If L is not divisible by chunk_size.
    """
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    if u.is_cuda and u.dtype in _TRITON_DTYPES and _triton_usable():
        return cast(
            Tensor,
            _ForwardScanCuda.apply(u, delta, A, B, C, D, config),  # type: ignore[no-untyped-call]
        )
    return _scan_eager(u, delta, A, B, C, D)


def triton_scan_resource_meta() -> dict[str, int] | None:
    """Resource metadata of the compiled Triton scan kernel, if any.

    Feed the result to ``gate_res_02_resource_limits`` via the harness;
    None (nothing compiled / no triton) keeps the gate not-applicable.
    """
    if not _triton_usable():
        return None
    from flash_mamba_rl.kernels.ops import _triton_fwd_scan

    return _triton_fwd_scan.resource_meta()
