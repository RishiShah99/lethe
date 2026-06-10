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

_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_TRITON: bool | None = None


def _triton_usable() -> bool:
    global _HAS_TRITON
    if _HAS_TRITON is None:
        _HAS_TRITON = _importlib_util.find_spec("triton") is not None
    return _HAS_TRITON


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
    ) -> Tensor:
        from flash_mamba_rl.kernels.ops import _triton_fwd_scan

        ctx.save_for_backward(u, delta, A, B, C, D)
        return _triton_fwd_scan.launch_forward_scan(u, delta, A, B, C, D)

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor, ...]:
        from flash_mamba_rl.kernels.ops import _triton_bwd_scan

        u, delta, a, b, c, d_skip = ctx.saved_tensors
        return _triton_bwd_scan.launch_backward_scan(u, delta, a, b, c, d_skip, grad_y)


def forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
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
            _ForwardScanCuda.apply(u, delta, A, B, C, D),  # type: ignore[no-untyped-call]
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
