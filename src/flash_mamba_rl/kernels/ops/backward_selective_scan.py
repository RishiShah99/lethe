"""C2 — hand-written SISO selective-scan backward (the #904-broken op).

Drop-in for ``reference_backward_selective_scan`` with the same signature
and semantics, widened to more dtypes and devices:

- CUDA + {fp32, fp16, bf16} with triton installed -> the Triton kernel
  (``_triton_bwd_scan``): analytic adjoint with chunked state recompute,
  no ``tl.dot`` anywhere, so Triton's eager TMEM-promotion pass — the
  root cause of the official kernel's num_warps>=4 compile failures on
  Blackwell — never engages.
- everything else (CPU, fp64, missing triton) -> ``torch.autograd.grad``
  through ``_scan_eager``, the differentiable eager path shared with the
  C1 forward op. For fp32 on CPU this replicates the reference oracle's
  gradients; fp64 is the verification dtype (gradcheck) and deliberately
  routes here. When ``dy.requires_grad``, the eager path builds the
  double-backward graph (``create_graph=True``) — the VJP is linear in
  ``dy``, which is exactly what CMP-02's gradcheck differentiates.

Deviations from the reference: the reference rejects non-fp32 inputs; this
op defines the mixed-precision contract instead (compute in fp32, round
once per gradient output). ``chunk_size`` is validated identically but is
a blocking hint only — gradients do not depend on it, and the Triton
kernel picks its own recompute-chunk internally.
"""

from __future__ import annotations

import torch
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig
from flash_mamba_rl.kernels.references.backward_selective_scan import SelectiveScanGrads

from .forward_chunked_scan import _TRITON_DTYPES, _auto_chunk_len, _scan_eager, _triton_usable


def _bwd_eager(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor, dy: Tensor
) -> SelectiveScanGrads:
    """Gradients via autograd through the eager forward — correct by construction."""
    create_graph = dy.requires_grad
    with torch.enable_grad():
        leaves = [t.detach().requires_grad_(True) for t in (u, delta, A, B, C, D)]
        y = _scan_eager(*leaves)
        grads = torch.autograd.grad(y, leaves, dy, create_graph=create_graph)
    return SelectiveScanGrads(*grads)


def backward_selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    dy: Tensor,
    *,
    chunk_size: int = 64,
    config: KernelConfig | None = None,
) -> SelectiveScanGrads:
    """Selective-scan backward pass (SISO, Mamba-1 recurrence).

    Args/semantics mirror ``reference_backward_selective_scan``:
    ``u``/``delta``/``dy`` [B, L, D], ``A`` [D, N], ``B``/``C`` [B, L, N],
    ``D`` [D]; returns a ``SelectiveScanGrads`` named tuple whose fields
    match the corresponding input shapes and dtypes.

    Raises:
        ValueError: If L is not divisible by chunk_size.
    """
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    if u.is_cuda and u.dtype in _TRITON_DTYPES and _triton_usable():
        if config is not None and config.scan_mode == "chunk_parallel":
            from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_bwd

            k = _auto_chunk_len(seq_len, config.chunk_len)
            return SelectiveScanGrads(
                *_triton_chunk_parallel_bwd.launch_backward_chunk_parallel_scan(
                    u, delta, A, B, C, D, dy, chunk_len=k, config=config
                )
            )
        from flash_mamba_rl.kernels.ops import _triton_bwd_scan

        return SelectiveScanGrads(
            *_triton_bwd_scan.launch_backward_scan(u, delta, A, B, C, D, dy, config=config)
        )
    return _bwd_eager(u, delta, A, B, C, D, dy)


def triton_bwd_scan_resource_meta(config: KernelConfig | None = None) -> dict[str, int] | None:
    """Resource metadata of the compiled Triton backward kernel, if any.

    Feed the result to ``gate_res_02_resource_limits`` via the harness;
    None (nothing compiled / no triton) keeps the gate not-applicable. With a
    chunk_parallel config RES-02 must audit the chunk-parallel kernels, not the
    serial one.
    """
    if not _triton_usable():
        return None
    if config is not None and config.scan_mode == "chunk_parallel":
        from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_bwd

        return _triton_chunk_parallel_bwd.resource_meta()
    from flash_mamba_rl.kernels.ops import _triton_bwd_scan

    return _triton_bwd_scan.resource_meta()
