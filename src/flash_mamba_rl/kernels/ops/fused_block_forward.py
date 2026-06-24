"""C5 — hand-written Mamba fused-block forward.

Drop-in for ``reference_fused_block_forward`` with the same signature and
semantics, widened to more dtypes and devices:

- CUDA + {fp32, fp16, bf16} with triton installed -> the two-kernel fused
  path (``_triton_fused_block``: conv1d + SiLU + selective scan in one
  serial-L kernel, deterministic RMSNorm in a second — the why-not-one-
  kernel rationale lives on that module), wrapped in an autograd.Function
  whose backward is the C6 Triton pipeline (first-order only — see
  ``_FusedBlockCuda``).
- everything else (CPU, fp64, missing triton) -> ``_fused_eager``, a
  differentiable eager-torch path replicating the reference op-for-op.
  For fp32 on the same device it is bitwise-equal to the reference.
  fp64 is a verification-only dtype (gradcheck); half dtypes upcast once
  and round once at the output (the mixed-precision contract — the
  reference rejects non-fp32).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from flash_mamba_rl.kernels.autotune import KernelConfig

from .forward_chunked_scan import (
    _TRITON_DTYPES,
    _auto_chunk_len,
    _resolve_scan_mode,
    _scan_eager,
    _triton_usable,
)


def _fused_eager(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    *,
    eps: float,
) -> Tensor:
    """Reference math, replicated op-for-op, in fp32 (fp64 stays fp64)."""
    compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    inputs = (x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
    xc, wc, bc, dc, ac, bpc, cpc, dsc, nwc = (t.to(compute_dtype) for t in inputs)

    conv_out = F.conv1d(xc.transpose(1, 2), wc, bc, groups=xc.shape[-1]).transpose(1, 2)
    z = F.silu(conv_out)
    # The reference's scan output inherits z's transposed-conv strides, and
    # torch's last-dim mean reduction rounds differently on strided vs
    # contiguous memory; restore that layout so the norm is bit-identical.
    y_scan = _scan_eager(z, dc, ac, bpc, cpc, dsc).transpose(1, 2).contiguous().transpose(1, 2)
    rms = y_scan.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return (y_scan / rms * nwc).to(x.dtype)


class _FusedBlockCuda(torch.autograd.Function):
    """Triton forward; backward is the hand-written Triton pipeline (C6).

    First-order only: the Triton backward returns plain tensors with no
    graph, so double-backward through the CUDA path is silently absent
    rather than an error — route through the eager path (CPU or fp64)
    when higher-order gradients are needed. All nine gradients are
    computed unconditionally; ``needs_input_grad`` is not consulted
    (autograd drops the extras — C2's convention).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        conv_weight: Tensor,
        conv_bias: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        norm_weight: Tensor,
        eps: float,
        config: KernelConfig | None = None,
    ) -> Tensor:
        from flash_mamba_rl.kernels.ops import _triton_fused_block

        ctx.save_for_backward(x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight)
        ctx.eps = eps
        ctx.config = config
        return _triton_fused_block.launch_fused_block_forward(
            x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, eps, config=config
        )

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        x, conv_weight, conv_bias, delta, a, b, c, d_skip, norm_weight = ctx.saved_tensors
        config = ctx.config
        l_out = delta.shape[1]
        batch, _, width = x.shape
        if _resolve_scan_mode(config, l_out, batch, width, is_forward=False) == "chunk_parallel":
            from flash_mamba_rl.kernels.ops import _triton_chunk_parallel_fused_bwd

            k = _auto_chunk_len(l_out, config.chunk_len if config is not None else None)
            grads = _triton_chunk_parallel_fused_bwd.launch_fused_block_backward_chunk_parallel(
                x,
                conv_weight,
                conv_bias,
                delta,
                a,
                b,
                c,
                d_skip,
                norm_weight,
                grad_y,
                ctx.eps,
                chunk_len=k,
                config=config,
            )
        else:
            from flash_mamba_rl.kernels.ops import _triton_fused_block_bwd

            grads = _triton_fused_block_bwd.launch_fused_block_backward(
                x,
                conv_weight,
                conv_bias,
                delta,
                a,
                b,
                c,
                d_skip,
                norm_weight,
                grad_y,
                ctx.eps,
                config=config,
            )
        return (*grads, None, None)


def fused_block_forward(
    x: Tensor,
    conv_weight: Tensor,
    conv_bias: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    norm_weight: Tensor,
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
    config: KernelConfig | None = None,
) -> Tensor:
    """Mamba fused-block forward: conv1d -> SiLU -> selective scan -> RMSNorm.

    Args/semantics mirror ``reference_fused_block_forward``: ``x`` [B, L, D]
    carries the causal left-padding (valid conv, L_out = L - (K-1)),
    ``conv_weight`` [D, 1, K], ``conv_bias``/``D``/``norm_weight`` [D],
    ``delta`` [B, L_out, D], ``A`` [D, N], ``B``/``C`` [B, L_out, N];
    returns ``y`` [B, L_out, D] in ``x``'s dtype. All floating inputs share
    ``x``'s dtype (dispatch keys on ``x`` alone, the C1-C4 family contract).
    ``chunk_size`` is validated identically to the scan ops but is a
    blocking hint only — the result does not depend on it.

    Raises:
        ValueError: If ``conv_kernel_size`` disagrees with ``conv_weight``'s
            trailing dim (the reference silently keys on the weight; a
            mismatch would shift every downstream shape), or if L_out is
            not divisible by ``chunk_size``.
    """
    conv_k = conv_weight.shape[-1]
    if conv_k != conv_kernel_size:
        raise ValueError(
            f"conv_kernel_size={conv_kernel_size} disagrees with conv_weight K={conv_k}"
        )
    l_out = x.shape[1] - (conv_k - 1)
    if l_out % chunk_size != 0:
        raise ValueError(f"output length {l_out} must be divisible by chunk_size {chunk_size}")
    if x.is_cuda and x.dtype in _TRITON_DTYPES and _triton_usable():
        return cast(
            Tensor,
            _FusedBlockCuda.apply(  # type: ignore[no-untyped-call]
                x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, eps, config
            ),
        )
    return _fused_eager(x, conv_weight, conv_bias, delta, A, B, C, D, norm_weight, eps=eps)


def triton_fused_block_resource_meta() -> dict[str, int] | None:
    """Resource metadata over both compiled fused-block kernels, if any.

    Feed the result to ``gate_res_02_resource_limits`` via the harness;
    None (nothing compiled / no triton) keeps the gate not-applicable.
    """
    if not _triton_usable():
        return None
    from flash_mamba_rl.kernels.ops import _triton_fused_block

    return _triton_fused_block.resource_meta()
