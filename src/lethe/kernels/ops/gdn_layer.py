"""GDN-2 autograd Function and public op, the bridge between nn.Module and the native backward."""

from __future__ import annotations

import os
from typing import Any, cast

import torch
from torch import Tensor

from lethe.kernels.ops.gdn_backward import gdn2_backward
from lethe.kernels.references.gdn_backward import reference_gdn2_forward


class _GDN2Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        b: Tensor,
        w: Tensor,
        scale: float | None,
        use_qk_l2norm: bool,
    ) -> Tensor:
        # Forward runs under no_grad in fp32/fp64: the underlying implementations reject half dtypes.
        compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        ctx.fwd_stash = None
        with torch.no_grad():
            if os.environ.get("FMR_SAVE_FWD"):
                # Run the chunkwise forward, keeping chunk-local intermediates so backward skips the restage.
                from lethe.kernels.cute.gdn2_assemble import pick_chunk_len
                from lethe.kernels.references.gdn2_chunkwise_cw import (
                    chunkwise_restage_cw,
                )

                rst = chunkwise_restage_cw(
                    q.to(compute_dtype),
                    k.to(compute_dtype),
                    v.to(compute_dtype),
                    g.to(compute_dtype),
                    b.to(compute_dtype),
                    w.to(compute_dtype),
                    chunk_len=pick_chunk_len(q.shape[1]),
                    scale=scale,
                    use_qk_l2norm=use_qk_l2norm,
                )
                o = rst.o.to(q.dtype)
                rst.decay_rel = None
                ctx.fwd_stash = rst
            else:
                o = reference_gdn2_forward(
                    q.to(compute_dtype),
                    k.to(compute_dtype),
                    v.to(compute_dtype),
                    g.to(compute_dtype),
                    b.to(compute_dtype),
                    w.to(compute_dtype),
                    scale=scale,
                    use_qk_l2norm=use_qk_l2norm,
                ).to(q.dtype)
        ctx.save_for_backward(q, k, v, g, b, w)
        ctx.scale = scale
        ctx.use_qk_l2norm = use_qk_l2norm
        return o

    @staticmethod
    def backward(ctx: Any, do: Tensor) -> tuple[Tensor | None, ...]:
        q, k, v, g, b, w = ctx.saved_tensors
        # backward runs under no_grad, but stage B / L2-norm VJPs inside gdn2_backward need grad mode on.
        with torch.enable_grad():
            grads = gdn2_backward(
                q,
                k,
                v,
                g,
                b,
                w,
                do.detach(),
                scale=ctx.scale,
                use_qk_l2norm=ctx.use_qk_l2norm,
                fwd_stash=ctx.fwd_stash,
            )
        # 8 inputs: q, k, v, g, b, w, scale, use_qk_l2norm
        return (
            grads.grad_q,
            grads.grad_k,
            grads.grad_v,
            grads.grad_g,
            grads.grad_b,
            grads.grad_w,
            None,
            None,
        )


def gdn2_op(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> Tensor:
    """GDN-2 differentiable op."""
    return cast(Tensor, _GDN2Function.apply(q, k, v, g, b, w, scale, use_qk_l2norm))  # type: ignore[no-untyped-call]
