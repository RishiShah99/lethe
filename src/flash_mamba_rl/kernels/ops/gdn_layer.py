"""GDN-2 autograd Function and public op — bridge between nn.Module and the native backward.

Dispatch contract: forward runs ``reference_gdn2_forward`` (token-serial oracle, no
gradient tracking on the forward side); backward calls ``gdn2_backward``, which routes
to the native tcgen05 assembly on sm_100 and falls through to the oracle-faithful eager
path elsewhere.  Inputs that are not tensors (scale, use_qk_l2norm) are non-differentiable
ctx fields.

Public surface: ``gdn2_op(q, k, v, g, b, w, ...)`` -> ``o`` [B, L, H, d_v].  Shape
conventions mirror ``reference_gdn2_forward``; supported dtypes are fp32, fp64, fp16, bf16
(half dtypes are accepted — the backward op handles upcasting).
"""

from __future__ import annotations

import os
from typing import Any, cast

import torch
from torch import Tensor

from flash_mamba_rl.kernels.ops.gdn_backward import gdn2_backward
from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_forward


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
        # Forward must not participate in the backward graph — evaluate in fp32/fp64
        # so the forward implementations (which reject half) can run; save originals.
        compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        ctx.fwd_stash = None
        with torch.no_grad():
            if os.environ.get("FMR_SAVE_FWD"):
                # Lever 2b (the fla move): run the chunkwise forward and keep its
                # chunk-local intermediates so the backward skips the restage. The
                # 1.07 GB decay_rel stash is dropped — holding it across fwd->bwd
                # costs more than the one rebuild the closed stage B does.
                from flash_mamba_rl.kernels.cute.gdn2_assemble import pick_chunk_len
                from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import (
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
        # Function.backward runs with grad mode disabled; the native assembly's
        # supporting stages take autograd VJPs internally (stage B, L2-norm), so
        # they need grad mode on — the eager path enables it itself, the native
        # path must inherit it from here.
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
    """GDN-2 differentiable op.

    Args
    ----
    q, k, g, b : (batch, seqlen, nheads, d_k)  — query, key, log-decay, erase gate
    v, w       : (batch, seqlen, nheads, d_v)  — value, write gate
    scale      : query scale (defaults to ``d_k ** -0.5``)
    use_qk_l2norm : whether to L2-normalise q/k before the scan

    Returns
    -------
    o : (batch, seqlen, nheads, d_v) — same dtype as inputs
    """
    return cast(Tensor, _GDN2Function.apply(q, k, v, g, b, w, scale, use_qk_l2norm))  # type: ignore[no-untyped-call]
