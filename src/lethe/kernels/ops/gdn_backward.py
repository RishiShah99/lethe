"""GDN-2 chunkwise backward candidate op (oracle-faithful eager path)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from lethe.kernels.references.gdn_backward import Gdn2Grads, reference_gdn2_forward

if TYPE_CHECKING:
    from lethe.kernels.references.gdn2_chunkwise_cw import ChunkwiseForwardCW


def gdn2_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    fwd_stash: ChunkwiseForwardCW | None = None,
) -> Gdn2Grads:
    """GDN-2 backward."""
    if q.is_cuda and q.dtype in (torch.bfloat16, torch.float16, torch.float32):
        from lethe.kernels.cute.gdn2_backward import native_gdn2_backward

        native = native_gdn2_backward(
            q,
            k,
            v,
            g,
            b,
            w,
            do,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            stage_b_closed=fwd_stash is not None,
            fwd_stash=fwd_stash,
        )
        if native is not None:
            return native

    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    inputs = (q, k, v, g, b, w)
    create_graph = do.requires_grad
    with torch.enable_grad():
        leaves = [t.detach().to(compute_dtype).requires_grad_(True) for t in inputs]
        o = reference_gdn2_forward(*leaves, scale=scale, use_qk_l2norm=use_qk_l2norm)
        grads = torch.autograd.grad(o, leaves, do.to(compute_dtype), create_graph=create_graph)
    dq, dk, dv, dg, db, dw = (grad.to(t.dtype) for grad, t in zip(grads, inputs, strict=True))
    return Gdn2Grads(dq, dk, dv, dg, db, dw, None)
