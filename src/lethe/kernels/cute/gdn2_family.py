"""Family entry wrappers, GLA / LA / SSD-class / KDA modes of the GDN-2 assembly."""

from __future__ import annotations

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_assemble import (
    K1FnCW,
    K2FnCW,
    assemble_gdn2_backward_channelwise,
    assemble_gdn2_backward_no_erase,
)
from lethe.kernels.references.family_oracles import (
    GlaGrads,
    KdaGrads,
    LaGrads,
    SsdGrads,
)

_HALF_DTYPES = (torch.float16, torch.bfloat16)


def gla_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
    fast_path: bool = True,
) -> GlaGrads:
    """GLA (per-channel decay, no erase): ``b = 0, w = 1``; returns (dq, dk, dv, dg)."""
    dq, dk, dv, dg, _ = _no_erase_family_backward(
        q,
        k,
        v,
        g,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_fn,
        k2_fn=k2_fn,
        fast_path=fast_path,
    )
    return GlaGrads(grad_q=dq, grad_k=dk, grad_v=dv, grad_g=dg)


def la_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
    fast_path: bool = True,
) -> LaGrads:
    """Plain linear attention: ``g = 0, b = 0, w = 1``; returns (dq, dk, dv)."""
    g = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
    dq, dk, dv, _, _ = _no_erase_family_backward(
        q,
        k,
        v,
        g,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_fn,
        k2_fn=k2_fn,
        fast_path=fast_path,
    )
    return LaGrads(grad_q=dq, grad_k=dk, grad_v=dv)


def ssd_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
    fast_path: bool = True,
) -> SsdGrads:
    """SSD-class (scalar per-head decay, no erase): ``g`` [B, L, H] broadcast across d_k."""
    if g.dim() != 3:
        raise ValueError(f"SSD-class g must be [B, L, H], got {tuple(g.shape)}")
    g_cw = g.unsqueeze(-1).expand(*g.shape, q.shape[-1])
    dq, dk, dv, dg_cw, half_dtype = _no_erase_family_backward(
        q,
        k,
        v,
        g_cw,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_fn,
        k2_fn=k2_fn,
        fast_path=fast_path,
        skip_half_round=True,
    )
    dg = dg_cw.sum(-1)
    if half_dtype is not None:
        dq, dk, dv, dg = (t.to(half_dtype) for t in (dq, dk, dv, dg))
    return SsdGrads(grad_q=dq, grad_k=dk, grad_v=dv, grad_g=dg)


def kda_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
) -> KdaGrads:
    """KDA (per-channel decay, scalar β): ``b = w = β·1``, the full WY machinery."""
    if beta.dim() != 3:
        raise ValueError(f"KDA beta must be [B, L, H], got {tuple(beta.shape)}")
    in_dtype = q.dtype
    half = in_dtype in _HALF_DTYPES
    if half:
        q, k, v, g, beta, do = (t.to(torch.float32) for t in (q, k, v, g, beta, do))

    b = beta.unsqueeze(-1).expand(*beta.shape, q.shape[-1])
    w = beta.unsqueeze(-1).expand(*beta.shape, v.shape[-1])
    grads = assemble_gdn2_backward_channelwise(
        q,
        k,
        v,
        g,
        b,
        w,
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_fn,
        k2_fn=k2_fn,
        create_graph=do.requires_grad,
    )
    out = KdaGrads(
        grad_q=grads.dq,
        grad_k=grads.dk,
        grad_v=grads.dv,
        grad_g=grads.dg,
        grad_beta=grads.db.sum(-1) + grads.dw.sum(-1),
    )
    if half:
        out = KdaGrads(*(t.to(in_dtype) for t in out))
    return out


def _no_erase_family_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    do: Tensor,
    *,
    scale: float | None,
    use_qk_l2norm: bool,
    k1_fn: K1FnCW | None,
    k2_fn: K2FnCW | None,
    fast_path: bool,
    skip_half_round: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor, torch.dtype | None]:
    """Shared ``b = 0, w = 1`` route: (dq, dk, dv, dg, in_dtype_if_half) with dg per-channel."""
    if fast_path and k2_fn is not None:
        raise ValueError("fast_path skips K#2; pass fast_path=False to inject k2_fn")
    in_dtype = q.dtype
    half = in_dtype in _HALF_DTYPES
    if half:
        q, k, v, g, do = (t.to(torch.float32) for t in (q, k, v, g, do))

    w = torch.ones(v.shape, dtype=v.dtype, device=v.device)
    if fast_path:
        ne = assemble_gdn2_backward_no_erase(
            q,
            k,
            v,
            g,
            w,
            do,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            k1_fn=k1_fn,
            create_graph=do.requires_grad,
        )
        dq, dk, dv, dg = ne.dq, ne.dk, ne.dv, ne.dg
    else:
        b = torch.zeros(q.shape, dtype=q.dtype, device=q.device)
        cw = assemble_gdn2_backward_channelwise(
            q,
            k,
            v,
            g,
            b,
            w,
            do,
            scale=scale,
            use_qk_l2norm=use_qk_l2norm,
            k1_fn=k1_fn,
            k2_fn=k2_fn,
            create_graph=do.requires_grad,
        )
        dq, dk, dv, dg = cw.dq, cw.dk, cw.dv, cw.dg

    if half and not skip_half_round:
        dq, dk, dv, dg = (t.to(in_dtype) for t in (dq, dk, dv, dg))
        return dq, dk, dv, dg, None
    if half and skip_half_round:
        return dq, dk, dv, dg, in_dtype
    return dq, dk, dv, dg, None
