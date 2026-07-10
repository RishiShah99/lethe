"""Scalar-GDN backward ASSEMBLY, composes the two native kernels into six grads."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from lethe.kernels.references.gdn2_chunkwise import (
    ChunkwiseForward,
    chunkwise_forward,
    masked_decay_ratio,
)
from lethe.kernels.references.gdn2_chunkwise_cw import (
    ChunkwiseForwardCW,
    chunkwise_forward_cw,
    chunkwise_restage_cw,
    masked_decay_rel,
)
from lethe.kernels.references.gdn_backward import Gdn2Grads

RCP_LN2 = 1.0 / math.log(2.0)
LN2 = math.log(2.0)

# Kernel callables match the compiled tcgen05 signatures (run_k1_incB/run_k2) and the torch refs.
K1Fn = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor],
]
K2Fn = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor],
]

# Channel-wise (Phase 3) kernel callables.
K1FnCW = K1Fn
K2FnCW = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _from_chunks(x: Tensor) -> Tensor:
    """[B, H, NT, C, D] -> [B, T, H, D]."""
    b, h, nt, c, d = x.shape
    return x.reshape(b, h, nt * c, d).transpose(1, 2).contiguous()


def _to_chunks(x: Tensor, chunk_len: int) -> Tensor:
    """[B, T, H, D] -> [B, H, NT, C, D]."""
    b, t, h, d = x.shape
    return x.transpose(1, 2).reshape(b, h, t // chunk_len, chunk_len, d)


def pick_chunk_len(seqlen: int) -> int:
    """Largest chunk in [1, 64] dividing ``seqlen`` (prefers 64; falls to 1)."""
    for c in (64, 32, 16, 8, 4, 2, 1):
        if seqlen % c == 0:
            return c
    return 1


def k1_reverse_state_ref(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """K#1 reference, reverse inter-chunk state scan (B4), kernel statement order."""
    b, h, nt, c, _d_k = q.shape
    d_v = do.shape[-1]
    gamma = torch.exp2(g2)
    b_dh = dht.clone()
    dh = torch.zeros_like(dht).unsqueeze(2).repeat(1, 1, nt, 1, 1)
    dv2 = torch.zeros(b, h, nt, c, d_v, dtype=q.dtype, device=q.device)

    for it in reversed(range(nt)):
        dh[:, :, it] = b_dh
        b_dv = k[:, :, it] @ b_dh
        decay = torch.exp2(g_last[:, :, it][..., None] - g2[:, :, it])
        b_dv = b_dv * decay[..., None] + dv_local[:, :, it]
        dv2[:, :, it] = b_dv
        qg = q[:, :, it] * gamma[:, :, it][..., None]
        t = qg.transpose(-1, -2) @ do[:, :, it] - w[:, :, it].transpose(-1, -2) @ b_dv
        b_dh = torch.exp2(g_last[:, :, it])[..., None, None] * b_dh + t

    return dh, dv2, b_dh


def k2_wy_vjp_ref(
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dw: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """K#2 reference, WY / triangular-inverse VJP (B6)."""
    b, h, nt, c, _ = k.shape

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _flat(k), _flat(v), _flat(beta), _flat(g2), _flat(t_mat)
    dwf, duf = _flat(dw), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    bv = betaf[..., None] * vf
    bgk = (betaf * gamma)[..., None] * kf
    kk = kf @ kf.transpose(-1, -2)
    ratio = masked_decay_ratio(g2f)

    tt = tf.transpose(-1, -2)
    dbv = tt @ duf
    dbgk = tt @ dwf
    d_t = duf @ bv.transpose(-1, -2) + dwf @ bgk.transpose(-1, -2)
    d_m = torch.where(sl, -(tt @ d_t @ tt), torch.zeros_like(d_t))

    d_kk = d_m * betaf[..., :, None] * ratio
    dk = (d_kk + d_kk.transpose(-1, -2)) @ kf + (betaf * gamma)[..., None] * dbgk
    dv = betaf[..., None] * dbv
    p = (dbgk * kf).sum(-1)
    r = (dbv * vf).sum(-1)
    db = r + gamma * p + (d_m * (ratio * kk)).sum(-1)

    e = (d_m * betaf[..., :, None] * kk) * ratio
    dg2_total = LN2 * (e.sum(-1) - e.sum(-2)) + (betaf * p) * LN2 * gamma
    dg_tok = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2_total, [-1]), -1), [-1])

    def _unflat(x: Tensor) -> Tensor:
        return x.reshape(b, h, nt, *x.shape[1:])

    return _unflat(dk), _unflat(dv), _unflat(db), _unflat(dg_tok)


def _k2_beta_split(
    k: Tensor, v: Tensor, beta: Tensor, g2: Tensor, t_mat: Tensor, dw: Tensor, du: Tensor
) -> tuple[Tensor, Tensor]:
    """Split the combined beta grad into (erase=key-side, write=value-side) parts."""
    b, h, nt, c, _ = k.shape

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _flat(k), _flat(v), _flat(beta), _flat(g2), _flat(t_mat)
    dwf, duf = _flat(dw), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    bv = betaf[..., None] * vf
    bgk = (betaf * gamma)[..., None] * kf
    kk = kf @ kf.transpose(-1, -2)
    ratio = masked_decay_ratio(g2f)
    tt = tf.transpose(-1, -2)
    dbv = tt @ duf
    dbgk = tt @ dwf
    d_t = duf @ bv.transpose(-1, -2) + dwf @ bgk.transpose(-1, -2)
    d_m = torch.where(sl, -(tt @ d_t @ tt), torch.zeros_like(d_t))

    p = (dbgk * kf).sum(-1)
    db_erase = gamma * p + (d_m * (ratio * kk)).sum(-1)
    dw_write = (dbv * vf).sum(-1)

    def _unflat(x: Tensor) -> Tensor:
        return x.reshape(b, h, nt, *x.shape[1:])

    return _unflat(db_erase), _unflat(dw_write)


def _stage_b_vjp(
    fwd: ChunkwiseForward, do: Tensor, *, create_graph: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """VJP of Stage B (u, w, h0 cut as leaves)."""
    dtype, dev = fwd.q.dtype, fwd.q.device
    b, h, nt, c, _d_k = fwd.q.shape

    q_sn = fwd.q.detach().clone().requires_grad_(True)
    k_n = fwd.k.detach().clone().requires_grad_(True)
    u_lf = fwd.u.detach().clone().requires_grad_(True)
    w_lf = fwd.w.detach().clone().requires_grad_(True)
    h0_lf = fwd.h_list[0].detach().clone().requires_grad_(True)

    g_tok = torch.diff(
        fwd.g2 / RCP_LN2, dim=-1, prepend=torch.zeros(b, h, nt, 1, dtype=dtype, device=dev)
    )
    g_lf = g_tok.detach().clone().requires_grad_(True)

    g2 = torch.cumsum(g_lf, dim=-1) * RCP_LN2
    gamma = torch.exp2(g2)
    g_last = g2[..., -1]
    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), 0)
    do_c = _to_chunks(do, c)

    o_list: list[Tensor] = []
    hh = h0_lf
    for ci in range(nt):
        gamma_c, g2_c, glast_c = gamma[:, :, ci], g2[:, :, ci], g_last[:, :, ci]
        ratio = masked_decay_ratio(g2_c)
        a_qk = (q_sn[:, :, ci] @ k_n[:, :, ci].transpose(-1, -2)) * ratio
        a_qk = torch.where(lower_incl, a_qk, torch.zeros_like(a_qk))
        v_new = u_lf[:, :, ci] - w_lf[:, :, ci] @ hh
        qg = q_sn[:, :, ci] * gamma_c[..., None]
        o_list.append(qg @ hh + a_qk @ v_new)
        decay_end = torch.exp2(glast_c[..., None] - g2_c)
        k_dec = k_n[:, :, ci] * decay_end[..., None]
        hh = torch.exp2(glast_c)[..., None, None] * hh + k_dec.transpose(-1, -2) @ v_new

    o = torch.stack(o_list, dim=2)
    dq_b, dk_b, dg_b, dw, du, dh0 = torch.autograd.grad(
        o, (q_sn, k_n, g_lf, w_lf, u_lf, h0_lf), do_c, create_graph=create_graph
    )
    return dq_b, dk_b, dg_b, dw, du, dh0


@dataclass
class ScalarGdn2Grads:
    """Scalar-GDN backward output: six grads (dq/dk/dv, dg/db_erase/dw_write scalar) + dh0."""

    dq: Tensor
    dk: Tensor
    dv: Tensor
    dg: Tensor
    db_erase: Tensor
    dw_write: Tensor
    dh0: Tensor


def assemble_gdn2_backward_scalar(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    *,
    chunk_len: int | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1Fn | None = None,
    k2_fn: K2Fn | None = None,
    create_graph: bool = False,
) -> ScalarGdn2Grads:
    """Assemble the six scalar-GDN grads from K#1 + K#2 + supporting torch stages."""
    k1 = k1_fn if k1_fn is not None else k1_reverse_state_ref
    k2 = k2_fn if k2_fn is not None else k2_wy_vjp_ref
    cl = pick_chunk_len(q.shape[1]) if chunk_len is None else chunk_len

    fwd = chunkwise_forward(
        q, k, v, g, beta, chunk_len=cl, scale=scale, use_qk_l2norm=use_qk_l2norm
    )
    dq_b, dk_b, dg_b, dw, _du_b, _dh0_b = _stage_b_vjp(fwd, do, create_graph=create_graph)

    do_c = _to_chunks(do, cl)
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(fwd.h_list[0])

    # K#1/K#2 route through DLPack, which rejects grad-requiring tensors, so operands get detached.
    def _det(t: Tensor) -> Tensor:
        return t if create_graph else t.detach()

    _dh, dv2, dh0 = k1(
        _det(fwd.q),
        _det(fwd.k),
        _det(fwd.w),
        _det(fwd.g2),
        _det(fwd.g_last),
        _det(do_c),
        _det(dv_local),
        _det(dht),
    )

    dk2, dv, _db, dg2 = k2(
        _det(fwd.k), _det(fwd.v), _det(fwd.beta), _det(fwd.g2), _det(fwd.T), _det(dw), _det(dv2)
    )
    db_erase, dw_write = _k2_beta_split(
        _det(fwd.k), _det(fwd.v), _det(fwd.beta), _det(fwd.g2), _det(fwd.T), _det(dw), _det(dv2)
    )

    dq_n = _from_chunks(dq_b)
    dk_n = _from_chunks(dk_b + dk2)
    dg = _from_chunks((dg_b + dg2).unsqueeze(-1)).squeeze(-1)
    dv_full = _from_chunks(dv)
    db_e = _from_chunks(db_erase.unsqueeze(-1)).squeeze(-1)
    dw_w = _from_chunks(dw_write.unsqueeze(-1)).squeeze(-1)

    # B8: L2-norm (+ scale) VJP back to the raw query/key.
    if use_qk_l2norm:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        q_lf = q.detach().clone().requires_grad_(True)
        k_lf = k.detach().clone().requires_grad_(True)
        q_sn = _l2norm(q_lf) * s
        k_nn = _l2norm(k_lf)
        dq, dk = torch.autograd.grad(
            (q_sn, k_nn), (q_lf, k_lf), (dq_n, dk_n), create_graph=create_graph
        )
    else:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        dq, dk = dq_n * s, dk_n

    return ScalarGdn2Grads(dq=dq, dk=dk, dv=dv_full, dg=dg, db_erase=db_e, dw_write=dw_w, dh0=dh0)


def assembled_scalar_gdn2_backward(
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
    k1_fn: K1Fn | None = None,
    k2_fn: K2Fn | None = None,
) -> Gdn2Grads:
    """GDN-2-signature wrapper around the scalar assembly (the dispatch + gate entry)."""
    d_k = g.shape[-1]
    d_v = w.shape[-1]
    in_dtype = q.dtype
    half = in_dtype in (torch.float16, torch.bfloat16)
    if half:
        q, k, v, g, b, w, do = (t.to(torch.float32) for t in (q, k, v, g, b, w, do))

    grads = assemble_gdn2_backward_scalar(
        q,
        k,
        v,
        g[..., 0],
        b[..., 0],
        do,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        k1_fn=k1_fn,
        k2_fn=k2_fn,
        create_graph=do.requires_grad,
    )
    grad_g = grads.dg.unsqueeze(-1).expand(*grads.dg.shape, d_k) / d_k
    grad_b = grads.db_erase.unsqueeze(-1).expand(*grads.db_erase.shape, d_k) / d_k
    grad_w = grads.dw_write.unsqueeze(-1).expand(*grads.dw_write.shape, d_v) / d_v
    out = Gdn2Grads(
        grad_q=grads.dq,
        grad_k=grads.dk,
        grad_v=grads.dv,
        grad_g=grad_g.contiguous(),
        grad_b=grad_b.contiguous(),
        grad_w=grad_w.contiguous(),
        grad_initial_state=None,
    )
    if half:
        out = Gdn2Grads(
            grad_q=out.grad_q.to(in_dtype),
            grad_k=out.grad_k.to(in_dtype),
            grad_v=out.grad_v.to(in_dtype),
            grad_g=out.grad_g.to(in_dtype),
            grad_b=out.grad_b.to(in_dtype),
            grad_w=out.grad_w.to(in_dtype),
            grad_initial_state=None,
        )
    return out


def k1_reverse_state_cw_ref(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Channel-wise K#1, reverse inter-chunk state scan (B4), kernel statement order."""
    b, h, nt, c, _d_k = q.shape
    d_v = do.shape[-1]
    gamma = torch.exp2(g2)
    b_dh = dht.clone()
    dh = torch.zeros_like(dht).unsqueeze(2).repeat(1, 1, nt, 1, 1)
    dv2 = torch.zeros(b, h, nt, c, d_v, dtype=q.dtype, device=q.device)

    for it in reversed(range(nt)):
        dh[:, :, it] = b_dh
        decay_end = torch.exp2(g_last[:, :, it][..., None, :] - g2[:, :, it])  # [B,H,C,d_k]
        b_dv = (k[:, :, it] * decay_end) @ b_dh + dv_local[:, :, it]
        dv2[:, :, it] = b_dv
        qg = q[:, :, it] * gamma[:, :, it]
        t = qg.transpose(-1, -2) @ do[:, :, it] - wy[:, :, it].transpose(-1, -2) @ b_dv
        b_dh = torch.exp2(g_last[:, :, it])[..., :, None] * b_dh + t

    return dh, dv2, b_dh


def k2_wy_vjp_cw_ref(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Channel-wise K#2, WY/triangular-inverse VJP (B6)."""
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = (
        _flat(k),
        _flat(v),
        _flat(b),
        _flat(w),
        _flat(g2),
        _flat(t_mat),
    )
    dwyf, duf = _flat(dwy), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)  # [N,C,d_k]
    decay_rel = masked_decay_rel(g2f)  # [N,C,C,d_k]
    bk = bf * kf  # [N,C,d_k]
    pwv = wf * vf  # write term P = w (.) v  [N,C,d_v]
    q_kbg = bk * gamma  # Q = b (.) gamma (.) k  [N,C,d_k]

    tt = tf.transpose(-1, -2)
    dp = tt @ duf  # [N,C,d_v]
    dq_wy = tt @ dwyf  # [N,C,d_k]
    d_t = duf @ pwv.transpose(-1, -2) + dwyf @ q_kbg.transpose(-1, -2)  # [N,C,C]
    d_m = torch.where(sl, -(tt @ d_t @ tt), torch.zeros_like(d_t))  # [N,C,C]

    dv = dp * wf  # from P = w (.) v
    dw = dp * vf

    dbk_q = dq_wy * gamma  # from Q's bk factor
    dg2_q = dq_wy * q_kbg * LN2  # from Q's gamma factor (gamma = exp2(g2))

    dbk_m = torch.einsum("nis,nsd,nisd->nid", d_m, kf, decay_rel)
    dk_m = torch.einsum("nis,nid,nisd->nsd", d_m, bk, decay_rel)
    # E = dM*bk*k*decay_rel never materializes; dg2_m reuses rowsum/colsum via dbk_m/dk_m.
    dg2_m = LN2 * (bk * dbk_m - kf * dk_m)  # [N,C,d_k]

    dbk = dbk_q + dbk_m
    db = dbk * kf
    dk = dbk * bf + dk_m
    dg2 = dg2_q + dg2_m
    dg = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [1]), 1), [1])  # reverse-cumsum over C

    def _unflat(x: Tensor, last: int) -> Tensor:
        return x.reshape(bsz, hh, nt, c, last)

    return (
        _unflat(dk, d_k),
        _unflat(dv, d_v),
        _unflat(db, d_k),
        _unflat(dw, d_v),
        _unflat(dg, d_k),
    )


def _stage_b_vjp_cw(
    fwd: ChunkwiseForwardCW, do: Tensor, *, create_graph: bool
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """VJP of channel-wise Stage B (u, wy, h0 cut as leaves)."""
    dtype, dev = fwd.q.dtype, fwd.q.device
    b, h, nt, c, d_k = fwd.q.shape

    q_sn = fwd.q.detach().clone().requires_grad_(True)
    k_n = fwd.k.detach().clone().requires_grad_(True)
    u_lf = fwd.u.detach().clone().requires_grad_(True)
    wy_lf = fwd.wy.detach().clone().requires_grad_(True)
    h0_lf = fwd.h_list[0].detach().clone().requires_grad_(True)

    g_tok = torch.diff(
        fwd.g2 / RCP_LN2, dim=-2, prepend=torch.zeros(b, h, nt, 1, d_k, dtype=dtype, device=dev)
    )
    g_lf = g_tok.detach().clone().requires_grad_(True)

    g2 = torch.cumsum(g_lf, dim=-2) * RCP_LN2
    gamma = torch.exp2(g2)
    g_last = g2[..., -1, :]
    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), 0)
    do_c = _to_chunks(do, c)

    o_list: list[Tensor] = []
    hh = h0_lf
    for ci in range(nt):
        gamma_c, g2_c, glast_c = gamma[:, :, ci], g2[:, :, ci], g_last[:, :, ci]
        decay_rel = masked_decay_rel(g2_c)
        a_qk = torch.einsum("...id,...sd,...isd->...is", q_sn[:, :, ci], k_n[:, :, ci], decay_rel)
        a_qk = torch.where(lower_incl, a_qk, torch.zeros_like(a_qk))
        v_new = u_lf[:, :, ci] - wy_lf[:, :, ci] @ hh
        qg = q_sn[:, :, ci] * gamma_c
        o_list.append(qg @ hh + a_qk @ v_new)
        decay_end = torch.exp2(glast_c[..., None, :] - g2_c)
        k_dec = k_n[:, :, ci] * decay_end
        hh = torch.exp2(glast_c)[..., :, None] * hh + k_dec.transpose(-1, -2) @ v_new

    o = torch.stack(o_list, dim=2)
    dq_b, dk_b, dg_b, dwy, du, dh0 = torch.autograd.grad(
        o, (q_sn, k_n, g_lf, wy_lf, u_lf, h0_lf), do_c, create_graph=create_graph
    )
    return dq_b, dk_b, dg_b, dwy, du, dh0


def _stage_b_vjp_cw_closed(
    fwd: ChunkwiseForwardCW,
    do: Tensor,
    dh: Tensor,
    dv2: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Closed-form channel-wise Stage-B VJP, chunk-local, batched over [B·H·NT]."""

    def _det(t: Tensor) -> Tensor:
        return t if create_graph else t.detach()

    do_c = _to_chunks(_det(do), fwd.chunk_len)
    h_entry = _det(fwd.h[:, :, :-1])  # [B,H,NT,d_k,d_v] chunk ENTRY states
    gamma = _det(fwd.gamma)
    g2, g_last = _det(fwd.g2), _det(fwd.g_last)
    q, k, v_new = _det(fwd.q), _det(fwd.k), _det(fwd.v_new)
    dh, dv2 = _det(dh), _det(dv2)
    c = fwd.chunk_len

    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=do.device), 0)
    decay_end = torch.exp2(g_last[..., None, :] - g2)  # [B,H,NT,C,d_k]

    dqg = do_c @ h_entry.transpose(-1, -2)  # [B,H,NT,C,d_k]
    da_qk = do_c @ v_new.transpose(-1, -2)
    da_qk = torch.where(lower_incl, da_qk, torch.zeros_like(da_qk))

    # Uses the fused SIMT/exp2 kernel on CUDA when available; else torch einsums over decay_rel.
    sbe: tuple[Tensor, Tensor] | None = None
    if (
        not create_graph
        and do.is_cuda
        and g2.dtype == torch.float32  # the kernel is the f32 hot path; fp64 keeps torch
        and not os.environ.get("FMR_DISABLE_SBE")
    ):
        from lethe.kernels.cute.gdn2_sb_einsum import (
            is_available as _sbe_available,
        )
        from lethe.kernels.cute.gdn2_sb_einsum import run_sb_einsum, sbe_dims_ok

        if _sbe_available() and sbe_dims_ok(c, k.shape[-1]):
            sbe = run_sb_einsum(da_qk, k, q, g2)
    if sbe is not None:
        dq_intra, dk_intra = sbe
    else:
        if fwd.decay_rel is not None and not create_graph:
            decay_rel = fwd.decay_rel  # [B,H,NT,C,C,d_k]
        else:
            decay_rel = masked_decay_rel(g2)
        dq_intra = torch.einsum("...is,...sd,...isd->...id", da_qk, k, decay_rel)
        dk_intra = torch.einsum("...is,...id,...isd->...sd", da_qk, q, decay_rel)
    dk_dec = v_new @ dh.transpose(-1, -2)  # [B,H,NT,C,d_k]

    dq_b = dqg * gamma + dq_intra
    dk_b = dk_intra + dk_dec * decay_end
    dwy = -dv2 @ h_entry.transpose(-1, -2)

    dde = dk_dec * k * decay_end
    dg2 = LN2 * (dqg * q * gamma + q * dq_intra - k * dk_intra - dde)
    dg_last = LN2 * (torch.exp2(g_last) * (dh * h_entry).sum(-1) + dde.sum(-2))  # [B,H,NT,d_k]
    dg2 = dg2 + torch.cat([torch.zeros_like(dg2[..., :-1, :]), dg_last.unsqueeze(-2)], dim=-2)
    dg_b = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [-2]), -2), [-2])

    return dq_b, dk_b, dg_b, dwy


@dataclass
class ChannelwiseGdn2Grads:
    """Channel-wise GDN-2 backward output: six grads (dg/db key axis, dw value axis) + dh0."""

    dq: Tensor
    dk: Tensor
    dv: Tensor
    dg: Tensor
    db: Tensor
    dw: Tensor
    dh0: Tensor


def assemble_gdn2_backward_channelwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    chunk_len: int | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
    create_graph: bool = False,
    stage_b_closed: bool = False,
    fwd_stash: ChunkwiseForwardCW | None = None,
) -> ChannelwiseGdn2Grads:
    """Assemble the six channel-wise GDN-2 grads from K#1 + K#2 + supporting torch stages."""
    k1 = k1_fn if k1_fn is not None else k1_reverse_state_cw_ref
    k2 = k2_fn if k2_fn is not None else k2_wy_vjp_cw_ref

    # The closed path uses fwd_stash or a no-grad restage; stage B needs the graphed forward.
    if stage_b_closed and not create_graph and fwd_stash is not None:
        fwd = fwd_stash
        cl = fwd.chunk_len
    elif stage_b_closed and not create_graph:
        cl = pick_chunk_len(q.shape[1]) if chunk_len is None else chunk_len
        fwd = chunkwise_restage_cw(
            q, k, v, g, b, w, chunk_len=cl, scale=scale, use_qk_l2norm=use_qk_l2norm
        )
    else:
        cl = pick_chunk_len(q.shape[1]) if chunk_len is None else chunk_len
        fwd = chunkwise_forward_cw(
            q, k, v, g, b, w, chunk_len=cl, scale=scale, use_qk_l2norm=use_qk_l2norm
        )

    do_c = _to_chunks(do, cl)
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(fwd.h_list[0])

    def _det(t: Tensor) -> Tensor:
        return t if create_graph else t.detach()

    # Default keeps Stage-B-before-K#1 order (graph-safe); the closed path needs K#1's dh first.
    if not stage_b_closed:
        dq_b, dk_b, dg_b, dwy, _du_b, _dh0_b = _stage_b_vjp_cw(fwd, do, create_graph=create_graph)

    dh_k1, dv2, dh0 = k1(
        _det(fwd.q),
        _det(fwd.k),
        _det(fwd.wy),
        _det(fwd.g2),
        _det(fwd.g_last),
        _det(do_c),
        _det(dv_local),
        _det(dht),
    )

    if stage_b_closed:
        dq_b, dk_b, dg_b, dwy = _stage_b_vjp_cw_closed(
            fwd, do, dh_k1, dv2, create_graph=create_graph
        )

    dk2, dv, db, dw, dg2 = k2(
        _det(fwd.k),
        _det(fwd.v),
        _det(fwd.b),
        _det(fwd.w_gate),
        _det(fwd.g2),
        _det(fwd.T),
        _det(dwy),
        _det(dv2),
    )

    dq_n = _from_chunks(dq_b)
    dk_n = _from_chunks(dk_b + dk2)
    dg = _from_chunks(dg_b + dg2)
    dv_full = _from_chunks(dv)
    db_full = _from_chunks(db)
    dw_full = _from_chunks(dw)

    if use_qk_l2norm:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        q_lf = q.detach().clone().requires_grad_(True)
        k_lf = k.detach().clone().requires_grad_(True)
        q_sn = _l2norm(q_lf) * s
        k_nn = _l2norm(k_lf)
        dq, dk = torch.autograd.grad(
            (q_sn, k_nn), (q_lf, k_lf), (dq_n, dk_n), create_graph=create_graph
        )
    else:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        dq, dk = dq_n * s, dk_n

    return ChannelwiseGdn2Grads(dq=dq, dk=dk, dv=dv_full, dg=dg, db=db_full, dw=dw_full, dh0=dh0)


@dataclass
class NoEraseGrads:
    """``b = 0`` (GLA/LA/SSD-class regime) assembly output."""

    dq: Tensor
    dk: Tensor
    dv: Tensor
    dg: Tensor
    dw: Tensor
    dh0: Tensor


def assemble_gdn2_backward_no_erase(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    chunk_len: int | None = None,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
    k1_fn: K1FnCW | None = None,
    create_graph: bool = False,
) -> NoEraseGrads:
    """The ``b = 0`` fast path: skip-T forward, no K#2, K#1 with ``wy = 0``."""
    k1 = k1_fn if k1_fn is not None else k1_reverse_state_cw_ref
    cl = pick_chunk_len(q.shape[1]) if chunk_len is None else chunk_len
    b = torch.zeros_like(g)

    fwd = chunkwise_forward_cw(
        q, k, v, g, b, w, chunk_len=cl, scale=scale, use_qk_l2norm=use_qk_l2norm, skip_erase=True
    )
    dq_b, dk_b, dg_b, _dwy, _du_b, _dh0_b = _stage_b_vjp_cw(fwd, do, create_graph=create_graph)

    do_c = _to_chunks(do, cl)
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c
    dht = torch.zeros_like(fwd.h_list[0])

    def _det(t: Tensor) -> Tensor:
        return t if create_graph else t.detach()

    _dh, dv2, dh0 = k1(
        _det(fwd.q),
        _det(fwd.k),
        _det(fwd.wy),
        _det(fwd.g2),
        _det(fwd.g_last),
        _det(do_c),
        _det(dv_local),
        _det(dht),
    )

    dv = dv2 * _det(fwd.w_gate)
    dw = dv2 * _det(fwd.v)

    dq_n = _from_chunks(dq_b)
    dk_n = _from_chunks(dk_b)
    dg = _from_chunks(dg_b)
    dv_full = _from_chunks(dv)
    dw_full = _from_chunks(dw)

    if use_qk_l2norm:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        q_lf = q.detach().clone().requires_grad_(True)
        k_lf = k.detach().clone().requires_grad_(True)
        q_sn = _l2norm(q_lf) * s
        k_nn = _l2norm(k_lf)
        dq, dk = torch.autograd.grad(
            (q_sn, k_nn), (q_lf, k_lf), (dq_n, dk_n), create_graph=create_graph
        )
    else:
        s = q.shape[-1] ** -0.5 if scale is None else scale
        dq, dk = dq_n * s, dk_n

    return NoEraseGrads(dq=dq, dk=dk, dv=dv_full, dg=dg, dw=dw_full, dh0=dh0)


def assembled_channelwise_gdn2_backward(
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
    k1_fn: K1FnCW | None = None,
    k2_fn: K2FnCW | None = None,
    stage_b_closed: bool = False,
    fwd_stash: ChunkwiseForwardCW | None = None,
) -> Gdn2Grads:
    """GDN-2-signature wrapper around the channel-wise assembly (the gate entry)."""
    in_dtype = q.dtype
    half = in_dtype in (torch.float16, torch.bfloat16)
    if half:
        q, k, v, g, b, w, do = (t.to(torch.float32) for t in (q, k, v, g, b, w, do))

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
        stage_b_closed=stage_b_closed,
        fwd_stash=fwd_stash,
    )
    out = Gdn2Grads(
        grad_q=grads.dq,
        grad_k=grads.dk,
        grad_v=grads.dv,
        grad_g=grads.dg,
        grad_b=grads.db,
        grad_w=grads.dw,
        grad_initial_state=None,
    )
    if half:
        out = Gdn2Grads(
            grad_q=out.grad_q.to(in_dtype),
            grad_k=out.grad_k.to(in_dtype),
            grad_v=out.grad_v.to(in_dtype),
            grad_g=out.grad_g.to(in_dtype),
            grad_b=out.grad_b.to(in_dtype),
            grad_w=out.grad_w.to(in_dtype),
            grad_initial_state=None,
        )
    return out
