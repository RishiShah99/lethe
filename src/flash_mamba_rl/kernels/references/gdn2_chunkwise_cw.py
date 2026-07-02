"""Channel-wise GDN-2 chunkwise backward reference — the Phase-3 crown ground truth.

The Phase-3 native kernels (channel-wise B4 reverse inter-chunk state scan, channel-wise
B6 WY/triangular-inverse VJP) are validated against the intermediates this module exposes.
It is the strict generalization of ``gdn2_chunkwise`` (scalar GDN) to GDN-2's full gating:

* decay ``g`` is **per key-channel** (``[d_k]``), so the within-chunk cumulative decay
  ``gamma`` is ``[C, d_k]`` — the decay no longer factors out of the key-key inner product;
* erase gate ``b in [0,1]^{d_k}`` on the key axis (was the scalar ``beta``);
* write gate ``w in [0,1]^{d_v}`` on the value axis (was the scalar ``beta``).

The algebraic skeleton is identical to the scalar case once the per-channel decay is folded
into the weighted inner products. Writing ``decay_rel[i,s,j] = exp(G_i[j] - G_s[j])`` (G the
inclusive within-chunk cumsum of g; ``<= 0`` on the strict-lower triangle, so bounded):

    M[i,s] = sum_j (b*k)_i[j] * k_s[j] * decay_rel[i,s,j]          (strict lower)
    A[i,s] = sum_j   q_i[j]   * k_s[j] * decay_rel[i,s,j]          (lower incl. diag)
    T      = (I + M)^{-1}
    u      = T (w (.) v)            wy = T (b (.) gamma (.) k)
    v_new  = u - wy @ h            o  = (q (.) gamma) @ h + A @ v_new
    h_next = Diag(gamma_C) h + (k (.) gamma_C/gamma)^T @ v_new

Setting ``g`` channel-constant and ``b = w = beta`` collapses every per-channel quantity to
the scalar form, so this reference reproduces ``gdn2_chunkwise`` to machine precision in that regime
(the Phase-3 -> Phase-2 reduction kill-gate) and the token-serial GDN-2 oracle
(``references/gdn_backward``) end-to-end.

Conventions match ``gdn2_chunkwise``: decays carried in log2 (``exp2`` with ``g`` pre-scaled
by ``RCP_LN2``); ``T`` solved once and reused (its adjoint = autograd through the inverse);
fp32/fp64 only; ``q`` is the pre-scaled, pre-L2-normed query the kernel sees (B8 is external).
Every backward quantity is taken via ``torch.autograd`` through the explicit forward
(autograd-VJP == the hand-VJP the kernels implement).

The WY representation matrix is named ``wy`` here (not ``w``) to free ``w`` for the write gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

RCP_LN2 = 1.0 / math.log(2.0)


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def _to_chunks(x: Tensor, chunk_len: int) -> Tensor:
    """[B, T, H, D] -> [B, H, NT, C, D] (head-major, chunked along T)."""
    b, t, h, d = x.shape
    if t % chunk_len != 0:
        raise ValueError(f"seqlen {t} not divisible by chunk_len {chunk_len}")
    nt = t // chunk_len
    return x.transpose(1, 2).reshape(b, h, nt, chunk_len, d)


def _from_chunks(x: Tensor) -> Tensor:
    """[B, H, NT, C, D] -> [B, T, H, D]."""
    b, h, nt, c, d = x.shape
    return x.reshape(b, h, nt * c, d).transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Forward (canonical chunkwise decomposition, channel-wise) with intermediates
# ---------------------------------------------------------------------------


@dataclass
class ChunkwiseForwardCW:
    """Channel-wise forward output ``o`` plus every intermediate, head-major + chunked.

    Tensors carry grad history (non-leaf) so ``autograd.grad`` can pull backward
    intermediates from them. Chunked shapes use NT chunks of length C.

    o        : [B, T, H, d_v]
    q, k     : [B, H, NT, C, d_k]   (q pre-scaled+normed; k normed)
    v        : [B, H, NT, C, d_v]
    b        : [B, H, NT, C, d_k]   erase gate (key axis)
    w_gate   : [B, H, NT, C, d_v]   write gate (value axis)
    g2       : [B, H, NT, C, d_k]   within-chunk log2 cumsum (per channel)
    g_last   : [B, H, NT, d_k]      g2[..., -1, :]
    gamma    : [B, H, NT, C, d_k]   exp2(g2)
    T        : [B, H, NT, C, C]     (I + M)^{-1}, lower-tri
    u, wy    : [B, H, NT, C, d_v] / [B, H, NT, C, d_k]
    v_new    : [B, H, NT, C, d_v]
    h        : [B, H, NT + 1, d_k, d_v]   entry states; h[...,0]=h0, h[...,NT]=final
    A_qk     : [B, H, NT, C, C]     lower-incl-diag channel-decayed score
    *_list   : per-chunk non-leaf tensors for autograd.grad
    leaves   : (q, k, v, g, b, w, h0) leaves the forward was built from
    """

    o: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    b: Tensor
    w_gate: Tensor
    g2: Tensor
    g_last: Tensor
    gamma: Tensor
    T: Tensor
    u: Tensor
    wy: Tensor
    v_new: Tensor
    h: Tensor
    A_qk: Tensor
    h_list: list[Tensor]
    v_new_list: list[Tensor]
    wy_list: list[Tensor]
    u_list: list[Tensor]
    leaves: tuple[Tensor, ...]
    chunk_len: int
    scale: float
    skip_erase: bool = False


def chunkwise_forward_cw(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    *,
    chunk_len: int = 64,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = False,
    skip_erase: bool = False,
) -> ChunkwiseForwardCW:
    """Channel-wise GDN-2 chunkwise forward, retaining all intermediates.

    Shapes: q, k, g, b [B, T, H, d_k]; v, w [B, T, H, d_v]. ``g`` is per-channel
    log-decay (key axis); ``b``/``w`` are the erase/write gates. The graph is rooted at
    fresh leaves cloned from the inputs so ``autograd.grad`` can be called against both
    leaves and intermediates.

    ``skip_erase`` is the ``b = 0`` (GLA/LA/SSD-class) fast path: with the erase gate
    off, ``M = 0`` so ``T = I`` and ``wy = 0`` exactly — the M-einsum, the triangular
    solve, and the ``wy @ h`` GEMM are skipped, ``u = w ⊙ v``, ``v_new = u``. Output
    equality with the full path at ``b = 0`` is pinned by test (the b path leaves the
    graph, so ``chunkwise_backward_cw`` rejects a skip-erase forward).
    """
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64, got {q.dtype}")
    if skip_erase and bool((b != 0).any()):
        raise ValueError("skip_erase requires b == 0 everywhere")
    bsz, t, h, d_k = q.shape
    d_v = v.shape[-1]
    s = d_k**-0.5 if scale is None else scale

    q_lf = q.detach().clone().requires_grad_(True)
    k_lf = k.detach().clone().requires_grad_(True)
    v_lf = v.detach().clone().requires_grad_(True)
    g_lf = g.detach().clone().requires_grad_(True)
    b_lf = b.detach().clone().requires_grad_(True)
    w_lf = w.detach().clone().requires_grad_(True)
    leaves = [q_lf, k_lf, v_lf, g_lf, b_lf, w_lf]

    qh = _l2norm(q_lf) if use_qk_l2norm else q_lf
    kh = _l2norm(k_lf) if use_qk_l2norm else k_lf
    qh = qh * s

    nt = t // chunk_len
    qc = _to_chunks(qh, chunk_len)
    kc = _to_chunks(kh, chunk_len)
    vc = _to_chunks(v_lf, chunk_len)
    bc = _to_chunks(b_lf, chunk_len)  # [B,H,NT,C,d_k]
    wc = _to_chunks(w_lf, chunk_len)  # [B,H,NT,C,d_v]
    g_chan = _to_chunks(g_lf, chunk_len)  # [B,H,NT,C,d_k]

    g2 = torch.cumsum(g_chan, dim=-2) * RCP_LN2  # [B,H,NT,C,d_k]
    g_last = g2[..., -1, :]  # [B,H,NT,d_k]
    gamma = torch.exp2(g2)  # [B,H,NT,C,d_k]

    if initial_state is not None:
        h0 = initial_state.detach().clone().to(q.dtype).requires_grad_(True)
        leaves.append(h0)
    else:
        h0 = torch.zeros(bsz, h, d_k, d_v, dtype=q.dtype, device=q.device).requires_grad_(True)

    eye = torch.eye(chunk_len, dtype=q.dtype, device=q.device)
    strict_lower = torch.tril(
        torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=q.device), -1
    )
    lower_incl = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=q.device), 0)

    h_list: list[Tensor] = [h0]
    o_list: list[Tensor] = []
    T_list: list[Tensor] = []
    u_list: list[Tensor] = []
    wy_list: list[Tensor] = []
    v_new_list: list[Tensor] = []
    A_qk_list: list[Tensor] = []

    for c in range(nt):
        kc_c, qc_c, vc_c = kc[:, :, c], qc[:, :, c], vc[:, :, c]
        bc_c, wc_c = bc[:, :, c], wc[:, :, c]
        gamma_c, g2_c = gamma[:, :, c], g2[:, :, c]
        glast_c = g_last[:, :, c]

        # decay_rel[i,s,j] = exp2(g2_i[j] - g2_s[j]); bounded (<=1) on the strict-lower
        # triangle that M/A actually read.
        decay_rel = torch.exp2(g2_c[..., :, None, :] - g2_c[..., None, :, :])  # [B,H,C,C,d_k]

        h_c = h_list[c]
        if skip_erase:
            t_mat = eye.expand(bsz, h, chunk_len, chunk_len)
            T_list.append(t_mat)
            u_c = wc_c * vc_c
            wy_c = torch.zeros(bsz, h, chunk_len, d_k, dtype=q.dtype, device=q.device)
            v_new_c = u_c
        else:
            bk = bc_c * kc_c  # erase-gated key  [B,H,C,d_k]
            m = torch.einsum("...id,...sd,...isd->...is", bk, kc_c, decay_rel)
            m = torch.where(strict_lower, m, torch.zeros_like(m))
            t_mat = torch.linalg.solve_triangular(eye + m, eye, upper=False, unitriangular=True)
            T_list.append(t_mat)

            u_c = t_mat @ (wc_c * vc_c)  # T (w (.) v)   [B,H,C,d_v]
            kbg = bk * gamma_c  # b (.) gamma (.) k       [B,H,C,d_k]
            wy_c = t_mat @ kbg  # T (b gamma k)           [B,H,C,d_k]
            v_new_c = u_c - wy_c @ h_c  # [B,H,C,d_v]

        qg = qc_c * gamma_c
        o_inter = qg @ h_c
        a_qk = torch.einsum("...id,...sd,...isd->...is", qc_c, kc_c, decay_rel)
        a_qk = torch.where(lower_incl, a_qk, torch.zeros_like(a_qk))
        o_intra = a_qk @ v_new_c
        o_list.append(o_inter + o_intra)

        decay_end = torch.exp2(glast_c[..., None, :] - g2_c)  # gamma_C / gamma_i  [B,H,C,d_k]
        k_dec = kc_c * decay_end
        h_next = torch.exp2(glast_c)[..., :, None] * h_c + k_dec.transpose(-1, -2) @ v_new_c
        h_list.append(h_next)

        u_list.append(u_c)
        wy_list.append(wy_c)
        v_new_list.append(v_new_c)
        A_qk_list.append(a_qk)

    o_chunks = torch.stack(o_list, dim=2)  # [B,H,NT,C,d_v]
    o = _from_chunks(o_chunks)

    return ChunkwiseForwardCW(
        o=o,
        q=qc,
        k=kc,
        v=vc,
        b=bc,
        w_gate=wc,
        g2=g2,
        g_last=g_last,
        gamma=gamma,
        T=torch.stack(T_list, dim=2),
        u=torch.stack(u_list, dim=2),
        wy=torch.stack(wy_list, dim=2),
        v_new=torch.stack(v_new_list, dim=2),
        h=torch.stack(h_list, dim=2),
        A_qk=torch.stack(A_qk_list, dim=2),
        h_list=h_list,
        v_new_list=v_new_list,
        wy_list=wy_list,
        u_list=u_list,
        leaves=tuple(leaves),
        chunk_len=chunk_len,
        scale=s,
        skip_erase=skip_erase,
    )


# ---------------------------------------------------------------------------
# Backward intermediates (autograd through the explicit forward)
# ---------------------------------------------------------------------------


@dataclass
class ChunkwiseBackwardCW:
    """Channel-wise final grads + per-kernel intermediates, all autograd-derived.

    Final grads (w.r.t. the pre-normed leaves; ``dh0`` is None without an initial state):
      dq, dk [B,T,H,d_k]; dv [B,T,H,d_v]; dg, db [B,T,H,d_k]; dw [B,T,H,d_v]; dh0.

    K#1 (B4) — checks (chunked, head-major):
      dh   [B,H,NT,d_k,d_v]   dh[c] = dL/dh_{c+1} (chunk-exit state grad)
      dh0  [B,H,d_k,d_v]      = dL/dh_0
      dv2  [B,H,NT,C,d_v]     = total dL/dv_new (= du fed to K#2)
    K#1 — input to feed the kernel:
      dv_local [B,H,NT,C,d_v] intra-only grad of v_new = tril(A_qk,0)^T @ do

    K#2 (B6) — checks:
      dk2 [B,H,NT,C,d_k], dv_final [B,H,NT,C,d_v], db_wy [B,H,NT,C,d_k],
      dw_wy [B,H,NT,C,d_v], dg2 [B,H,NT,C,d_k]
    K#2 — input to feed the kernel:
      dwy [B,H,NT,C,d_k] = dL/dwy (from B5)
    """

    dq: Tensor
    dk: Tensor
    dv: Tensor
    dg: Tensor
    db: Tensor
    dw: Tensor
    dh0: Tensor | None
    dh: Tensor
    dh0_state: Tensor
    dv2: Tensor
    dv_local: Tensor
    dwy: Tensor
    dk2: Tensor
    dv_final: Tensor
    db_wy: Tensor
    dw_wy: Tensor
    dg2: Tensor


def chunkwise_backward_cw(fwd: ChunkwiseForwardCW, do: Tensor) -> ChunkwiseBackwardCW:
    """Channel-wise backward intermediates from a :class:`ChunkwiseForwardCW`, via autograd.

    ``do`` matches ``o`` [B, T, H, d_v]. All quantities are autograd grads through the
    explicit channel-wise forward (final grads w.r.t. the leaves; intermediates w.r.t.
    the retained chunk tensors), plus the B1 sub-map VJP for the WY-VJP partial grads,
    plus the closed-form B3 ``dv_local``.
    """
    if fwd.skip_erase:
        raise ValueError("chunkwise_backward_cw needs the full graph; rebuild without skip_erase")
    o = fwd.o
    leaves = fwd.leaves

    grads = torch.autograd.grad(o, leaves, do, retain_graph=True, allow_unused=True)
    dq, dk, dv, dg, db, dw = grads[0], grads[1], grads[2], grads[3], grads[4], grads[5]
    dh0_leaf = grads[6] if len(leaves) > 6 else None

    dh_states = torch.autograd.grad(o, fwd.h_list[1:], do, retain_graph=True, allow_unused=True)
    dh_filled = [d if d is not None else torch.zeros_like(fwd.h_list[0]) for d in dh_states]
    dh = torch.stack(dh_filled, dim=2)  # [B,H,NT,d_k,d_v]
    dh0_state = torch.autograd.grad(o, fwd.h_list[0], do, retain_graph=True)[0]
    dv2_states = torch.autograd.grad(o, fwd.v_new_list, do, retain_graph=True)
    dv2 = torch.stack(dv2_states, dim=2)  # [B,H,NT,C,d_v]

    do_c = _to_chunks(do, fwd.chunk_len)  # [B,H,NT,C,d_v]
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c  # [B,H,NT,C,d_v]

    dwy_states = torch.autograd.grad(o, fwd.wy_list, do, retain_graph=True)
    dwy = torch.stack(dwy_states, dim=2)  # [B,H,NT,C,d_k]

    dk2, dv_final, db_wy, dw_wy, dg2 = _b1_submap_vjp_cw(fwd, dv2, dwy)

    return ChunkwiseBackwardCW(
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        db=db,
        dw=dw,
        dh0=dh0_leaf,
        dh=dh,
        dh0_state=dh0_state,
        dv2=dv2,
        dv_local=dv_local,
        dwy=dwy,
        dk2=dk2,
        dv_final=dv_final,
        db_wy=db_wy,
        dw_wy=dw_wy,
        dg2=dg2,
    )


def _b1_submap_vjp_cw(
    fwd: ChunkwiseForwardCW, du: Tensor, dwy: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """VJP of the channel-wise B1 map (k, v, b, w, g) -> (u, wy) per chunk.

    ``u = T(w (.) v)``, ``wy = T(b (.) gamma (.) k)``, ``T = (I + M)^{-1}`` with M the
    channel-decayed erase-gated key-key matrix. Differentiating through T's construction
    reproduces the inverse adjoint the kernel does via two triangular GEMMs. Returns
    (dk2, dv_final, db_wy, dw_wy, dg2) — the channel-wise WY-VJP partial grads.
    """
    b, h, nt, c, _ = fwd.k.shape
    dtype, dev = fwd.k.dtype, fwd.k.device
    eye = torch.eye(c, dtype=dtype, device=dev)
    strict_lower = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    dk2 = torch.zeros_like(fwd.k)
    dv_final = torch.zeros_like(fwd.v)
    db_wy = torch.zeros_like(fwd.b)
    dw_wy = torch.zeros_like(fwd.w_gate)
    dg2 = torch.zeros_like(fwd.b)

    # raw per-token channel-wise g recovered from the within-chunk cumsum (key axis).
    g_tok_all = torch.diff(
        fwd.g2 / RCP_LN2,
        dim=-2,
        prepend=torch.zeros(b, h, nt, 1, fwd.b.shape[-1], dtype=dtype, device=dev),
    )

    for ci in range(nt):
        k_lf = fwd.k[:, :, ci].detach().clone().requires_grad_(True)
        v_lf = fwd.v[:, :, ci].detach().clone().requires_grad_(True)
        b_lf = fwd.b[:, :, ci].detach().clone().requires_grad_(True)
        w_lf = fwd.w_gate[:, :, ci].detach().clone().requires_grad_(True)
        g_lf = g_tok_all[:, :, ci].detach().clone().requires_grad_(True)

        g2_c = torch.cumsum(g_lf, dim=-2) * RCP_LN2
        gamma_c = torch.exp2(g2_c)
        decay_rel = torch.exp2(g2_c[..., :, None, :] - g2_c[..., None, :, :])
        bk = b_lf * k_lf
        m = torch.einsum("...id,...sd,...isd->...is", bk, k_lf, decay_rel)
        m = torch.where(strict_lower, m, torch.zeros_like(m))
        t_mat = torch.linalg.solve_triangular(eye + m, eye, upper=False, unitriangular=True)
        u_c = t_mat @ (w_lf * v_lf)
        wy_c = t_mat @ (bk * gamma_c)

        gk, gv, gb, gw, gg = torch.autograd.grad(
            (u_c, wy_c), (k_lf, v_lf, b_lf, w_lf, g_lf), (du[:, :, ci], dwy[:, :, ci])
        )
        dk2[:, :, ci] = gk
        dv_final[:, :, ci] = gv
        db_wy[:, :, ci] = gb
        dw_wy[:, :, ci] = gw
        dg2[:, :, ci] = gg

    return dk2, dv_final, db_wy, dw_wy, dg2


# ---------------------------------------------------------------------------
# Micro-gate bundles (kernel inputs + expected outputs, head-major + chunked)
# ---------------------------------------------------------------------------


@dataclass
class MicroGateBundleCW:
    """One channel-wise kernel's micro-gate payload: ``inputs`` to feed, ``expected`` to check."""

    name: str
    inputs: dict[str, Tensor]
    expected: dict[str, Tensor]
    meta: dict[str, float | int]


def build_microgate_bundles_cw(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    chunk_len: int = 64,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> dict[str, MicroGateBundleCW]:
    """Build the channel-wise K#1 (B4) and K#2 (B6) micro-gate bundles from one pass.

    Inputs match :func:`chunkwise_forward_cw`. Returns ``{"k1": ..., "k2": ...}``. K#1 is
    fed ``dht = 0`` (no final-state grad); its expected ``dh`` is under that same assumption.
    """
    fwd = chunkwise_forward_cw(
        q,
        k,
        v,
        g,
        b,
        w,
        chunk_len=chunk_len,
        scale=scale,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
    )
    bwd = chunkwise_backward_cw(fwd, do)
    do_c = _to_chunks(do, chunk_len)
    h0 = fwd.h_list[0].detach()
    dht = torch.zeros_like(h0)

    k1 = MicroGateBundleCW(
        name="k1_reverse_state_scan_cw",
        inputs={
            "q": fwd.q.detach(),
            "k": fwd.k.detach(),
            "wy": fwd.wy.detach(),
            "g2": fwd.g2.detach(),
            "g_last": fwd.g_last.detach(),
            "do": do_c.detach(),
            "dv_local": bwd.dv_local.detach(),
            "h0": h0,
            "dht": dht,
        },
        expected={
            "dh": bwd.dh.detach(),
            "dh0": bwd.dh0_state.detach(),
            "dv2": bwd.dv2.detach(),
        },
        meta={"scale": fwd.scale, "chunk_len": fwd.chunk_len},
    )

    k2 = MicroGateBundleCW(
        name="k2_wy_triangular_vjp_cw",
        inputs={
            "k": fwd.k.detach(),
            "v": fwd.v.detach(),
            "b": fwd.b.detach(),
            "w": fwd.w_gate.detach(),
            "g2": fwd.g2.detach(),
            "T": fwd.T.detach(),
            "dwy": bwd.dwy.detach(),
            "du": bwd.dv2.detach(),
        },
        expected={
            "dk2": bwd.dk2.detach(),
            "dv": bwd.dv_final.detach(),
            "db": bwd.db_wy.detach(),
            "dw": bwd.dw_wy.detach(),
            "dg2": bwd.dg2.detach(),
        },
        meta={"scale": fwd.scale, "chunk_len": fwd.chunk_len},
    )
    return {"k1": k1, "k2": k2}
