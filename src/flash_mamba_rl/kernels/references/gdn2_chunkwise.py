"""Chunkwise scalar gated-delta (Phase-2 GDN) backward reference — the micro-gate ground truth.

The two native CuTe kernels (B4 reverse inter-chunk state scan, B6 WY/triangular-inverse
VJP) are validated in isolation against the intermediates this module exposes. The forward
here is the *canonical chunkwise decomposition* (UT/WY transform), not the token-serial
oracle: per chunk it builds ``T = (I + M)^{-1}``, ``u = T(beta·v)``, ``w = T(beta·gamma·k)``,
``v_new = u - w·h``, runs the inter-chunk state scan, and reads ``o``. Every backward
intermediate is then taken via ``torch.autograd`` *through this explicit forward* — correct
by construction, and exactly the VJP the hand-written kernels implement (autograd-VJP ≡
hand-VJP of the same forward). The B1 sub-map is differentiated on its own so the WY-VJP's
*partial* grads (dk2, dg2 — accumulated with B5's in B7) come out cleanly.

Scope = Phase 2: scalar decay ``g`` (one log-decay per token) and gates ``b = w = beta``.
This reduces the GDN-2 oracle (``references/gdn_backward.py``) to machine precision when its ``g`` is
fed channel-constant and ``b/w`` are ``beta`` broadcast — the regime the Hopper kill-gate
already verified to machine precision. Phase 3 lifts ``beta`` to per-channel ``b∈[0,1]^{d_k}``
/ ``w∈[0,1]^{d_v}`` and channel-wise decay.

Conventions (load-bearing; ``docs/gdn2_phase2_refs.md`` §fla):
* decays carried in **log2**: ``g2 = cumsum(g)·RCP_LN2`` within a chunk (reset per chunk),
  ``gamma = exp2(g2)``; cross-chunk via ``g_last``. ``exp2(x·RCP_LN2) == exp(x)`` so the
  forward is identical to the oracle's natural-exp scan while the exposed units match the kernel.
* ``T`` solved once in the forward and reused (its adjoint = autograd through the inverse).
* fp32/fp64 only (the reference; the kernel is bf16 I/O + fp32 accum).
* ``q`` is the pre-scaled, pre-L2-normed query the kernel sees; the L2-norm VJP (B8) is
  external to the two kernels and out of scope here.

Micro-gate bundles
------------------
``k1_microgate_bundle`` → inputs {q, k, w, g2, g_last, do, dv_local, h0, dht} and expected
outputs {dh, dh0, dv2} for K#1 (B4). ``k2_microgate_bundle`` → inputs {k, v, beta, g, T, dw,
du(=dv2)} and expected outputs {dk2, dv, db, dg2} for K#2 (B6).
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
# Forward (canonical chunkwise decomposition) with all intermediates retained
# ---------------------------------------------------------------------------


@dataclass
class ChunkwiseForward:
    """Forward output ``o`` plus every intermediate, head-major + chunked.

    Tensors carry grad history (non-leaf) so ``autograd.grad`` can pull backward
    intermediates from them. Chunked shapes use NT chunks of length C.

    o        : [B, T, H, d_v]
    q, k     : [B, H, NT, C, d_k]   (q pre-scaled+normed; k normed)
    v        : [B, H, NT, C, d_v]
    beta     : [B, H, NT, C]
    g2       : [B, H, NT, C]         within-chunk log2 cumsum
    g_last   : [B, H, NT]            g2[..., -1]
    gamma    : [B, H, NT, C]         exp2(g2)
    T        : [B, H, NT, C, C]      (I + M)^{-1}, lower-tri
    u, w     : [B, H, NT, C, d_v] / [B, H, NT, C, d_k]
    v_new    : [B, H, NT, C, d_v]
    h        : [B, H, NT + 1, d_k, d_v]   entry states; h[...,0]=h0, h[...,NT]=final
    A_qk     : [B, H, NT, C, C]      lower-incl-diag output score
    h_list / v_new_list / w_list / u_list : per-chunk non-leaf tensors for autograd.grad
    leaves   : (q, k, v, g_tok, beta, h0) leaves the forward was built from
    """

    o: Tensor
    q: Tensor
    k: Tensor
    v: Tensor
    beta: Tensor
    g2: Tensor
    g_last: Tensor
    gamma: Tensor
    T: Tensor
    u: Tensor
    w: Tensor
    v_new: Tensor
    h: Tensor
    A_qk: Tensor
    h_list: list[Tensor]
    v_new_list: list[Tensor]
    w_list: list[Tensor]
    u_list: list[Tensor]
    leaves: tuple[Tensor, ...]
    chunk_len: int
    scale: float


def chunkwise_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    chunk_len: int = 64,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> ChunkwiseForward:
    """Chunkwise scalar gated-delta forward, retaining all intermediates.

    Shapes: q, k [B, T, H, d_k]; v [B, T, H, d_v]; g, beta [B, T, H] (scalar per-token
    log-decay and write strength). Returns a
    :class:`ChunkwiseForward`. The graph is rooted at fresh leaves cloned from the
    inputs so ``autograd.grad`` can be called against both leaves and intermediates.
    """
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32/float64, got {q.dtype}")
    b, t, h, d_k = q.shape
    d_v = v.shape[-1]
    s = d_k**-0.5 if scale is None else scale

    q_lf = q.detach().clone().requires_grad_(True)
    k_lf = k.detach().clone().requires_grad_(True)
    v_lf = v.detach().clone().requires_grad_(True)
    g_lf = g.detach().clone().requires_grad_(True)
    beta_lf = beta.detach().clone().requires_grad_(True)
    leaves = [q_lf, k_lf, v_lf, g_lf, beta_lf]

    qh = _l2norm(q_lf) if use_qk_l2norm else q_lf
    kh = _l2norm(k_lf) if use_qk_l2norm else k_lf
    qh = qh * s

    nt = t // chunk_len
    qc = _to_chunks(qh, chunk_len)
    kc = _to_chunks(kh, chunk_len)
    vc = _to_chunks(v_lf, chunk_len)
    g_tok = g_lf.transpose(1, 2).reshape(b, h, nt, chunk_len)  # [B, H, NT, C]
    betac = beta_lf.transpose(1, 2).reshape(b, h, nt, chunk_len)  # [B,H,NT,C]

    g2 = torch.cumsum(g_tok, dim=-1) * RCP_LN2  # [B,H,NT,C]
    g_last = g2[..., -1]  # [B,H,NT]
    gamma = torch.exp2(g2)  # [B,H,NT,C]

    # h0 is always a grad leaf so K#1's dh0 is exposed even with a zero initial
    # state; it counts as a *final input grad* only when the caller supplied one.
    if initial_state is not None:
        h0 = initial_state.detach().clone().to(q.dtype).requires_grad_(True)
        leaves.append(h0)
    else:
        h0 = torch.zeros(b, h, d_k, d_v, dtype=q.dtype, device=q.device).requires_grad_(True)

    eye = torch.eye(chunk_len, dtype=q.dtype, device=q.device)
    strict_lower = torch.tril(
        torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=q.device), -1
    )
    lower_incl = torch.tril(torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=q.device), 0)

    h_list: list[Tensor] = [h0]
    o_list: list[Tensor] = []
    T_list: list[Tensor] = []
    u_list: list[Tensor] = []
    w_list: list[Tensor] = []
    v_new_list: list[Tensor] = []
    A_qk_list: list[Tensor] = []

    for c in range(nt):
        kc_c, qc_c, vc_c = kc[:, :, c], qc[:, :, c], vc[:, :, c]
        beta_c, gamma_c, g2_c = betac[:, :, c], gamma[:, :, c], g2[:, :, c]
        glast_c = g_last[:, :, c]

        ratio = gamma_c[..., :, None] / gamma_c[..., None, :]  # gamma_i / gamma_s
        kk = kc_c @ kc_c.transpose(-1, -2)
        m = beta_c[..., :, None] * ratio * kk
        m = torch.where(strict_lower, m, torch.zeros_like(m))
        t_mat = torch.linalg.solve_triangular(eye + m, eye, upper=False, unitriangular=True)
        T_list.append(t_mat)

        u_c = t_mat @ (beta_c[..., None] * vc_c)  # [B,H,C,d_v]
        kbg = (beta_c * gamma_c)[..., None] * kc_c  # beta·gamma·k
        w_c = t_mat @ kbg  # [B,H,C,d_k]
        h_c = h_list[c]
        v_new_c = u_c - w_c @ h_c  # [B,H,C,d_v]

        qg = qc_c * gamma_c[..., None]
        o_inter = qg @ h_c
        a_qk = (qc_c @ kc_c.transpose(-1, -2)) * ratio
        a_qk = torch.where(lower_incl, a_qk, torch.zeros_like(a_qk))
        o_intra = a_qk @ v_new_c
        o_list.append(o_inter + o_intra)

        decay_end = torch.exp2(glast_c[..., None] - g2_c)  # gamma_C / gamma_s
        k_dec = kc_c * decay_end[..., None]
        h_next = torch.exp2(glast_c)[..., None, None] * h_c + k_dec.transpose(-1, -2) @ v_new_c
        h_list.append(h_next)

        u_list.append(u_c)
        w_list.append(w_c)
        v_new_list.append(v_new_c)
        A_qk_list.append(a_qk)

    o_chunks = torch.stack(o_list, dim=2)  # [B,H,NT,C,d_v]
    o = _from_chunks(o_chunks)

    return ChunkwiseForward(
        o=o,
        q=qc,
        k=kc,
        v=vc,
        beta=betac,
        g2=g2,
        g_last=g_last,
        gamma=gamma,
        T=torch.stack(T_list, dim=2),
        u=torch.stack(u_list, dim=2),
        w=torch.stack(w_list, dim=2),
        v_new=torch.stack(v_new_list, dim=2),
        h=torch.stack(h_list, dim=2),
        A_qk=torch.stack(A_qk_list, dim=2),
        h_list=h_list,
        v_new_list=v_new_list,
        w_list=w_list,
        u_list=u_list,
        leaves=tuple(leaves),
        chunk_len=chunk_len,
        scale=s,
    )


# ---------------------------------------------------------------------------
# Backward intermediates (autograd through the explicit forward)
# ---------------------------------------------------------------------------


@dataclass
class ChunkwiseBackward:
    """Final grads + the per-kernel intermediates, all autograd-derived.

    Final grads (w.r.t. the pre-normed leaves; ``dh0`` is None without an initial state):
      dq, dk, dv [input shapes], dg [B,T,H,d_k] channel-broadcast, db [B,T,H], dh0.

    K#1 (B4) — checks (chunked, head-major):
      dh   [B,H,NT,d_k,d_v]  fla convention dh[c] = dL/dh_{c+1} (chunk-exit state grad)
      dh0  [B,H,d_k,d_v]     = dL/dh_0
      dv2  [B,H,NT,C,d_v]    = total dL/dv_new  (= du fed to K#2)
    K#1 — inputs to feed the kernel:
      dv_local [B,H,NT,C,d_v]  intra-only grad of v_new = tril(A_qk,0)^T @ do

    K#2 (B6) — checks:
      dk2 [B,H,NT,C,d_k], dv_final [B,H,NT,C,d_v], db [B,H,NT,C], dg2 [B,H,NT,C]
    K#2 — inputs to feed the kernel:
      dw  [B,H,NT,C,d_k]  = dL/dw  (from B5)
    """

    dq: Tensor
    dk: Tensor
    dv: Tensor
    dg: Tensor
    db: Tensor
    dh0: Tensor | None
    dh: Tensor
    dh0_state: Tensor
    dv2: Tensor
    dv_local: Tensor
    dw: Tensor
    dk2: Tensor
    dv_final: Tensor
    db_wy: Tensor
    dg2: Tensor


def chunkwise_backward(fwd: ChunkwiseForward, do: Tensor) -> ChunkwiseBackward:
    """Backward intermediates from a :class:`ChunkwiseForward`, via autograd.

    ``do`` matches ``o`` [B, T, H, d_v]. Returns a :class:`ChunkwiseBackward`. All
    quantities are autograd grads through the explicit chunkwise forward (final
    grads w.r.t. the leaves; intermediates w.r.t. the retained chunk tensors), plus
    the B1 sub-map VJP for the WY-VJP partial grads, plus the closed-form B3 dv_local.
    """
    o = fwd.o
    leaves = fwd.leaves

    # --- final grads (validate vs the oracle, reduced to scalar) ---
    grads = torch.autograd.grad(o, leaves, do, retain_graph=True, allow_unused=True)
    dq, dk, dv, dg_tok, db = grads[0], grads[1], grads[2], grads[3], grads[4]
    dh0_leaf = grads[5] if len(leaves) > 5 else None

    # --- K#1 intermediates: dh (fla convention), dh0, dv2 ---
    # dh[c] = dL/dh_{c+1}; the NT chunk-exit states are h_list[1..NT]. The final
    # state h_list[NT] is unused in o (no final-state grad fed) -> dh[NT-1] = 0,
    # exactly as the kernel inits its accumulator to dht = 0.
    dh_states = torch.autograd.grad(o, fwd.h_list[1:], do, retain_graph=True, allow_unused=True)
    dh_filled = [d if d is not None else torch.zeros_like(fwd.h_list[0]) for d in dh_states]
    dh = torch.stack(dh_filled, dim=2)  # [B,H,NT,d_k,d_v]
    dh0_state = torch.autograd.grad(o, fwd.h_list[0], do, retain_graph=True)[0]
    dv2_states = torch.autograd.grad(o, fwd.v_new_list, do, retain_graph=True)
    dv2 = torch.stack(dv2_states, dim=2)  # [B,H,NT,C,d_v]

    # --- K#1 input: dv_local (B3) = tril(A_qk, 0)^T @ do, per chunk (closed form) ---
    do_c = _to_chunks(do, fwd.chunk_len)  # [B,H,NT,C,d_v]
    dv_local = fwd.A_qk.transpose(-1, -2) @ do_c  # [B,H,NT,C,d_v]

    # --- K#2 input: dw (B5) = dL/dw ---
    dw_states = torch.autograd.grad(o, fwd.w_list, do, retain_graph=True)
    dw = torch.stack(dw_states, dim=2)  # [B,H,NT,C,d_k]

    # --- K#2 outputs: B1 sub-map VJP, per chunk, with grad_outputs (du=dv2, dw) ---
    dk2, dv_final, db_wy, dg2 = _b1_submap_vjp(fwd, dv2, dw)

    return ChunkwiseBackward(
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg_tok,
        db=db,
        dh0=dh0_leaf,
        dh=dh,
        dh0_state=dh0_state,
        dv2=dv2,
        dv_local=dv_local,
        dw=dw,
        dk2=dk2,
        dv_final=dv_final,
        db_wy=db_wy,
        dg2=dg2,
    )


def _b1_submap_vjp(
    fwd: ChunkwiseForward, du: Tensor, dw: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """VJP of the B1 map (k, v, beta, g_tok) -> (u, w) per chunk.

    ``u = T(beta·v)``, ``w = T(beta·gamma·k)``, ``T = (I + M)^{-1}`` with M from
    (k, beta, gamma); gamma = exp2(cumsum(g)·RCP_LN2). Differentiating through T's
    construction reproduces the inverse adjoint the kernel does via two triangular
    GEMMs. Returns (dk2, dv_final, db_wy, dg2) — the WY-VJP partial grads.
    """
    b, h, nt, c, _ = fwd.k.shape
    dtype, dev = fwd.k.dtype, fwd.k.device
    eye = torch.eye(c, dtype=dtype, device=dev)
    strict_lower = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    dk2 = torch.zeros_like(fwd.k)
    dv_final = torch.zeros_like(fwd.v)
    db_wy = torch.zeros_like(fwd.beta)
    dg2 = torch.zeros_like(fwd.beta)

    # raw per-token g (scalar) recovered from the within-chunk cumsum.
    g_tok_all = torch.diff(
        fwd.g2 / RCP_LN2, dim=-1, prepend=torch.zeros(b, h, nt, 1, dtype=dtype, device=dev)
    )

    for ci in range(nt):
        k_lf = fwd.k[:, :, ci].detach().clone().requires_grad_(True)
        v_lf = fwd.v[:, :, ci].detach().clone().requires_grad_(True)
        beta_lf = fwd.beta[:, :, ci].detach().clone().requires_grad_(True)
        g_lf = g_tok_all[:, :, ci].detach().clone().requires_grad_(True)

        g2_c = torch.cumsum(g_lf, dim=-1) * RCP_LN2
        gamma_c = torch.exp2(g2_c)
        ratio = gamma_c[..., :, None] / gamma_c[..., None, :]
        kk = k_lf @ k_lf.transpose(-1, -2)
        m = beta_lf[..., :, None] * ratio * kk
        m = torch.where(strict_lower, m, torch.zeros_like(m))
        t_mat = torch.linalg.solve_triangular(eye + m, eye, upper=False, unitriangular=True)
        u_c = t_mat @ (beta_lf[..., None] * v_lf)
        w_c = t_mat @ ((beta_lf * gamma_c)[..., None] * k_lf)

        gk, gv, gb, gg = torch.autograd.grad(
            (u_c, w_c), (k_lf, v_lf, beta_lf, g_lf), (du[:, :, ci], dw[:, :, ci])
        )
        dk2[:, :, ci] = gk
        dv_final[:, :, ci] = gv
        db_wy[:, :, ci] = gb
        dg2[:, :, ci] = gg

    return dk2, dv_final, db_wy, dg2


# ---------------------------------------------------------------------------
# Micro-gate bundles (kernel inputs + expected outputs, head-major + chunked)
# ---------------------------------------------------------------------------


@dataclass
class MicroGateBundle:
    """One kernel's micro-gate payload: ``inputs`` to feed it, ``expected`` to check.

    All tensors are head-major, chunked ([B, H, NT, C, ...]), in the reference dtype
    (fp32/fp64). The on-box harness casts to bf16 I/O for the kernel and compares
    within the verifier's tolerances. ``meta`` carries scalar params (scale, chunk_len).
    """

    name: str
    inputs: dict[str, Tensor]
    expected: dict[str, Tensor]
    meta: dict[str, float | int]


def build_microgate_bundles(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    do: Tensor,
    *,
    chunk_len: int = 64,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    use_qk_l2norm: bool = False,
) -> dict[str, MicroGateBundle]:
    """Build the K#1 (B4) and K#2 (B6) micro-gate bundles from one fwd/bwd pass.

    Inputs match :func:`chunkwise_forward`. Returns ``{"k1": ..., "k2": ...}``. K#1
    is fed ``dht = 0`` (no final-state grad); its expected ``dh`` is computed under
    that same assumption. Re-run the reference here so the bundle is self-contained.
    """
    fwd = chunkwise_forward(
        q,
        k,
        v,
        g,
        beta,
        chunk_len=chunk_len,
        scale=scale,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
    )
    bwd = chunkwise_backward(fwd, do)
    do_c = _to_chunks(do, chunk_len)
    h0 = fwd.h_list[0].detach()
    dht = torch.zeros_like(h0)

    k1 = MicroGateBundle(
        name="k1_reverse_state_scan",
        inputs={
            "q": fwd.q.detach(),
            "k": fwd.k.detach(),
            "w": fwd.w.detach(),
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

    k2 = MicroGateBundle(
        name="k2_wy_triangular_vjp",
        inputs={
            "k": fwd.k.detach(),
            "v": fwd.v.detach(),
            "beta": fwd.beta.detach(),
            "g2": fwd.g2.detach(),
            "T": fwd.T.detach(),
            "dw": bwd.dw.detach(),
            "du": bwd.dv2.detach(),
        },
        expected={
            "dk2": bwd.dk2.detach(),
            "dv": bwd.dv_final.detach(),
            "db": bwd.db_wy.detach(),
            "dg2": bwd.dg2.detach(),
        },
        meta={"scale": fwd.scale, "chunk_len": fwd.chunk_len},
    )
    return {"k1": k1, "k2": k2}
