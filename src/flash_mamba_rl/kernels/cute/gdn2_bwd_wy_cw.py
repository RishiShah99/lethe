"""K#2 channel-wise (Phase 3) — native Blackwell WY / triangular-inverse VJP (GDN-2 B6).

Box bring-up file (-> ``src/.../cute/gdn2_bwd_wy_cw.py`` at integration). The Phase-3 crown
lift of ``scratch/gdn2_bwd_wy.py``: per-channel decay g (key axis), and *separate* erase gate
b (key axis) and write gate w (value axis), replacing the single scalar beta. Returns FIVE
grads ``(dk2, dv, db, dw, dg2)`` (was four — db and dw split).

The map differentiated, per chunk (M from the channel-decayed erase-gated key-key product,
``T = (I+M)^{-1}`` GIVEN from fwd):

    u = T (w (.) v) ,  wy = T (b (.) gamma (.) k)

VJP, given du(=dv2 from B4) and dwy(from B5). The inverse adjoint is the SAME two-triangular-
GEMM form as scalar (``T`` reused, never re-inverted); the channel-wise decay only enters
when pushing ``dM`` back onto k/b/g:

    dp = T^T@du ; dq_wy = T^T@dwy ; dT = du P^T + dwy Q^T ; dM = strict_lower(-(T^T dT T^T))
    dv = dp (.) w ; dw = dp (.) v
    dbk = dq_wy (.) gamma + sum_s dM (.) k (.) decay_rel        # decay_rel[i,s,d]=exp2(g2_i-g2_s)
    db = dbk (.) k ; dk = dbk (.) b + sum_i dM (.) (b k) (.) decay_rel
    dg2 = ln2(rowsum E - colsum E) + ln2 dq_wy (.) Q ;  E = dM (.) (b k) (.) k (.) decay_rel
    dg = RCP_LN2 * reverse_cumsum(dg2)                          # dg exits reverse-cumsum

Ground truth = ``kernels.cute.gdn2_assemble.k2_wy_vjp_cw_ref`` (validated to fp64 by
``tests/test_gdn2_assemble_cw.py``). The decay-weighted pushes use ``decay_rel`` (<= 1 on the
strict-lower triangle dM reads) so they stay bounded in fp32 — no secondary normalization.

Step 1 (THIS): host-orchestrated — the six un-decayed GEMMs/chunk route through the proven
(128,64,128) config via :func:`gdn2_bwd_wy._mm_tc`; the decay-weighted pushes and the
reverse-cumsum stay fp32 torch (the verified split). The WY-VJP is per-chunk independent (no
reverse carry). BOX-UNTESTED (math de-risked). fp16 GEMM operands, fp32 accumulate.
"""
# NB: no `from __future__ import annotations` — keep consistent with the DSL kernel files.

import math

import torch
from torch import Tensor

from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import is_available as _k1_available
from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import maybe_sync

LN2 = math.log(2.0)
RCP_LN2 = 1.0 / LN2


def is_available() -> bool:
    return _k1_available()


def run_k2_cw_ref(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Validated pure-torch channel-wise WY-VJP (the spec the kernel transcribes).

    Returns ``(dk2, dv, db, dw, dg2)`` head-major chunked. Identical to
    ``gdn2_assemble.k2_wy_vjp_cw_ref`` — kept here so the box file is self-contained.
    """
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    decay_rel = torch.exp2(g2f[:, :, None, :] - g2f[:, None, :, :])
    bk = bf * kf
    pwv = wf * vf
    q_kbg = bk * gamma

    tt = tf.transpose(-1, -2)
    dp = tt @ duf
    dq_wy = tt @ dwyf
    d_t = duf @ pwv.transpose(-1, -2) + dwyf @ q_kbg.transpose(-1, -2)
    d_m = torch.where(sl, -(tt @ d_t @ tt), torch.zeros_like(d_t))

    dv = dp * wf
    dw = dp * vf
    dbk_q = dq_wy * gamma
    dg2_q = dq_wy * q_kbg * LN2
    dbk_m = torch.einsum("nis,nsd,nisd->nid", d_m, kf, decay_rel)
    dk_m = torch.einsum("nis,nid,nisd->nsd", d_m, bk, decay_rel)
    e = torch.einsum("nis,nid,nsd,nisd->nisd", d_m, bk, kf, decay_rel)
    dg2_m = LN2 * (e.sum(dim=2) - e.sum(dim=1))

    dbk = dbk_q + dbk_m
    db = dbk * kf
    dk = dbk * bf + dk_m
    dg2 = dg2_q + dg2_m
    dg = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [1]), 1), [1])

    def _unflat(x: Tensor, last: int) -> Tensor:
        return x.reshape(bsz, hh, nt, c, last)

    return (
        _unflat(dk, d_k),
        _unflat(dv, d_v),
        _unflat(db, d_k),
        _unflat(dw, d_v),
        _unflat(dg, d_k),
    )


def run_k2_serial(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Channel-wise K#2 step 1 — host-orchestrated WY-VJP; un-decayed GEMMs on tcgen05.

    The proven per-chunk fallback (silicon-verified). Mirrors ``gdn2_bwd_wy.run_k2_serial``:
    the six un-decayed matmuls (dp, dq_wy, dT's two GEMMs, dM's two GEMMs) route through the
    proven (128,64,128) GEMM via :func:`gdn2_bwd_wy._mm_tc`; the decay-weighted pushes
    (dbk_m, dk_m, E -> dg2_m via ``decay_rel`` <= 1) and the reverse-cumsum stay fp32 torch.
    Per-chunk independent. Returns ``(dk2, dv, db, dw, dg2)`` head-major chunked. Lever B's
    :func:`run_k2_batched` is the default; this stays for fallback/debug.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from flash_mamba_rl.kernels.cute.gdn2_bwd_wy import _mm_tc

    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]
    dev = k.device

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)
    n_idx = bsz * hh * nt
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    dk2 = torch.zeros(n_idx, c, d_k, dtype=torch.float32, device=dev)
    dvf = torch.zeros(n_idx, c, d_v, dtype=torch.float32, device=dev)
    dbf = torch.zeros(n_idx, c, d_k, dtype=torch.float32, device=dev)
    dwf = torch.zeros(n_idx, c, d_v, dtype=torch.float32, device=dev)
    dgf = torch.zeros(n_idx, c, d_k, dtype=torch.float32, device=dev)

    for i in range(n_idx):
        ki, vi, bi, wi, ti = kf[i], vf[i], bf[i], wf[i], tf[i]
        dwyi, dui = dwyf[i], duf[i]
        gamma = torch.exp2(g2f[i])  # [C, d_k]
        decay_rel = torch.exp2(g2f[i][:, None, :] - g2f[i][None, :, :])  # [C, C, d_k]
        bk = bi * ki
        pwv = wi * vi
        q_kbg = bk * gamma

        tt = ti.transpose(0, 1)
        dp = _mm_tc(tt, dui)  # [C, d_v]
        dq_wy = _mm_tc(tt, dwyi)  # [C, d_k]
        d_t = _mm_tc(dui, pwv.transpose(0, 1)) + _mm_tc(dwyi, q_kbg.transpose(0, 1))  # [C, C]
        d_m = torch.where(sl, -_mm_tc(tt, _mm_tc(d_t, tt)), torch.zeros_like(d_t))

        dvf[i] = dp * wi
        dwf[i] = dp * vi
        dbk_q = dq_wy * gamma
        dg2_q = dq_wy * q_kbg * LN2
        dbk_m = torch.einsum("is,sd,isd->id", d_m, ki, decay_rel)
        dk_m = torch.einsum("is,id,isd->sd", d_m, bk, decay_rel)
        e = torch.einsum("is,id,sd,isd->isd", d_m, bk, ki, decay_rel)
        dg2_m = LN2 * (e.sum(dim=1) - e.sum(dim=0))  # [C, d_k]

        dbk = dbk_q + dbk_m
        dbf[i] = dbk * ki
        dk2[i] = dbk * bi + dk_m
        dg2 = dg2_q + dg2_m
        dgf[i] = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [0]), 0), [0])

    torch.cuda.synchronize()
    return (
        dk2.reshape(bsz, hh, nt, c, d_k),
        dvf.reshape(bsz, hh, nt, c, d_v),
        dbf.reshape(bsz, hh, nt, c, d_k),
        dwf.reshape(bsz, hh, nt, c, d_v),
        dgf.reshape(bsz, hh, nt, c, d_k),
    )


def run_k2_batched(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Lever B — channel-wise WY-VJP with all n_idx chunks batched into one GEMM per step.

    :func:`run_k2_cw_ref`'s structure verbatim with each un-decayed ``@`` replaced by ONE
    batched tcgen05 GEMM (:func:`gdn2_bwd_dhu._bmm_tc`) over all chunks; the decay-weighted
    pushes (``decay_rel`` einsums) and the reverse-cumsum are already batched in the ref and
    stay fp32 torch. Per-chunk independent (no carry). Returns ``(dk2, dv, db, dw, dg2)``.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import _bmm_tc

    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    decay_rel = torch.exp2(g2f[:, :, None, :] - g2f[:, None, :, :])
    bk = bf * kf
    pwv = wf * vf
    q_kbg = bk * gamma

    tt = tf.transpose(-1, -2)
    dp = _bmm_tc(tt, duf)
    dq_wy = _bmm_tc(tt, dwyf)
    d_t = _bmm_tc(duf, pwv.transpose(-1, -2)) + _bmm_tc(dwyf, q_kbg.transpose(-1, -2))
    d_m = torch.where(sl, -_bmm_tc(tt, _bmm_tc(d_t, tt)), torch.zeros_like(d_t))

    dv = dp * wf
    dw = dp * vf
    dbk_q = dq_wy * gamma
    dg2_q = dq_wy * q_kbg * LN2
    dbk_m = torch.einsum("nis,nsd,nisd->nid", d_m, kf, decay_rel)
    dk_m = torch.einsum("nis,nid,nisd->nsd", d_m, bk, decay_rel)
    e = torch.einsum("nis,nid,nsd,nisd->nisd", d_m, bk, kf, decay_rel)
    dg2_m = LN2 * (e.sum(dim=2) - e.sum(dim=1))

    dbk = dbk_q + dbk_m
    db = dbk * kf
    dk = dbk * bf + dk_m
    dg2 = dg2_q + dg2_m
    dg = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [1]), 1), [1])

    maybe_sync()
    return (
        dk.reshape(bsz, hh, nt, c, d_k),
        dv.reshape(bsz, hh, nt, c, d_v),
        db.reshape(bsz, hh, nt, c, d_k),
        dw.reshape(bsz, hh, nt, c, d_v),
        dg.reshape(bsz, hh, nt, c, d_k),
    )


# Lever B batched path is the default; run_k2_serial stays as the proven per-chunk fallback.
run_k2 = run_k2_batched
