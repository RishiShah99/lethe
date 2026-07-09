"""K#2 channel-wise (Phase 3), native Blackwell WY / triangular-inverse VJP (GDN-2 B6).

The Phase-3 channel-wise lift of the scalar K#2: per-channel decay g (key axis), and
*separate* erase gate b (key axis) and write gate w (value axis), replacing the single
scalar beta. Returns FIVE grads ``(dk2, dv, db, dw, dg2)`` (was four; db and dw split).

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
strict-lower triangle dM reads) so they stay bounded in fp32; no secondary normalization.

``run_k2_serial`` routes the six un-decayed GEMMs/chunk through the (128,64,128) config
via :func:`gdn2_bwd_wy._mm_tc`; the decay-weighted pushes and the reverse-cumsum stay
fp32 torch. The WY-VJP is per-chunk independent (no reverse carry). fp16 GEMM operands,
fp32 accumulate.
"""
# NB: no `from __future__ import annotations`, keep consistent with the DSL kernel files.

import math

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_bwd_dhu import is_available as _k1_available
from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync
from lethe.kernels.references.gdn2_chunkwise_cw import masked_decay_rel

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
    ``gdn2_assemble.k2_wy_vjp_cw_ref``, kept here so this module is self-contained.
    """
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    decay_rel = masked_decay_rel(g2f)
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
    dg2_m = LN2 * (bk * dbk_m - kf * dk_m)  # rowsum(E)/colsum(E) reuse; E never built

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
    """Channel-wise K#2, host-orchestrated WY-VJP: un-decayed GEMMs on tcgen05.

    The per-chunk fallback. Mirrors ``gdn2_bwd_wy.run_k2_serial``: the six un-decayed
    matmuls (dp, dq_wy, dT's two GEMMs, dM's two GEMMs) route through the (128,64,128)
    GEMM via :func:`gdn2_bwd_wy._mm_tc`; the decay-weighted pushes (dbk_m, dk_m, E ->
    dg2_m via ``decay_rel`` <= 1) and the reverse-cumsum stay fp32 torch. Per-chunk
    independent. Returns ``(dk2, dv, db, dw, dg2)`` head-major chunked. The
    :func:`run_k2` selector (fused > batched) is the default; this stays for fallback/debug.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_wy import _mm_tc

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
        decay_rel = masked_decay_rel(g2f[i])  # [C, C, d_k]
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
        dg2_m = LN2 * (bk * dbk_m - ki * dk_m)  # [C, d_k]; E never built

        dbk = dbk_q + dbk_m
        dbf[i] = dbk * ki
        dk2[i] = dbk * bi + dk_m
        dg2 = dg2_q + dg2_m
        dgf[i] = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [0]), 0), [0])

    maybe_sync()
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
    """Lever B, channel-wise WY-VJP with all n_idx chunks batched into one GEMM per step.

    :func:`run_k2_cw_ref`'s structure verbatim with each un-decayed ``@`` replaced by ONE
    batched tcgen05 GEMM (:func:`gdn2_bwd_dhu._bmm_tc`) over all chunks; the decay-weighted
    pushes (``decay_rel`` einsums) and the reverse-cumsum are already batched in the ref and
    stay fp32 torch. Per-chunk independent (no carry). Returns ``(dk2, dv, db, dw, dg2)``.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_dhu import _bmm_tc

    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(bsz * hh * nt, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=kf.device), -1)

    gamma = torch.exp2(g2f)
    decay_rel = masked_decay_rel(g2f)
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
    dg2_m = LN2 * (bk * dbk_m - kf * dk_m)  # rowsum(E)/colsum(E) reuse; E never built

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


# ------------------------------------------------------------------
# Fused K#2: one grid-z-batched kernel per backward.
# Chunk-local (no carry), so the right grid is one CTA per chunk over Z=B·H·NT:
# the Level-2 batching layout, not L3's unroll. _run_k2_fused_modelled is the kernel
# spec (fp64-pinned vs k2_wy_vjp_cw_ref by tests/test_gdn2_k2_fused.py); _k2f_pack
# is the host pre-glue the launcher casts to fp16.
# ------------------------------------------------------------------

_M_PAD = 128  # tiler M: every A operand / accumulator M-pads to 128 rows
_N_TILE = 64  # tiler N: one accumulator tile; d_k=128 outputs span 2 tiles
_K_PAD = 128  # tiler K


def k2f_dims_ok(c: int, d_k: int, d_v: int) -> bool:
    """True iff the fused kernel's baked tile fits: C=64, d_k=128, d_v in {64,128}.

    d_v=128 fires the dp N-tiling path (a 2nd dp mainloop over b_du's 2nd N=64 tile);
    d_v=64 is the single-tile shape.
    """
    return c == 64 and d_k == 128 and d_v in (64, 128)


def _k2f_pack(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> dict[str, Tensor]:
    """Host pre-glue for the fused K#2 kernel: staged GEMM operands, flat over Z=B·H·NT.

    Six GEMM sites on the a@b^T (128,64,128) tiler: A operands [Z,128,128]
    (M/K zero-pad), B operands [Z,N,128] (K zero-pad); G2's d_k=128 output spans two
    N=64 tiles. Elementwise pre-glue (gamma, bk, pwv, q_kbg) is O(C·d) and stays host,
    matching the ``_incb2_pack`` layout. The SIMT epilogue operands ride along unpadded.
    Dtype-preserving; the launcher casts GEMM operands to fp16.
    """
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]
    if c > _N_TILE or d_v > 2 * _N_TILE or d_k > _K_PAD:
        raise ValueError(
            f"_k2f_pack stages on the ({_M_PAD},{_N_TILE},{_K_PAD}) tiler (d_v up to "
            f"2·N_TILE via the dp N-tiling); got c={c}, d_k={d_k}, d_v={d_v}"
        )
    # ``b_du`` carries d_v in its N-dim: one N=64 tile at d_v=64, two at d_v=128 (the
    # dp N-tiling increment, mirroring b_dwy's d_k=128 → 2·N_TILE staging).
    n_du = _N_TILE if d_v <= _N_TILE else 2 * _N_TILE
    z = bsz * hh * nt
    dev, dt = k.device, k.dtype

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(z, *x.shape[3:])

    kf, vf, bf, wf, g2f, tf = _flat(k), _flat(v), _flat(b), _flat(w), _flat(g2), _flat(t_mat)
    dwyf, duf = _flat(dwy), _flat(du)

    gamma = torch.exp2(g2f)
    bk = bf * kf
    pwv = wf * vf
    q_kbg = bk * gamma

    a_tt = torch.zeros(z, _M_PAD, _K_PAD, dtype=dt, device=dev)
    a_tt[:, :c, :c] = tf.transpose(-1, -2)
    a_du = torch.zeros(z, _M_PAD, _K_PAD, dtype=dt, device=dev)
    a_du[:, :c, :d_v] = duf
    a_dwy = torch.zeros(z, _M_PAD, _K_PAD, dtype=dt, device=dev)
    a_dwy[:, :c, :d_k] = dwyf

    b_du = torch.zeros(z, n_du, _K_PAD, dtype=dt, device=dev)
    b_du[:, :d_v, :c] = duf.transpose(-1, -2)
    b_dwy = torch.zeros(z, 2 * _N_TILE, _K_PAD, dtype=dt, device=dev)
    b_dwy[:, :d_k, :c] = dwyf.transpose(-1, -2)
    b_pwv = torch.zeros(z, _N_TILE, _K_PAD, dtype=dt, device=dev)
    b_pwv[:, :c, :d_v] = pwv
    b_qkbg = torch.zeros(z, _N_TILE, _K_PAD, dtype=dt, device=dev)
    b_qkbg[:, :c, :d_k] = q_kbg
    b_t = torch.zeros(z, _N_TILE, _K_PAD, dtype=dt, device=dev)
    b_t[:, :c, :c] = tf

    return {
        "a_tt": a_tt,
        "a_du": a_du,
        "a_dwy": a_dwy,
        "b_du": b_du,
        "b_dwy": b_dwy,
        "b_pwv": b_pwv,
        "b_qkbg": b_qkbg,
        "b_t": b_t,
        "k": kf,
        "v": vf,
        "b": bf,
        "w": wf,
        "g2": g2f,
        "gamma": gamma,
        "bk": bk,
        "q_kbg": q_kbg,
    }


def _run_k2_fused_modelled(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pure-torch model of the fused K#2 kernel's in-kernel dataflow (the kernel spec).

    Statement order mirrors the kernel: G1 dp / G2 dq_wy / G3+G4 dT (one shared
    accumulator), the two chained landings written into the zero-padded operand scratch
    (``s_dt`` A-shaped, ``s_x`` B-shaped/transposed, fp16) before G5/G6,
    strict-lower mask + negate in G6's epilogue, then the decay pushes and the
    reverse-cumsum. The kernel computes ``exp2(g2_i - g2_s)`` on the fly on the
    strict-lower triangle only; ``masked_decay_rel`` is bitwise the same on every entry
    ``d_m`` reads (its live diagonal multiplies ``d_m``'s zero diagonal). Device/dtype-
    agnostic; in fp64 it reproduces ``k2_wy_vjp_cw_ref`` to roundoff. Returns
    ``(dk2, dv, db, dw, dg2)`` head-major chunked.
    """
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]
    z = bsz * hh * nt
    dev, dt = k.device, k.dtype

    buf = _k2f_pack(k, v, b, w, g2, t_mat, dwy, du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    dp = (buf["a_tt"] @ buf["b_du"].transpose(-1, -2))[:, :c, :d_v]
    dq_wy = (buf["a_tt"] @ buf["b_dwy"].transpose(-1, -2))[:, :c, :d_k]
    d_t_acc = buf["a_du"] @ buf["b_pwv"].transpose(-1, -2) + buf["a_dwy"] @ buf["b_qkbg"].transpose(
        -1, -2
    )

    s_dt = torch.zeros(z, _M_PAD, _K_PAD, dtype=dt, device=dev)
    s_dt[:, :c, :c] = d_t_acc[:, :c, :c]
    x_acc = s_dt @ buf["b_t"].transpose(-1, -2)
    s_x = torch.zeros(z, _N_TILE, _K_PAD, dtype=dt, device=dev)
    s_x[:, :c, :c] = x_acc[:, :c, :c].transpose(-1, -2)
    dm_acc = buf["a_tt"] @ s_x.transpose(-1, -2)
    d_m = torch.where(sl, -dm_acc[:, :c, :c], torch.zeros_like(dm_acc[:, :c, :c]))

    kf, vf, bf, wf = buf["k"], buf["v"], buf["b"], buf["w"]
    gamma, bk, q_kbg = buf["gamma"], buf["bk"], buf["q_kbg"]
    decay_rel = masked_decay_rel(buf["g2"])

    dv = dp * wf
    dw = dp * vf
    dbk_q = dq_wy * gamma
    dg2_q = dq_wy * q_kbg * LN2
    dbk_m = torch.einsum("nis,nsd,nisd->nid", d_m, kf, decay_rel)
    dk_m = torch.einsum("nis,nid,nisd->nsd", d_m, bk, decay_rel)
    dg2_m = LN2 * (bk * dbk_m - kf * dk_m)  # rowsum(E)/colsum(E) reuse; E never built

    dbk = dbk_q + dbk_m
    db = dbk * kf
    dk = dbk * bf + dk_m
    dg2 = dg2_q + dg2_m
    dg = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [1]), 1), [1])

    return (
        dk.reshape(bsz, hh, nt, c, d_k),
        dv.reshape(bsz, hh, nt, c, d_v),
        db.reshape(bsz, hh, nt, c, d_k),
        dw.reshape(bsz, hh, nt, c, d_v),
        dg.reshape(bsz, hh, nt, c, d_k),
    )


def run_k2(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Default cw K#2: fused single-launch kernel > lever-B batched, by tile fit.

    The fused path (ONE grid-z launch for all chunks, no ``decay_rel``
    materialization; :mod:`gdn2_bwd_wy_f`) is dim-locked to C=64, d_k=128, d_v in
    {64,128} (d_v=128 fires the dp N-tiling path); other shapes fall to batched.
    Kill-switch: ``FMR_DISABLE_K2F=1`` forces the batched path. ``run_k2_serial``
    stays as the per-chunk fallback.
    """
    import os

    c, d_k = k.shape[3], k.shape[4]
    if not os.environ.get("FMR_DISABLE_K2F") and k2f_dims_ok(c, d_k, v.shape[-1]):
        from lethe.kernels.cute.gdn2_bwd_wy_f import run_k2_fused

        return run_k2_fused(k, v, b, w, g2, t_mat, dwy, du)
    return run_k2_batched(k, v, b, w, g2, t_mat, dwy, du)
