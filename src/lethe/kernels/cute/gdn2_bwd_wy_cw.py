"""K#2 channel-wise (Phase 3), native Blackwell WY / triangular-inverse VJP (GDN-2 B6)."""
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
    """Validated pure-torch channel-wise WY-VJP (the spec the kernel transcribes)."""
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
    """Channel-wise K#2, host-orchestrated WY-VJP: un-decayed GEMMs on tcgen05; the fallback."""
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
    """Lever B, channel-wise WY-VJP with all n_idx chunks batched into one GEMM per step."""
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


_M_PAD = 128  # tiler M: every A operand / accumulator M-pads to 128 rows
_N_TILE = 64  # tiler N: one accumulator tile; d_k=128 outputs span 2 tiles
_K_PAD = 128  # tiler K


def k2f_dims_ok(c: int, d_k: int, d_v: int) -> bool:
    """True iff the fused kernel's baked tile fits: C=64, d_k=128, d_v in {64,128}."""
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
    """Host pre-glue for the fused K#2 kernel: staged GEMM operands, flat over Z=B·H·NT."""
    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]
    if c > _N_TILE or d_v > 2 * _N_TILE or d_k > _K_PAD:
        raise ValueError(
            f"_k2f_pack stages on the ({_M_PAD},{_N_TILE},{_K_PAD}) tiler (d_v up to "
            f"2·N_TILE via the dp N-tiling); got c={c}, d_k={d_k}, d_v={d_v}"
        )
    # b_du carries d_v in N: one N=64 tile at d_v=64, two (the dp N-tiling) at d_v=128.
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
    """Pure-torch model of the fused K#2 kernel's in-kernel dataflow (the kernel spec)."""
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
    """Default cw K#2: fused single-launch kernel > lever-B batched, by tile fit."""
    import os

    c, d_k = k.shape[3], k.shape[4]
    if not os.environ.get("FMR_DISABLE_K2F") and k2f_dims_ok(c, d_k, v.shape[-1]):
        from lethe.kernels.cute.gdn2_bwd_wy_f import run_k2_fused

        return run_k2_fused(k, v, b, w, g2, t_mat, dwy, du)
    return run_k2_batched(k, v, b, w, g2, t_mat, dwy, du)
