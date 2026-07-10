"""K#2, native Blackwell (sm_100) WY / triangular-inverse VJP (GDN backward, B6)."""
# NB: no `from __future__ import annotations`, keep consistent with the DSL kernel files.

import math

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_bwd_dhu import is_available as _k1_available
from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync
from lethe.kernels.references.gdn2_chunkwise import masked_decay_ratio

LN2 = math.log(2.0)
RCP_LN2 = 1.0 / LN2
CHUNK = 64
D_K = 128
D_V = 64


def is_available() -> bool:
    return _k1_available()


def run_k2_ref(
    k: Tensor, v: Tensor, beta: Tensor, g2: Tensor, t_mat: Tensor, dw: Tensor, du: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Validated pure-torch WY-VJP (the spec/oracle the kernel transcribes)."""
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
    d_a = -(tt @ d_t @ tt)
    d_m = torch.where(sl, d_a, torch.zeros_like(d_a))

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


def _mm_tc(x: Tensor, y: Tensor) -> Tensor:
    """``x[M,K] @ y[K,N]`` via inc-A's (128,64,128) ``a@b^T`` GEMM, fp32 out."""
    m, kk = x.shape
    if m > D_K or kk > D_K:
        raise ValueError(
            f"_mm_tc stages M,K into a ({D_K},{D_K}) buffer; got x[{m},{kk}], both must "
            f"be <= {D_K} (N tiles by D_V={D_V}; M,K below {D_K} zero-pad and stay correct)"
        )
    from lethe.kernels.cute.gdn2_bwd_dhu import _gemm_aa

    f16, dev = torch.float16, x.device
    _, n = y.shape
    a = torch.zeros(D_K, D_K, dtype=f16, device=dev)
    a[:m, :kk] = x.to(f16)
    yt = y.transpose(0, 1).contiguous()  # [N, K]
    out = torch.zeros(m, n, dtype=torch.float32, device=dev)
    for n0 in range(0, n, D_V):
        wn = min(D_V, n - n0)
        bmat = torch.zeros(D_V, D_K, dtype=f16, device=dev)
        bmat[:wn, :kk] = yt[n0 : n0 + wn].to(f16)
        o = torch.zeros(D_K, D_V, dtype=f16, device=dev)
        _gemm_aa(a, bmat, o)
        out[:, n0 : n0 + wn] = o[:m, :wn].float()
    return out


def run_k2_serial(
    k: Tensor, v: Tensor, beta: Tensor, g2: Tensor, t_mat: Tensor, dw: Tensor, du: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """K#2, host-orchestrated WY-VJP: 7 GEMMs per chunk on tcgen05; the per-chunk fallback."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, h, nt, c, _ = k.shape
    d_k, d_v = k.shape[-1], v.shape[-1]
    dev = k.device

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _flat(k), _flat(v), _flat(beta), _flat(g2), _flat(t_mat)
    dwf, duf = _flat(dw), _flat(du)
    n_idx = b * h * nt
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    dk2 = torch.zeros(n_idx, c, d_k, dtype=torch.float32, device=dev)
    dvf = torch.zeros(n_idx, c, d_v, dtype=torch.float32, device=dev)
    dbf = torch.zeros(n_idx, c, dtype=torch.float32, device=dev)
    dgf = torch.zeros(n_idx, c, dtype=torch.float32, device=dev)

    for i in range(n_idx):
        ki, vi, bi, ti = kf[i], vf[i], betaf[i], tf[i]
        dwi, dui = dwf[i], duf[i]
        gamma = torch.exp2(g2f[i])  # [C]
        ratio = masked_decay_ratio(g2f[i])
        bv = bi[:, None] * vi
        bgk = (bi * gamma)[:, None] * ki

        kk = _mm_tc(ki, ki.transpose(0, 1))  # k@k^T
        tt = ti.transpose(0, 1)
        dbv = _mm_tc(tt, dui)
        dbgk = _mm_tc(tt, dwi)
        d_t = _mm_tc(dui, bv.transpose(0, 1)) + _mm_tc(dwi, bgk.transpose(0, 1))
        d_a = -_mm_tc(tt, _mm_tc(d_t, tt))
        d_m = torch.where(sl, d_a, torch.zeros_like(d_a))

        d_kk = d_m * bi[:, None] * ratio
        dk2[i] = _mm_tc(d_kk + d_kk.transpose(0, 1), ki) + (bi * gamma)[:, None] * dbgk
        dvf[i] = bi[:, None] * dbv
        p = (dbgk * ki).sum(-1)
        r = (dbv * vi).sum(-1)
        dbf[i] = r + gamma * p + (d_m * (ratio * kk)).sum(-1)

        e = (d_m * bi[:, None] * kk) * ratio
        dg2_total = LN2 * (e.sum(-1) - e.sum(-2)) + (bi * p) * LN2 * gamma
        dgf[i] = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2_total, [-1]), -1), [-1])

    maybe_sync()
    return (
        dk2.reshape(b, h, nt, c, d_k),
        dvf.reshape(b, h, nt, c, d_v),
        dbf.reshape(b, h, nt, c),
        dgf.reshape(b, h, nt, c),
    )


def run_k2_batched(
    k: Tensor, v: Tensor, beta: Tensor, g2: Tensor, t_mat: Tensor, dw: Tensor, du: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Lever B, WY-VJP with all n_idx=b*h*nt chunks batched into one GEMM per matmul-step."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_dhu import _bmm_tc

    b, h, nt, c, _ = k.shape
    d_k, d_v = k.shape[-1], v.shape[-1]

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _flat(k), _flat(v), _flat(beta), _flat(g2), _flat(t_mat)
    dwf, duf = _flat(dw), _flat(du)
    sl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=k.device), -1)

    gamma = torch.exp2(g2f)
    bv = betaf[..., None] * vf
    bgk = (betaf * gamma)[..., None] * kf
    ratio = masked_decay_ratio(g2f)

    kk = _bmm_tc(kf, kf.transpose(-1, -2))
    tt = tf.transpose(-1, -2)
    dbv = _bmm_tc(tt, duf)
    dbgk = _bmm_tc(tt, dwf)
    d_t = _bmm_tc(duf, bv.transpose(-1, -2)) + _bmm_tc(dwf, bgk.transpose(-1, -2))
    d_a = -_bmm_tc(tt, _bmm_tc(d_t, tt))
    d_m = torch.where(sl, d_a, torch.zeros_like(d_a))

    d_kk = d_m * betaf[..., :, None] * ratio
    dk2 = _bmm_tc(d_kk + d_kk.transpose(-1, -2), kf) + (betaf * gamma)[..., None] * dbgk
    dv = betaf[..., None] * dbv
    p = (dbgk * kf).sum(-1)
    r = (dbv * vf).sum(-1)
    db = r + gamma * p + (d_m * (ratio * kk)).sum(-1)

    e = (d_m * betaf[..., :, None] * kk) * ratio
    dg2_total = LN2 * (e.sum(-1) - e.sum(-2)) + (betaf * p) * LN2 * gamma
    dg = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2_total, [-1]), -1), [-1])

    maybe_sync()
    return (
        dk2.reshape(b, h, nt, c, d_k),
        dv.reshape(b, h, nt, c, d_v),
        db.reshape(b, h, nt, c),
        dg.reshape(b, h, nt, c),
    )


# Lever B batched path is the default; run_k2_serial stays as the proven per-chunk fallback.
run_k2 = run_k2_batched
