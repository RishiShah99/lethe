"""K#2 — native Blackwell (sm_100) WY / triangular-inverse VJP (GDN backward, B6).

Box bring-up file (-> ``src/flash_mamba_rl/kernels/cute/gdn2_bwd_wy.py`` at integration).
cuLA ships a *scalar* CuTe version (``chunk_wy_dqkg_sm100.py``); ours matches it for
Phase 2, Phase 3 extends to GDN-2 channel-wise. Built on the proven single-tile GEMM
in ``scratch/tcgen05_gemm_smoke.py`` (reused via ``gdn2_bwd_dhu._gemm_aa``).

Contract (``docs/gdn2_phase2_refs.md`` §B6; ground truth = ``references/gdn2_chunkwise.py``
via ``scratch/gen_k2_bundle.py``). One CTA owns one ``(b, hv, chunk)`` — the WY-VJP is
per-chunk independent (no cross-chunk carry, unlike K#1). The map differentiated, per
chunk (CxC; k [C,d_k], v [C,d_v]):

    gamma = exp2(g2);  bv = beta*v;  bgk = beta*gamma*k
    KK = k@k^T;  R_is = gamma_i/gamma_s;  M = strict_lower(beta_i * R_is * KK_is)
    T = (I+M)^-1 (unit lower-tri, GIVEN from fwd);  u = T@bv;  w = T@bgk

VJP, given du(=dv2 from B4) and dw(from B5):
    dbv = T^T@du ;  dbgk = T^T@dw ;  dT = du@bv^T + dw@bgk^T
    dA  = -(T^T@dT@T^T) ;  dM = strict_lower(dA)          # two triangular GEMMs, T REUSED
    dKK = dM*beta_i*R_is ;  dk = (dKK+dKK^T)@k + (beta*gamma)*dbgk ;  dv = beta*dbv
    db  = dbv.v + gamma*(dbgk.k) + (dM*R*KK).sum_s         # bv + bgk + M paths
    dg2 = ln2*(rowsum(dR*R) - colsum(dR*R)) + ln2*gamma*beta*(dbgk.k) ;  dR = dM*beta*KK
    dg_tok = RCP_LN2 * reverse_cumsum(dg2)                 # dg exits reverse-cumsum

Two desk findings (load-bearing, retire the K#1-class box surprises before spend):
  * MATH — ``scratch/k2_wy_desk_check.py`` reproduces the bundle's autograd dk2/dv/db/dg2
    to fp64 roundoff (scale_rel ~1e-7) on 2 shapes. The two-triangular-GEMM inverse
    adjoint with T reused is correct; never re-invert T.
  * NUMERICS — fp16 I/O + fp32 accum + fp32 decay scalings passes the 2e-2 gate with
    ~30x margin (worst scale_rel 6e-4; bf16 5e-3), with NO decay-factoring. The 44% of
    ``bgk`` and 22% of ``T`` entries that underflow fp16 are the small-gamma late-row ones
    whose contribution is negligible — the decay that shrinks them also makes them
    unimportant. So GEMM operands go straight to fp16; decay/ratio scalings stay fp32.

Bring-up ladder (each micro-gated):
  step 1 (THIS): ``run_k2`` host-orchestrated — the 7 GEMMs/chunk routed through inc-A's
     proven (128,64,128) config (M-pad to 128, N split into 64-tiles, K-pad to 128);
     scalings/masks/reductions in fp32 torch. De-risks the GEMM routing on silicon.
  step 2: fused per-chunk kernel (all 7 GEMMs + TMEM in-kernel). The per-chunk
     independence makes a 1-CTA-per-(b,hv,chunk) grid trivial — no reverse carry.

Numerics: fp16 I/O, fp32 accumulate; exp2 on g pre-scaled by RCP_LN2; deterministic, no
atomics. Off-box this imports cleanly and compiles nothing.
"""
# NB: no `from __future__ import annotations` — keep consistent with the DSL kernel files.

import math

import torch
from scratch.gdn2_bwd_dhu import is_available as _k1_available
from torch import Tensor

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
    """Validated pure-torch WY-VJP (the spec/oracle the kernel transcribes).

    Device/dtype-agnostic; in fp64 it reproduces the bundle to roundoff (see
    ``scratch/k2_wy_desk_check.py``). Returns ``(dk2, dv, db, dg2)`` head-major chunked.
    """
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
    ratio = gamma[..., :, None] / gamma[..., None, :]

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
    """``x[M,K] @ y[K,N]`` via inc-A's proven (128,64,128) ``a@b^T`` GEMM, fp32 out.

    M-pad to 128, K-pad to 128, N split into 64-wide tiles — every K#2 matmul lands on
    the ONE proven config. fp16 operands / fp32 accumulate (the verified numeric path).
    """
    from scratch.gdn2_bwd_dhu import _gemm_aa

    f16, dev = torch.float16, x.device
    m, kk = x.shape
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
    """K#2 step 1 — host-orchestrated WY-VJP; the 7 GEMMs/chunk on tcgen05, glue in fp32.

    The proven per-chunk fallback (silicon-verified). Mirrors
    ``gdn2_bwd_dhu.run_k1_incB_host``: every matmul routes through the proven (128,64,128)
    GEMM via :func:`_mm_tc`; decay/ratio scalings, masks and the reverse-cumsum stay fp32
    torch (the verified numeric split). Returns ``(dk2, dv, db, dg2)`` head-major chunked.
    Lever B's :func:`run_k2_batched` is the default; this stays for fallback/debug.
    """
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
        ratio = gamma[:, None] / gamma[None, :]
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

    torch.cuda.synchronize()
    return (
        dk2.reshape(b, h, nt, c, d_k),
        dvf.reshape(b, h, nt, c, d_v),
        dbf.reshape(b, h, nt, c),
        dgf.reshape(b, h, nt, c),
    )


def run_k2_batched(
    k: Tensor, v: Tensor, beta: Tensor, g2: Tensor, t_mat: Tensor, dw: Tensor, du: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Lever B — WY-VJP with all n_idx=b*h*nt chunks batched into one GEMM per matmul-step.

    The WY-VJP is per-chunk independent (no carry), so this is :func:`run_k2_ref`'s structure
    verbatim with each ``@`` replaced by ONE batched tcgen05 GEMM (:func:`_bmm_tc`) over all
    chunks; the glue (ratio/masks/reductions/reverse-cumsum) is already batched in the ref and
    stays fp32 torch. Collapses ~7·n_idx single-CTA launches into ~one batched launch per
    matmul-step — the biggest, lowest-risk lever-B win. Returns ``(dk2, dv, db, dg2)``.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from scratch.gdn2_bwd_dhu import _bmm_tc

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
    ratio = gamma[..., :, None] / gamma[..., None, :]

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

    torch.cuda.synchronize()
    return (
        dk2.reshape(b, h, nt, c, d_k),
        dv.reshape(b, h, nt, c, d_v),
        db.reshape(b, h, nt, c),
        dg.reshape(b, h, nt, c),
    )


# Lever B batched path is the default; run_k2_serial stays as the proven per-chunk fallback.
run_k2 = run_k2_batched
