"""K#1 channel-wise (Phase 3) — native Blackwell reverse inter-chunk state scan (GDN-2 B4).

Box bring-up file (-> ``src/.../cute/gdn2_bwd_dhu_cw.py`` at integration). The Phase-3 crown
lift of ``scratch/gdn2_bwd_dhu.py``: per-channel decay g (key axis) instead of one scalar per
token. The recurrence skeleton is identical; the decay folds *inside* the contractions:

  1. dh[i_t] = b_dh
  2. b_dv = (k (.) decay_end) @ b_dh + dv_local         store dv2[i_t]   # decay on key axis
  3. b_dh += (q (.) gamma)^T @ do - wy^T @ b_dv
  4. b_dh = exp2(g_last)[:,None] (.) b_dh               (carry decays per key channel)

decay_end = exp2(g_last - g2) and gamma = exp2(g2) are now ``[C, d_k]``; exp2(g_last) is
``[d_k]`` applied on the key (row) axis of ``b_dh``. Every decay multiplier is <= 1 on the
positions that contribute, so the pre-scaled GEMM operands stay in range (no secondary
normalization needed at this host-orchestrated step). Ground truth =
``kernels.cute.gdn2_assemble.k1_reverse_state_cw_ref`` (validated to fp64 by
``tests/test_gdn2_assemble_cw.py``); the math holds bit-for-bit before any box spend.

Step 1 (THIS): host-orchestrated — the two GEMMs/chunk route through the proven (128,64,128)
config via :func:`gdn2_bwd_wy._mm_tc` (M-pad to 128, K-pad to 128, N split into 64-tiles, so
d_v in {64, 128} both work); decay scalings/carry in fp32 torch. Step 2 fuses the loop in
kernel (inc-B2 lift). BOX-UNTESTED (authored desk-side; math de-risked).

Numerics: fp16 GEMM operands, fp32 accumulate; exp2 on g pre-scaled by RCP_LN2; deterministic.
Off-box this imports cleanly and compiles nothing.
"""
# NB: no `from __future__ import annotations` — keep consistent with the DSL kernel files.

import torch
from scratch.gdn2_bwd_dhu import is_available as _k1_available
from torch import Tensor


def is_available() -> bool:
    return _k1_available()


def run_k1_cw_ref(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validated pure-torch channel-wise reverse scan (the spec the kernel transcribes).

    Device/dtype-agnostic; in fp64 it reproduces the channel-wise bundle to roundoff.
    Returns ``(dh, dv2, dh0)`` head-major chunked. Identical to
    ``gdn2_assemble.k1_reverse_state_cw_ref`` — kept here so the box file is self-contained.
    """
    b, h, nt, c, _d_k = q.shape
    d_v = do.shape[-1]
    gamma = torch.exp2(g2)
    b_dh = dht.clone()
    dh = torch.zeros_like(dht).unsqueeze(2).repeat(1, 1, nt, 1, 1)
    dv2 = torch.zeros(b, h, nt, c, d_v, dtype=q.dtype, device=q.device)

    for it in reversed(range(nt)):
        dh[:, :, it] = b_dh
        decay_end = torch.exp2(g_last[:, :, it][..., None, :] - g2[:, :, it])
        b_dv = (k[:, :, it] * decay_end) @ b_dh + dv_local[:, :, it]
        dv2[:, :, it] = b_dv
        qg = q[:, :, it] * gamma[:, :, it]
        t = qg.transpose(-1, -2) @ do[:, :, it] - wy[:, :, it].transpose(-1, -2) @ b_dv
        b_dh = torch.exp2(g_last[:, :, it])[..., :, None] * b_dh + t

    return dh, dv2, b_dh


def run_k1_incB_serial(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Channel-wise K#1 step 1 — host-orchestrated reverse scan; GEMMs on tcgen05.

    The proven per-(b,hv) fallback. ``b_dh in [d_k, d_v]`` fp32 carried across the reverse
    chunk loop. Per chunk, two matmuls route through the proven (128,64,128) GEMM via
    :func:`gdn2_bwd_wy._mm_tc`: G1 ``(k (.) decay_end) @ b_dh`` (key-axis decay folded into
    the operand) and GA ``[qg | -wy] @ [do | b_dv]^T``. The carry
    ``b_dh = exp2(g_last)[:,None] (.) b_dh + GA`` decays per key channel. Returns
    ``(dh, dv2, dh0)`` head-major chunked. Lever B's :func:`run_k1_incB_batched` is the default.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from scratch.gdn2_bwd_wy import _mm_tc

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    dev = q.device
    n_bh = b * hv

    gamma = torch.exp2(g2)
    qg = q * gamma
    decay_end = torch.exp2(g_last[..., None, :] - g2)  # [B,HV,NT,C,d_k]
    glast_exp = torch.exp2(g_last)  # [B,HV,NT,d_k]
    k_dec = k * decay_end  # pre-scaled key for G1 (operand <= |k|)

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(n_bh, *x.shape[2:])

    kdf, qgf, wyf, dof = _flat(k_dec), _flat(qg), _flat(wy), _flat(do)
    dvlf, gef = _flat(dv_local), _flat(glast_exp)
    b_dh = _flat(dht).contiguous().float()  # [n_bh, d_k, d_v] resident

    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=torch.float32, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=torch.float32, device=dev)

    for i in range(n_bh):
        for it in reversed(range(nt)):
            dh[i, it] = b_dh[i]
            # G1: b_dv[C,d_v] = (k (.) decay_end)[C,d_k] @ b_dh[d_k,d_v]
            b_dv = _mm_tc(kdf[i, it], b_dh[i]) + dvlf[i, it]
            dv2[i, it] = b_dv
            # GA: t[d_k,d_v] = [qg | -wy][2C,d_k]^T @ [do | b_dv][2C,d_v]
            a_ga = torch.cat([qgf[i, it], -wyf[i, it]], dim=0)  # [2C, d_k]
            b_ga = torch.cat([dof[i, it], b_dv], dim=0)  # [2C, d_v]
            t = _mm_tc(a_ga.transpose(0, 1), b_ga)  # [d_k, d_v]
            b_dh[i] = gef[i, it][:, None] * b_dh[i] + t

    torch.cuda.synchronize()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


def run_k1_incB_batched(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lever B — channel-wise reverse scan with the (b,hv) groups batched per step.

    Same recurrence as :func:`run_k1_incB_serial` (identical math), but the ``for i in
    range(n_bh)`` host loop collapses into the batch dim: per reverse chunk ``it`` the two
    matmuls (G1 ``(k (.) decay_end)@b_dh``, GA ``[qg|−wy]@[do|b_dv]^T``) run ONCE over all
    n_bh groups via :func:`gdn2_bwd_dhu._bmm_tc` (d_v in {64,128} via its N-tiling). The
    ``it`` loop stays sequential — it carries ``b_dh``. Returns ``(dh, dv2, dh0)``.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from scratch.gdn2_bwd_dhu import _bmm_tc

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    dev = q.device
    n_bh = b * hv

    gamma = torch.exp2(g2)
    qg = q * gamma
    decay_end = torch.exp2(g_last[..., None, :] - g2)  # [B,HV,NT,C,d_k]
    glast_exp = torch.exp2(g_last)  # [B,HV,NT,d_k]
    k_dec = k * decay_end  # pre-scaled key for G1 (operand <= |k|)

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(n_bh, *x.shape[2:])

    kdf, qgf, wyf, dof = _flat(k_dec), _flat(qg), _flat(wy), _flat(do)
    dvlf, gef = _flat(dv_local), _flat(glast_exp)
    b_dh = _flat(dht).contiguous().float()  # [n_bh, d_k, d_v] resident

    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=torch.float32, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=torch.float32, device=dev)

    for it in reversed(range(nt)):
        dh[:, it] = b_dh
        b_dv = _bmm_tc(kdf[:, it], b_dh) + dvlf[:, it]  # [n_bh,C,d_v]
        dv2[:, it] = b_dv
        a_ga = torch.cat([qgf[:, it], -wyf[:, it]], dim=1).transpose(-1, -2)  # [n_bh,d_k,2C]
        b_ga = torch.cat([dof[:, it], b_dv], dim=1)  # [n_bh,2C,d_v]
        t = _bmm_tc(a_ga, b_ga)  # [n_bh,d_k,d_v]
        b_dh = gef[:, it][:, :, None] * b_dh + t

    torch.cuda.synchronize()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


# ------------------------------------------------------------------
# Lever D — inc-B2 channel-wise: the crown lift of the scalar fused reverse loop.
# Same fusion (one CTA per (b,hv), reverse it-loop in-kernel, b_dh resident, two GEMMs/chunk
# on the proven (128,64,128) config, b_ga GMEM round-trip) with the channel-wise glue: the
# decay folds into the G1 key operand (a_g1 = k⊙decay_end), b_dv adds dv_local directly, and
# the carry decays per key channel (glast ∈ [d_k]). _run_k1_incB2_modelled is the kernel spec;
# scratch/k1_incB2_orchestration_check.py grades it vs the fp64 channel-wise bundle.
# ------------------------------------------------------------------


def _incb2_pack_cw(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
) -> dict[str, Tensor]:
    """Host-precompute channel-wise inc-B2 operand buffers, flat over L = n_bh·NT.

    Mirrors :func:`scratch.gdn2_bwd_dhu._incb2_pack_scalar`; the channel-wise differences are
    the decay folded into the G1 key operand (``a_g1 = (k⊙decay_end)`` padded) and the
    per-key-channel carry (``glast ∈ [L, d_k]``). dv_local is added raw to b_dv (no post-decay).
    """
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    ll = n_bh * nt
    dev = q.device

    gamma = torch.exp2(g2)
    qg = q * gamma
    decay_end = torch.exp2(g_last[..., None, :] - g2)  # [B,HV,NT,C,d_k]
    glast = torch.exp2(g_last)  # [B,HV,NT,d_k]
    k_dec = k * decay_end

    def _flatL(x: Tensor) -> Tensor:
        return x.reshape(ll, *x.shape[3:])

    kdL, qgL, wyL, doL = _flatL(k_dec), _flatL(qg), _flatL(wy), _flatL(do)
    dvlL, geL = _flatL(dv_local), _flatL(glast)

    a_g1 = torch.zeros(ll, d_k, d_k, dtype=q.dtype, device=dev)
    a_g1[:, :c] = kdL
    a_ga = torch.cat([qgL, -wyL], dim=1).transpose(-1, -2).contiguous()  # [L, d_k, 2C]
    b_ga = torch.zeros(ll, d_v, 2 * c, dtype=q.dtype, device=dev)
    b_ga[:, :, :c] = doL.transpose(-1, -2)
    return {"a_g1": a_g1, "a_ga": a_ga, "b_ga": b_ga, "dv_local": dvlL, "glast": geL}


def _run_k1_incB2_modelled(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-torch model of channel-wise inc-B2's in-kernel dataflow (the kernel spec).

    Same statement order + b_ga round-trip as the scalar model; the glue is channel-wise.
    Device/dtype-agnostic; in fp64 it reproduces the channel-wise K#1 bundle to roundoff.
    Returns ``(dh, dv2, dh0)`` head-major chunked.
    """
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    dev = q.device
    n_bh = b * hv
    buf = _incb2_pack_cw(q, k, wy, g2, g_last, do, dv_local)
    a_g1, a_ga, b_ga = buf["a_g1"], buf["a_ga"], buf["b_ga"]
    dv_local_L, glast = buf["dv_local"], buf["glast"]

    b_dh = dht.reshape(n_bh, d_k, d_v).clone()
    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=q.dtype, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=q.dtype, device=dev)

    for i in range(n_bh):
        for it in reversed(range(nt)):
            lid = i * nt + it
            dh[i, it] = b_dh[i]
            bdv_raw = a_g1[lid] @ b_dh[i]  # [128, d_v]; G1 with decay folded into a_g1
            b_dv = bdv_raw[:c] + dv_local_L[lid]  # [C, d_v]
            dv2[i, it] = b_dv
            b_ga[lid, :, c:] = b_dv.transpose(-1, -2)
            t = a_ga[lid] @ b_ga[lid].transpose(-1, -2)  # [d_k, d_v]
            b_dh[i] = glast[lid][:, None] * b_dh[i] + t  # per-key-channel carry

    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


def run_k1_incB2(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lever D channel-wise — the fused persistent reverse scan (the crown lift of run_k1_incB2).

    Reuses the scalar module's ``_incb2_kernel`` verbatim; the channel-wise regime lives entirely
    in the packed buffers (:func:`_incb2_pack_cw` folds decay into ``a_g1`` → the in-kernel G1
    glue multiplies by ``decay = 1``, and ``glast`` is the real per-key-channel [L, d_k]). Values
    are desk-gated bit-exact vs the fp64 channel-wise bundle by k1_incB2_orchestration_check.py.
    Returns ``(dh, dv2, dh0)`` head-major chunked. BOX-UNTESTED — ``run_k1_incB`` stays default.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from scratch.gdn2_bwd_dhu import _incb2_launch

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    buf = _incb2_pack_cw(q, k, wy, g2, g_last, do, dv_local)
    b_dh = dht.reshape(n_bh, d_k, d_v).contiguous().float()
    decay = torch.ones(n_bh * nt, c, dtype=torch.float32, device=q.device)  # folded into a_g1
    dh, dv2, dh0 = _incb2_launch(
        buf["a_g1"], buf["a_ga"], buf["b_ga"], b_dh, decay, buf["dv_local"], buf["glast"],
        n_bh, nt, c, d_k, d_v,
    )
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        dh0.reshape(b, hv, d_k, d_v),
    )


# Lever B batched path is the default; run_k1_incB_serial stays as the proven fallback.
# inc-B2 (run_k1_incB2 above, the fused kernel) supersedes both once box-green.
run_k1_incB = run_k1_incB_batched
