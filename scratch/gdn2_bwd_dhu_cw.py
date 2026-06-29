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


def run_k1_incB(
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

    ``b_dh in [d_k, d_v]`` fp32 carried across the reverse chunk loop. Per chunk, two
    matmuls route through the proven (128,64,128) GEMM via :func:`gdn2_bwd_wy._mm_tc`:
    G1 ``(k (.) decay_end) @ b_dh`` (the key-axis decay folded into the operand) and GA
    ``[qg | -wy] @ [do | b_dv]^T``. The carry ``b_dh = exp2(g_last)[:,None] (.) b_dh + GA``
    decays per key channel. BOX-UNTESTED (math de-risked via run_k1_cw_ref). Returns
    ``(dh, dv2, dh0)`` head-major chunked.
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
