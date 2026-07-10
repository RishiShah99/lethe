"""K#1 channel-wise (Phase 3), native Blackwell reverse inter-chunk state scan (GDN-2 B4)."""
# NB: no `from __future__ import annotations`, keep consistent with the DSL kernel files.

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_bwd_dhu import is_available as _k1_available
from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync


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
    """Validated pure-torch channel-wise reverse scan (the spec the kernel transcribes)."""
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
    """Channel-wise K#1, host-orchestrated reverse scan on tcgen05; the per-(b,hv) fallback."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_wy import _mm_tc

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

    maybe_sync()
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
    """Lever B, channel-wise reverse scan with the (b,hv) groups batched per step."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_dhu import _bmm_tc

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

    maybe_sync()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


def _incb2_pack_cw(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
) -> dict[str, Tensor]:
    """Host-precompute channel-wise inc-B2 operand buffers, flat over L = n_bh·NT."""
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
    """Pure-torch model of channel-wise inc-B2's in-kernel dataflow (the kernel spec)."""
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
    """Lever D channel-wise, the fused persistent reverse scan."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    from lethe.kernels.cute.gdn2_bwd_dhu import _incb2_launch

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    buf = _incb2_pack_cw(q, k, wy, g2, g_last, do, dv_local)
    b_dh = dht.reshape(n_bh, d_k, d_v).clone().float()
    decay = torch.ones(n_bh * nt, c, dtype=torch.float32, device=q.device)  # folded into a_g1
    dh, dv2, dh0 = _incb2_launch(
        buf["a_g1"],
        buf["a_ga"],
        buf["b_ga"],
        b_dh,
        decay,
        buf["dv_local"],
        buf["glast"],
        n_bh,
        nt,
        c,
        d_k,
        d_v,
    )
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        dh0.reshape(b, hv, d_k, d_v),
    )


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
    """Default cw K#1: Level-3 fused kernel > Level-2 epilogue-fused > Lever B, by tile fit."""
    import os

    from lethe.kernels.cute.gdn2_bwd_dhu_l2 import l2_dims_ok, run_k1_incB_l2
    from lethe.kernels.cute.gdn2_bwd_dhu_l3 import l3_dims_ok, run_k1_incB2_v3

    c, d_k = q.shape[3], q.shape[4]
    if not os.environ.get("FMR_DISABLE_L3") and l3_dims_ok(c, d_k, do.shape[-1]):
        dh, dv2, dh0 = run_k1_incB2_v3(q, k, wy, g2, g_last, do, dv_local, dht, cw=True)
        return dh.float(), dv2.float(), dh0
    if not os.environ.get("FMR_DISABLE_L2") and l2_dims_ok(c, d_k, do.shape[-1]):
        dh, dv2, dh0 = run_k1_incB_l2(q, k, wy, g2, g_last, do, dv_local, dht)
        return dh.float(), dv2.float(), dh0
    return run_k1_incB_batched(q, k, wy, g2, g_last, do, dv_local, dht)


# run_k1_incB_serial is the fallback; run_k1_incB2 supersedes it once hardware-validated.
