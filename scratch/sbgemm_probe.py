"""Stage-B GEMM-fusion PROBE (board #2) — measure the ceiling BEFORE building.

The L3/K#1 lesson: build only after the dominant cost is measured. Board #2 proposes
fusing stage-B's batched GEMMs (``_stage_b_vjp_cw_closed``: dqg/da_qk/dk_dec/dwy — four
[B,H,NT] batched matmuls) into one grid-z tcgen05 kernel. SBE already fused the two
intra einsums. This probe decomposes the CURRENT closed stage-B at dv64/L2048 into:

  (a) the 4 batched GEMMs           (what a fusion would replace)
  (b) the SBE intra-einsum call     (already fused; irreducible floor)
  (c) the finalization glue         (elementwise + dg flip-cumsum; materialization)

against the whole-backward savefwd number (5.67 ms) and the K#1/K#2 rungs, so the
decision to build (or record a K#1-style negative) is evidence-led. Event-path, box-only.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/sbgemm_probe.py
"""

from __future__ import annotations

import json
import os

import torch

D_K = 128
D_V = int(os.environ.get("FMR_PROBE_DV", "64"))
B, L, H = 2, 2048, 8
OUT = f"results/sbgemm_probe_dv{D_V}.json"


def _time(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    ev[0].record()
    for _ in range(n):
        fn()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / n


def _inputs(dev, seed=7):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda d: torch.randn(B, L, H, d, generator=gen)  # noqa: E731
    q, k, v = mk(D_K), mk(D_K), mk(D_V)
    g = -torch.rand(B, L, H, D_K, generator=gen) * 0.1
    bg = torch.rand(B, L, H, D_K, generator=gen).sigmoid()
    wg = torch.rand(B, L, H, D_V, generator=gen).sigmoid()
    do = torch.randn(B, L, H, D_V, generator=gen)
    return tuple(x.to(device=dev, dtype=torch.float32).contiguous() for x in (q, k, v, g, bg, wg, do))


def main() -> None:
    if not torch.cuda.is_available():
        print("no CUDA (desk)")
        return

    from lethe.kernels.cute.gdn2_assemble import (
        LN2,
        RCP_LN2,
        _stage_b_vjp_cw_closed,
        _to_chunks,
        assembled_channelwise_gdn2_backward,
        chunkwise_forward_cw,
        masked_decay_rel,
        pick_chunk_len,
    )
    from lethe.kernels.cute.gdn2_backward import _load_box_kernels_cw
    from lethe.kernels.cute.gdn2_sb_einsum import (
        is_available as sbe_available,
    )
    from lethe.kernels.cute.gdn2_sb_einsum import (
        run_sb_einsum,
        sbe_dims_ok,
    )

    dev = torch.device("cuda")
    k1_cw, k2_cw = _load_box_kernels_cw()
    q, k, v, g, bg, wg, do = _inputs(dev)
    cl = pick_chunk_len(L)

    # ---- whole-backward context ----
    def full():
        return assembled_channelwise_gdn2_backward(
            q, k, v, g, bg, wg, do, use_qk_l2norm=True, k1_fn=k1_cw, k2_fn=k2_cw,
        )

    full_ms = _time(full, n=5, warm=3)

    # ---- rebuild the closed-path prefix (fwd -> dv_local -> K#1) ----
    fwd = chunkwise_forward_cw(q, k, v, g, bg, wg, chunk_len=cl, use_qk_l2norm=True)
    do_c = _to_chunks(do, cl).detach()
    dv_local = (fwd.A_qk.transpose(-1, -2) @ do_c).detach()
    dht = torch.zeros_like(fwd.h_list[0]).detach()
    dh_k1, dv2, _dh0 = k1_cw(
        fwd.q.detach(), fwd.k.detach(), fwd.wy.detach(), fwd.g2.detach(),
        fwd.g_last.detach(), do_c, dv_local, dht,
    )
    dh_k1, dv2 = dh_k1.detach(), dv2.detach()

    # ---- (0) whole closed stage-B, SBE on vs off ----
    os.environ.pop("FMR_DISABLE_SBE", None)
    sbe_on_ms = _time(lambda: _stage_b_vjp_cw_closed(fwd, do, dh_k1, dv2))
    os.environ["FMR_DISABLE_SBE"] = "1"
    sbe_off_ms = _time(lambda: _stage_b_vjp_cw_closed(fwd, do, dh_k1, dv2))
    os.environ.pop("FMR_DISABLE_SBE", None)

    # ---- decompose the closed body (fields exactly as in _stage_b_vjp_cw_closed) ----
    h_entry = fwd.h[:, :, :-1].detach()
    gamma, g2, g_last = fwd.gamma.detach(), fwd.g2.detach(), fwd.g_last.detach()
    qd, kd, v_new = fwd.q.detach(), fwd.k.detach(), fwd.v_new.detach()
    c = fwd.chunk_len
    lower_incl = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), 0)
    decay_end = torch.exp2(g_last[..., None, :] - g2)
    het = h_entry.transpose(-1, -2)
    vnt = v_new.transpose(-1, -2)
    dht_t = dh_k1.transpose(-1, -2)

    def gemms():
        dqg = do_c @ het
        da_qk = do_c @ vnt
        da_qk = torch.where(lower_incl, da_qk, torch.zeros_like(da_qk))
        dk_dec = v_new @ dht_t
        dwy = -dv2 @ het
        return dqg, da_qk, dk_dec, dwy

    dqg, da_qk, dk_dec, dwy = gemms()
    gemms_ms = _time(gemms)

    sbe_ok = sbe_available() and sbe_dims_ok(c, D_K)
    if sbe_ok:
        sbe_ms = _time(lambda: run_sb_einsum(da_qk, kd, qd, g2))
        dq_intra, dk_intra = run_sb_einsum(da_qk, kd, qd, g2)
    else:
        drel = masked_decay_rel(g2)
        sbe_ms = _time(
            lambda: (
                torch.einsum("...is,...sd,...isd->...id", da_qk, kd, drel),
                torch.einsum("...is,...id,...isd->...sd", da_qk, qd, drel),
            )
        )
        dq_intra = torch.einsum("...is,...sd,...isd->...id", da_qk, kd, drel)
        dk_intra = torch.einsum("...is,...id,...isd->...sd", da_qk, qd, drel)

    def finalize():
        dq_b = dqg * gamma + dq_intra
        dk_b = dk_intra + dk_dec * decay_end
        dde = dk_dec * kd * decay_end
        dg2 = LN2 * (dqg * qd * gamma + qd * dq_intra - kd * dk_intra - dde)
        dg_last = LN2 * (torch.exp2(g_last) * (dh_k1 * h_entry).sum(-1) + dde.sum(-2))
        dg2 = dg2 + torch.cat([torch.zeros_like(dg2[..., :-1, :]), dg_last.unsqueeze(-2)], dim=-2)
        dg_b = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2, [-2]), -2), [-2])
        return dq_b, dk_b, dg_b

    fin_ms = _time(finalize)

    # ---- K#1 / K#2 rungs for context ----
    k1_ms = _time(lambda: k1_cw(
        fwd.q.detach(), fwd.k.detach(), fwd.wy.detach(), fwd.g2.detach(),
        fwd.g_last.detach(), do_c, dv_local, dht,
    ))
    k2_ms = _time(lambda: k2_cw(
        fwd.k.detach(), fwd.v.detach(), fwd.b.detach(), fwd.w_gate.detach(),
        fwd.g2.detach(), fwd.T.detach(), dwy.detach(), dv2,
    ))

    out = {
        "shape": [B, L, H], "d_v": D_V, "chunk_len": c,
        "full_bwd_ms": full_ms,
        "closed_stageB_sbe_on_ms": sbe_on_ms,
        "closed_stageB_sbe_off_ms": sbe_off_ms,
        "decomp": {
            "four_gemms_ms": gemms_ms,
            "sbe_einsum_ms": sbe_ms,
            "finalize_glue_ms": fin_ms,
            "sbe_kernel_used": sbe_ok,
        },
        "k1_rung_ms": k1_ms,
        "k2_rung_ms": k2_ms,
        "savefwd_ref_ms": 5.67,
    }
    print(json.dumps(out, indent=2))
    fusible = gemms_ms + fin_ms
    print(f"\n--- CEILING ANALYSIS (dv{D_V}, L{L}) ---")
    print(f"closed stage-B (SBE on)      : {sbe_on_ms:7.3f} ms")
    print(f"  4 GEMMs                    : {gemms_ms:7.3f} ms")
    print(f"  SBE einsum (irreducible)   : {sbe_ms:7.3f} ms")
    print(f"  finalize glue              : {fin_ms:7.3f} ms")
    print(f"fusible (GEMMs+finalize)     : {fusible:7.3f} ms  <- ceiling on the win")
    print(f"as share of savefwd 5.67 ms  : {100 * fusible / 5.67:5.1f}%")
    print(f"as share of full bwd {full_ms:.1f}ms : {100 * fusible / full_ms:5.1f}%")
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
