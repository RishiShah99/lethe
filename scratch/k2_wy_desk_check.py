"""Desk validation of the K#2 (B6) WY / triangular-inverse VJP (no kernel, no box).

Loads a K#2 bundle (``scratch/gen_k2_bundle.py``) and runs the WY-VJP in pure torch
as the *hand-derivation the CuTe kernel will implement* — the two-triangular-GEMM
inverse adjoint with ``T`` reused from the forward (NEVER re-inverted) — then checks
dk2/dv/db/dg2 against the bundle's autograd-derived expected outputs. If this matches
at fp64 roundoff, the K#2 contract is correct and the only remaining risk is the DSL
transcription (covered by the on-box micro-gate). Mirror of ``k1_incB_desk_check.py``.

The map differentiated (per chunk, CxC; k [C,d_k], v [C,d_v]):
    gamma = exp2(g2);  bv = beta·v;  bgk = beta·gamma·k
    KK = k·kᵀ;  R_is = gamma_i/gamma_s;  M = strict_lower(beta_i · R_is · KK_is)
    T = (I+M)^{-1} (unit lower-tri, given);  u = T·bv;  w = T·bgk
VJP, given du(=dv2 from B4) and dw(from B5):
    dbv = Tᵀ·du ;  dbgk = Tᵀ·dw ;  dT = du·bvᵀ + dw·bgkᵀ
    dA = -Tᵀ·dT·Tᵀ ;  dM = strict_lower(dA)            (the two triangular GEMMs)
    push dM/dbv/dbgk back onto k, v, beta, g2; dg exits reverse-cumsum·RCP_LN2.

Run: PYTHONPATH=src uv run --no-sync python scratch/k2_wy_desk_check.py --bundle k2_bundle_nt4.pt
"""

from __future__ import annotations

import argparse
import math

import torch

LN2 = math.log(2.0)
RCP_LN2 = 1.0 / LN2


def wy_vjp(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """WY-VJP in torch; mirrors the kernel's GEMM-shaped statement order exactly."""
    k, v, beta, g2, t_mat = inp["k"], inp["v"], inp["beta"], inp["g2"], inp["T"]
    dw, du = inp["dw"], inp["du"]
    b, h, nt, c, _ = k.shape

    def _flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(b * h * nt, *x.shape[3:])

    kf, vf, betaf, g2f, tf = _flat(k), _flat(v), _flat(beta), _flat(g2), _flat(t_mat)
    dwf, duf = _flat(dw), _flat(du)
    dev = kf.device
    strict_lower = torch.tril(torch.ones(c, c, dtype=torch.bool, device=dev), -1)

    gamma = torch.exp2(g2f)  # [L,C]
    bv = betaf[..., None] * vf  # [L,C,d_v]
    bgk = (betaf * gamma)[..., None] * kf  # [L,C,d_k]
    kk = kf @ kf.transpose(-1, -2)  # [L,C,C]
    ratio = gamma[..., :, None] / gamma[..., None, :]  # R_is = gamma_i/gamma_s

    tt = tf.transpose(-1, -2)  # Tᵀ (upper-tri)
    dbv = tt @ duf  # [L,C,d_v]
    dbgk = tt @ dwf  # [L,C,d_k]
    d_t = duf @ bv.transpose(-1, -2) + dwf @ bgk.transpose(-1, -2)  # dT [L,C,C]
    d_a = -(tt @ d_t @ tt)  # inverse adjoint; two triangular GEMMs
    d_m = torch.where(strict_lower, d_a, torch.zeros_like(d_a))  # dM

    # push dM through M_is = beta_i · R_is · KK_is
    d_kk = d_m * betaf[..., :, None] * ratio  # [L,C,C]
    dk_from_kk = (d_kk + d_kk.transpose(-1, -2)) @ kf  # KK = k·kᵀ

    # push dbgk/dbv through the row scalings
    p = (dbgk * kf).sum(-1)  # dbgk_i·k_i  [L,C]
    r = (dbv * vf).sum(-1)  # dbv_i·v_i   [L,C]
    dk_from_bgk = (betaf * gamma)[..., None] * dbgk
    dk = dk_from_kk + dk_from_bgk
    dv = betaf[..., None] * dbv

    db = r + gamma * p + (d_m * (ratio * kk)).sum(-1)  # bv + bgk + M paths

    # decay grads -> g2, then reverse-cumsum to raw per-token g
    e = (d_m * betaf[..., :, None] * kk) * ratio  # dR_is · R_is
    dg2_from_r = LN2 * (e.sum(-1) - e.sum(-2))  # rowsum_i - colsum_i
    dg2_from_gamma = (betaf * p) * LN2 * gamma
    dg2_total = dg2_from_r + dg2_from_gamma  # grad wrt g2 (log2 cumsum)
    dg_tok = RCP_LN2 * torch.flip(torch.cumsum(torch.flip(dg2_total, [-1]), -1), [-1])

    def _unflat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(b, h, nt, *x.shape[1:])

    return {
        "dk2": _unflat(dk),
        "dv": _unflat(dv),
        "db": _unflat(db),
        "dg2": _unflat(dg_tok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default="k2_bundle_nt4.pt")
    args = ap.parse_args()

    payload = torch.load(args.bundle, weights_only=False)
    inp = {key: v.double() for key, v in payload["inputs"].items()}
    exp = {key: v.double() for key, v in payload["expected"].items()}

    got = wy_vjp(inp)
    print(f"bundle={args.bundle}  meta={payload['meta']}")
    ok = True
    for name in ("dk2", "dv", "db", "dg2"):
        diff = (got[name] - exp[name]).abs()
        scale = exp[name].abs().max().clamp_min(1e-12)
        scale_rel = (diff.max() / scale).item()
        passed = scale_rel < 1e-6
        ok = ok and passed
        print(
            f"  {name:4s} max_abs={diff.max().item():.3e}  "
            f"scale_rel={scale_rel:.3e}  {'OK' if passed else 'FAIL'}"
        )
    print(f"\nmath_correct={ok}")


if __name__ == "__main__":
    main()
