"""Desk validation of the K#1 increment-B reverse-loop recurrence (no kernel, no box).

Loads a NT>1 K#1 bundle (``scratch/gen_k1_bundle.py``) and runs the reverse
inter-chunk state scan in pure torch, in the *exact* statement order the CuTe
kernel will use, then checks dh/dh0/dv2 against the bundle's reference expected
outputs. If this matches at ~1e-10, the increment-B contract is correct and the
only remaining risk is the DSL transcription — which the on-box micro-gate covers.

Run: PYTHONPATH=src uv run --no-sync python scratch/k1_incB_desk_check.py --bundle k1_bundle_nt4.pt
"""

from __future__ import annotations

import argparse

import torch


def reverse_loop(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reverse-state scan in torch; mirrors the kernel's per-chunk order exactly.

    Natural [d_k,d_v] orientation (the kernel uses the transposed b_dhT layout to
    avoid in-kernel transposes; the math is identical). b_dh holds dL/dh_{it+1}.
    """
    q, k, w = inp["q"], inp["k"], inp["w"]
    g2, g_last = inp["g2"], inp["g_last"]
    do, dv_local, dht = inp["do"], inp["dv_local"], inp["dht"]
    b, h, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    gamma = torch.exp2(g2)

    b_dh = dht.clone()  # [b,h,d_k,d_v] = dL/dh_{NT}
    dh = torch.zeros(b, h, nt, d_k, d_v, dtype=q.dtype)
    dv2 = torch.zeros(b, h, nt, c, d_v, dtype=q.dtype)

    for it in reversed(range(nt)):
        dh[:, :, it] = b_dh  # dL/dh_{it+1}
        # G1: b_dv = (k @ b_dh) ; uses the OLD b_dh (= dL/dh_{it+1})
        b_dv = k[:, :, it] @ b_dh  # [C,d_k]@[d_k,d_v] -> [C,d_v]
        decay = torch.exp2(g_last[:, :, it][..., None] - g2[:, :, it])  # [b,h,C]
        b_dv = b_dv * decay[..., None] + dv_local[:, :, it]
        dv2[:, :, it] = b_dv
        # GA: t = (q*gamma)^T @ do - w^T @ b_dv
        qg = q[:, :, it] * gamma[:, :, it][..., None]  # [C,d_k]
        t = qg.transpose(-1, -2) @ do[:, :, it] - w[:, :, it].transpose(-1, -2) @ b_dv
        b_dh = torch.exp2(g_last[:, :, it])[..., None, None] * b_dh + t

    return {"dh": dh, "dh0": b_dh, "dv2": dv2}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, default="k1_bundle_nt4.pt")
    args = ap.parse_args()

    payload = torch.load(args.bundle, weights_only=False)
    inp = {key: v.double() for key, v in payload["inputs"].items()}
    exp = {key: v.double() for key, v in payload["expected"].items()}

    got = reverse_loop(inp)
    print(f"bundle={args.bundle}  meta={payload['meta']}")
    ok = True
    # Verifier-style metric: max_abs normalised by the tensor's global inf-norm
    # (scale-aware), the convention the contract gates use. fp64 roundoff through
    # the reference's divergent path (solve_triangular + autograd, gamma spanning
    # ~1e10) sits ~1e-7 of scale; a real contract error would be O(1).
    for name in ("dh", "dv2", "dh0"):
        diff = (got[name] - exp[name]).abs()
        scale = exp[name].abs().max().clamp_min(1e-12)
        scale_rel = (diff.max() / scale).item()
        passed = scale_rel < 1e-6
        ok = ok and passed
        print(f"  {name:4s} max_abs={diff.max().item():.3e}  scale_rel={scale_rel:.3e}  {'OK' if passed else 'FAIL'}")
    print(f"\nmath_correct={ok}")


if __name__ == "__main__":
    main()
