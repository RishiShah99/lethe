"""Parameterization Rosetta stone — fla GDN-2 vs our GDN-2 oracle.

Inlines ``naive_recurrent_gdn2`` from the fla source to be self-contained.

Source:
  URL:    https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gdn2/naive.py
  Commit: dd7867d261fbe2f30868e0b62bf6963e9ea38e9e  (2026-06-03)
  License: MIT — https://github.com/fla-org/flash-linear-attention/blob/main/LICENSE

The script:
  1. Generates seeded inputs in OUR convention.
  2. Maps them into fla's convention under each of 4 candidate mappings.
  3. Runs fla naive fwd+bwd vs our ``reference_gdn2_backward`` under each mapping.
  4. Reports ``scale_rel`` per grad, declares the winning mapping (<1e-3 for fp32),
     writes JSON (--out).

Mappings tested
---------------
  A  identical inputs, use_qk_l2norm=False in ours    — tests the bare recurrence match
  B  identical inputs, use_qk_l2norm=True  in ours    — tests with our default L2-norm
  C  b/w in [0,1] vs fla range [0,2]: multiply ours b/w by 2 before feeding fla  — tests gate scale
  D  q/k L2-norm applied outside (so both sides see normed q/k), no-norm internally
     — tests whether fla expects pre-normed q/k

A clean parity (scale_rel < 1e-3) in ANY mapping = the Rosetta is solved.
The first clean mapping is declared winner; if none, the closest miss is reported.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from torch import Tensor

from flash_mamba_rl.kernels.references.gdn_backward import reference_gdn2_backward

# ---------------------------------------------------------------------------
# Inlined fla naive_recurrent_gdn2
# Source: https://github.com/fla-org/flash-linear-attention @ dd7867d / fla/ops/gdn2/naive.py
# License: MIT
# ---------------------------------------------------------------------------


def _fla_naive_recurrent_gdn2(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Token-by-token reference forward pass for GDN-2 (fla convention).

    Args:
        q: queries [B, T, H, K]
        k: keys    [B, T, H, K]
        v: values  [B, T, H, V]
        g: log-decay [B, T, H, K]  (natural-log base; should be <= 0)
        b: channel-wise erase gate [B, T, H, K]  (range typically [0, 2])
        w: channel-wise write gate [B, T, H, V]  (range typically [0, 1])
        scale: attention scale; defaults to 1/sqrt(K)
        initial_state: optional [B, H, K, V] fp32 initial state
        output_final_state: whether to return the final state

    Returns:
        o:           [B, T, H, V]
        final_state: [B, H, K, V] or None
    """
    if scale is None:
        scale = float(q.shape[-1] ** -0.5)
    q2, k2, v2, g2, b2, w2 = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, g, b, w))
    B, H, T, K = k2.shape
    V = v2.shape[-1]
    o2 = torch.zeros(B, H, T, V, device=v2.device, dtype=torch.float32)
    h = torch.zeros(B, H, K, V, device=v2.device, dtype=torch.float32)
    if initial_state is not None:
        h = initial_state.to(torch.float32).clone()
    q2 = q2 * scale

    for t in range(T):
        b_q = q2[:, :, t]
        b_k = k2[:, :, t]
        b_v = v2[:, :, t]
        b_g = g2[:, :, t]
        b_b = b2[:, :, t]
        b_w = w2[:, :, t]
        h = h * b_g.exp().unsqueeze(-1)
        erase = ((b_b * b_k).unsqueeze(-1) * h).sum(-2)
        b_v_new = b_w * b_v - erase
        h = h + b_k.unsqueeze(-1) * b_v_new.unsqueeze(-2)
        o2[:, :, t] = (b_q.unsqueeze(-1) * h).sum(-2)

    o2 = o2.transpose(1, 2).contiguous().to(v.dtype)
    final = h if output_final_state else None
    return o2, final


# ---------------------------------------------------------------------------
# L2-norm helper (mirrors our oracle exactly)
# ---------------------------------------------------------------------------


def _l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# scale_rel helper
# ---------------------------------------------------------------------------


def _scale_rel(a: Tensor, b: Tensor) -> float:
    """||a - b|| / (||b|| + 1e-6)."""
    return float((a - b).norm() / (b.norm() + 1e-6))


# ---------------------------------------------------------------------------
# Reference backward via autograd through fla naive
# ---------------------------------------------------------------------------


def _fla_naive_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    scale: float | None = None,
) -> dict[str, Tensor]:
    """Backward through the fla naive forward (autograd, fp64 leaves)."""
    q_l = q.detach().double().requires_grad_(True)
    k_l = k.detach().double().requires_grad_(True)
    v_l = v.detach().double().requires_grad_(True)
    g_l = g.detach().double().requires_grad_(True)
    b_l = b.detach().double().requires_grad_(True)
    w_l = w.detach().double().requires_grad_(True)
    leaves = [q_l, k_l, v_l, g_l, b_l, w_l]

    o, _ = _fla_naive_recurrent_gdn2(q_l, k_l, v_l, g_l, b_l, w_l, scale=scale)
    grads = torch.autograd.grad(o, leaves, do.double())
    names = ["dq", "dk", "dv", "dg", "db", "dw"]
    return dict(zip(names, (g.float() for g in grads), strict=True))


# ---------------------------------------------------------------------------
# Candidate mappings
# ---------------------------------------------------------------------------


def _run_mapping(
    name: str,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    b: Tensor,
    w: Tensor,
    do: Tensor,
    *,
    # how to transform our inputs before feeding fla
    fla_q: Tensor,
    fla_k: Tensor,
    fla_b: Tensor,
    fla_w: Tensor,
    fla_scale: float | None,
    # how to obtain our reference grads
    our_use_qk_l2norm: bool,
) -> dict[str, Any]:
    """Run fla naive fwd+bwd and our oracle; compare grads."""
    # fla backward
    fla_grads = _fla_naive_backward(fla_q, fla_k, v, g, fla_b, fla_w, do, scale=fla_scale)

    # our reference backward (fp64 inputs)
    our = reference_gdn2_backward(
        q.double(),
        k.double(),
        v.double(),
        g.double(),
        b.double(),
        w.double(),
        do.double(),
        scale=fla_scale,
        use_qk_l2norm=our_use_qk_l2norm,
    )

    rows: dict[str, float] = {}
    for grad_name, our_g, fla_g in (
        ("dq", our.grad_q, fla_grads["dq"]),
        ("dk", our.grad_k, fla_grads["dk"]),
        ("dv", our.grad_v, fla_grads["dv"]),
        ("dg", our.grad_g, fla_grads["dg"]),
        ("db", our.grad_b, fla_grads["db"]),
        ("dw", our.grad_w, fla_grads["dw"]),
    ):
        rows[grad_name] = _scale_rel(our_g.float(), fla_g.float())

    max_sr = max(rows.values())
    win = max_sr < 1e-3
    return {"mapping": name, "scale_rel": rows, "max_scale_rel": max_sr, "win": win}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--b", type=int, default=2)
    p.add_argument("--t", type=int, default=32)
    p.add_argument("--h", type=int, default=3)
    p.add_argument("--dk", type=int, default=16)
    p.add_argument("--dv", type=int, default=16)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    B, T, H, dk, dv = args.b, args.t, args.h, args.dk, args.dv
    gen = torch.Generator().manual_seed(args.seed)

    # OUR convention: q/k raw (not pre-normed), g <= 0, b/w in [0,1]
    q_raw = torch.randn(B, T, H, dk, generator=gen, dtype=torch.float32)
    k_raw = torch.randn(B, T, H, dk, generator=gen, dtype=torch.float32)
    v = torch.randn(B, T, H, dv, generator=gen, dtype=torch.float32)
    g = -torch.rand(B, T, H, dk, generator=gen, dtype=torch.float32) * 0.5
    b_our = torch.rand(B, T, H, dk, generator=gen, dtype=torch.float32) * 0.8 + 0.1  # [0.1, 0.9]
    w_our = torch.rand(B, T, H, dv, generator=gen, dtype=torch.float32) * 0.8 + 0.1
    do = torch.randn(B, T, H, dv, generator=gen, dtype=torch.float32)

    # Pre-normed q/k (used for mapping D)
    q_normed = _l2norm(q_raw)
    k_normed = _l2norm(k_raw)

    default_scale = float(dk**-0.5)

    results = []

    # Mapping A: identical inputs, our L2-norm OFF (bare recurrence)
    # Hypothesis: same update rule, same scale, no L2-norm -> exact match
    results.append(
        _run_mapping(
            "A_no_l2norm",
            q_raw,
            k_raw,
            v,
            g,
            b_our,
            w_our,
            do,
            fla_q=q_raw,
            fla_k=k_raw,
            fla_b=b_our,
            fla_w=w_our,
            fla_scale=default_scale,
            our_use_qk_l2norm=False,
        )
    )

    # Mapping B: identical inputs, our L2-norm ON (default)
    # Hypothesis: ours norms q/k but fla doesn't -> mismatch in dq/dk but similar dv
    results.append(
        _run_mapping(
            "B_with_l2norm",
            q_raw,
            k_raw,
            v,
            g,
            b_our,
            w_our,
            do,
            fla_q=q_raw,
            fla_k=k_raw,
            fla_b=b_our,
            fla_w=w_our,
            fla_scale=default_scale,
            our_use_qk_l2norm=True,
        )
    )

    # Mapping C: fla b/w in [0,2] range — multiply our b/w by 2 before fla
    # Hypothesis: the bar script's grad_w=None arose from a gate-range mismatch
    results.append(
        _run_mapping(
            "C_bw_scaled_x2",
            q_raw,
            k_raw,
            v,
            g,
            b_our,
            w_our,
            do,
            fla_q=q_raw,
            fla_k=k_raw,
            fla_b=b_our * 2.0,
            fla_w=w_our * 2.0,
            fla_scale=default_scale,
            our_use_qk_l2norm=False,
        )
    )

    # Mapping D: pre-normalize q/k before both sides (fla also sees normed q/k, scale=1)
    # Hypothesis: fla expects pre-normed inputs and a scale of 1
    results.append(
        _run_mapping(
            "D_prenorm_scale1",
            q_normed,
            k_normed,
            v,
            g,
            b_our,
            w_our,
            do,
            fla_q=q_normed,
            fla_k=k_normed,
            fla_b=b_our,
            fla_w=w_our,
            fla_scale=1.0,
            our_use_qk_l2norm=False,
        )
    )

    # Summarise
    winners = [r for r in results if r["win"]]
    if winners:
        verdict = (
            f"MATCH — {winners[0]['mapping']} (max_scale_rel={winners[0]['max_scale_rel']:.2e})"
        )
    else:
        best = min(results, key=lambda r: r["max_scale_rel"])
        verdict = (
            f"NO_MATCH — closest miss: {best['mapping']}"
            f" (max_scale_rel={best['max_scale_rel']:.2e})"
        )

    print(f"\nRosetta verdict: {verdict}\n")
    for r in results:
        sr = r["scale_rel"]
        win_marker = " *** WIN ***" if r["win"] else ""
        print(
            f"  {r['mapping']:25s}  max={r['max_scale_rel']:.2e}"
            f"  dq={sr['dq']:.2e} dk={sr['dk']:.2e} dv={sr['dv']:.2e}"
            f"  dg={sr['dg']:.2e} db={sr['db']:.2e} dw={sr['dw']:.2e}"
            f"{win_marker}"
        )

    out_doc = {
        "verdict": verdict,
        "mappings": results,
        "inputs": {
            "B": B,
            "T": T,
            "H": H,
            "dk": dk,
            "dv": dv,
            "seed": args.seed,
        },
        "fla_source": {
            "url": "https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gdn2/naive.py",
            "commit": "dd7867d261fbe2f30868e0b62bf6963e9ea38e9e",
            "date": "2026-06-03",
        },
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_doc, f, indent=2)
        print(f"\nResults -> {args.out}")
    else:
        print(json.dumps(out_doc, indent=2))


if __name__ == "__main__":
    main()
