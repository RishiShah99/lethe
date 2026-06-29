"""Phase-2 integration box gate — the assembled native backward vs the oracle on B200.

The desk reduction gate (``verify_gdn2_reduction_op_all_grads``, refs) proves the
assembly wiring + all 12 contract properties on CPU. This is its silicon counterpart:
run the FULL assembly with the real tcgen05 kernels (``native_gdn2_backward`` with
``is_available`` lifted) at the kernels' production tile dims (d_k=128, d_v=64, C=64)
and grade the six grads against the independent token-serial oracle, across a small
correctness grid, plus determinism and an fp16-vs-fp32 mixed-precision check.

Why focused, not the 12-gate harness with kernels: the kernels are dim-locked to
128/64/64, but several gates use tiny shapes and CMP-02 (``gradcheck``) at d_v=64
would re-invoke the kernel O(N^2) ~ 1e7 times. Value-correctness (CMP), determinism
(ORD-02) and mixed-precision (PRC-02) are the gates that actually exercise the kernel;
they carry the silicon verdict. Differentiability/exception/shape gates are CPU/refs
(the desk reduction gate). Scalar regime: g channel-constant, b = w = beta; per-channel
grad_g/grad_b/grad_w are graded by channel-sum (the only quantity a scalar kernel gives).

Run on box:
  PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python scratch/gdn2_integration_box.py --out results/gdn2_integration_box.json
Desk dry-run (refs stand in for the kernels; proves the harness + shapes, no box):
  PYTHONPATH="src;." uv run --no-sync python scratch/gdn2_integration_box.py --ref-candidate
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import torch

import flash_mamba_rl.kernels.cute.gdn2_backward as gdn2_native
from flash_mamba_rl.kernels.cute.gdn2_assemble import assembled_scalar_gdn2_backward
from flash_mamba_rl.kernels.references.gdn_backward import Gdn2Grads, reference_gdn2_backward

D_K = 128
D_V = 64
CHUNK = 64
# (batch, seqlen, nheads) at the kernel tile dims; seqlen % CHUNK == 0.
GRID = [(1, 64, 1), (1, 128, 2), (2, 128, 1), (1, 256, 1)]
# Re-pinned against the real assembled tcgen05 path on B200 (worst observed scale_rel
# 3.29e-3 across the grid; results/gdn2_integration_box.json). 5e-3 = ~1.5x headroom,
# tighter than the 2e-2 K#1/K#2 micro-gate floor while staying bf16-realistic.
ATOL_BF16 = 5e-3


def _inputs(b: int, t: int, h: int, dtype: torch.dtype, dev: torch.device, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g_s = -torch.rand(b, t, h, generator=gen) * 0.1
    beta = torch.rand(b, t, h, generator=gen) * 0.8 + 0.1
    do = torch.randn(b, t, h, D_V, generator=gen)
    g = g_s.unsqueeze(-1).expand(b, t, h, D_K).contiguous()
    bg = beta.unsqueeze(-1).expand(b, t, h, D_K).contiguous()
    wg = beta.unsqueeze(-1).expand(b, t, h, D_V).contiguous()

    def cast(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=dev, dtype=dtype).contiguous()

    return cast(q), cast(k), cast(v), cast(g), cast(bg), cast(wg), cast(do)


def _reduced(grads: Gdn2Grads) -> dict[str, torch.Tensor]:
    """The 5 scalar-recoverable views: q/k/v channel-wise, g + beta by channel-sum."""
    return {
        "grad_q": grads.grad_q.float(),
        "grad_k": grads.grad_k.float(),
        "grad_v": grads.grad_v.float(),
        "grad_g": grads.grad_g.float().sum(-1),
        "grad_beta": grads.grad_b.float().sum(-1) + grads.grad_w.float().sum(-1),
    }


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a - e).abs().max() / e.abs().max().clamp_min(1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--ref-candidate", action="store_true", help="desk dry-run: refs, not kernels")
    ap.add_argument("--atol", type=float, default=ATOL_BF16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" and not args.ref_candidate else torch.float32

    candidate: Callable[..., Gdn2Grads | None]
    if args.ref_candidate:
        candidate = assembled_scalar_gdn2_backward  # refs; desk dry-run
        mode = "refs(desk)"
    else:
        gdn2_native.is_available = lambda device=None: True  # lift the box gate for this run
        candidate = gdn2_native.native_gdn2_backward
        mode = "native(box)"

    results: list[dict[str, object]] = []
    worst = 0.0
    ok = True
    for b, t, h in GRID:
        q, k, v, g, bg, wg, do = _inputs(b, t, h, dtype, dev, seed=b * 100 + t)
        got = candidate(q, k, v, g, bg, wg, do)
        if got is None:
            results.append({"shape": [b, t, h], "error": "candidate returned None"})
            ok = False
            continue
        # oracle in fp32 from the same (rounded) bits — the mixed-precision contract.
        orc = reference_gdn2_backward(*(x.float() for x in (q, k, v, g, bg, wg, do)))
        cand_r, orc_r = _reduced(got), _reduced(orc)
        rels = {name: _scale_rel(cand_r[name], orc_r[name]) for name in cand_r}
        shape_worst = max(rels.values())
        worst = max(worst, shape_worst)

        # determinism: bit-identical re-run.
        got2 = candidate(q, k, v, g, bg, wg, do)
        assert got2 is not None
        det = bool(torch.equal(got.grad_v, got2.grad_v) and torch.equal(got.grad_q, got2.grad_q))

        passed = shape_worst <= args.atol and det
        ok = ok and passed
        results.append(
            {"shape": [b, t, h], "scale_rel": rels, "worst": shape_worst, "deterministic": det,
             "passed": passed}
        )
        print(f"  shape=({b},{t},{h}) worst={shape_worst:.3e} det={det} {'OK' if passed else 'FAIL'}")

    verdict = {"mode": mode, "device": str(dev), "dtype": str(dtype), "atol": args.atol,
               "worst_scale_rel": worst, "results": results, "GO": ok}
    print(f"\nmode={mode} worst_scale_rel={worst:.3e} GO={ok}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
