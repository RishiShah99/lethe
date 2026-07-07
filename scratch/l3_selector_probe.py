"""Burst-4 forensic — did the selector route L3, in eager AND inside the capture?

The e1/e2 graph benches returned bit-identical replay times (69.660 vs 69.661 ms
at L2048) while the rung bench shows L3-vs-L2 differ by ~5.6 ms of device time —
so the two captured graphs must contain the same K#1. Values cannot adjudicate
(L3/L2 are bit-identical: same GEMM config, same fp16 landing points — the rung
scale_rels match to the last digit). This probe counts CALLS:

  1. spy on gdn2_bwd_dhu_l3.run_k1_incB2_v3; call the selector eager -> count
  2. same spy around a GraphedBackward build of the cw assembly -> count in-capture
  3. fresh-input replay correctness: graphed(x2) vs eager(x2) with x2 != capture
     inputs — a stale-output replay (tcgen05 launches escaped the graph) shows up
     as graphed(x2) == eager(x1) instead.

Run: PYTHONPATH=src:. ~/cuteenv/bin/python scratch/l3_selector_probe.py --out probe.json
"""

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    res: dict[str, Any] = {}
    try:
        from scratch.k1_incb2_v3_unroll import _build_bundle

        import lethe.kernels.cute.gdn2_bwd_dhu_l3 as l3
        from lethe.kernels.cute.gdn2_bwd_dhu_cw import run_k1_incB
        from lethe.verifier.timing import benchmark

        calls = {"n": 0}
        orig = l3.run_k1_incB2_v3

        def spy(*a: Any, **kw: Any) -> Any:
            calls["n"] += 1
            return orig(*a, **kw)

        l3.run_k1_incB2_v3 = spy  # type: ignore[assignment]

        # 1. eager selector routing + timing at the L2048 bench group count
        payload = _build_bundle("cw", nt=32, b=2, h=8)
        inp = {k: v.cuda() for k, v in payload["inputs"].items()}
        k1_args = (
            inp["q"], inp["k"], inp["wy"], inp["g2"], inp["g_last"],
            inp["do"], inp["dv_local"], inp["dht"],
        )
        calls["n"] = 0
        run_k1_incB(*k1_args)
        res["eager_selector_l3_calls"] = calls["n"]

        probe = torch.zeros(1, device="cuda")
        res["eager_selector_ms"] = benchmark(
            lambda _p: run_k1_incB(*k1_args), (probe,), warmup=3, trials=20
        ).median_ms

        # 2 + 3. capture routing + fresh-input replay correctness (assembly, L512 dv=64)
        from scratch.gdn2_graph import GraphedBackward

        from lethe.kernels.cute.gdn2_assemble import (
            assembled_channelwise_gdn2_backward,
        )
        from lethe.kernels.cute.gdn2_backward import _load_box_kernels_cw

        k1_cw, k2_cw = _load_box_kernels_cw()

        def adapter(q, k, v, g, bg, wg, do):  # type: ignore[no-untyped-def]
            gr = assembled_channelwise_gdn2_backward(
                q, k, v, g, bg, wg, do, use_qk_l2norm=True, k1_fn=k1_cw, k2_fn=k2_cw
            )
            return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

        def _mk(seed: int) -> tuple[torch.Tensor, ...]:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            b, t, h, d_k, d_v = 1, 512, 4, 128, 64
            q = torch.randn(b, t, h, d_k, generator=gen)
            k = torch.randn(b, t, h, d_k, generator=gen)
            v = torch.randn(b, t, h, d_v, generator=gen)
            g = -torch.rand(b, t, h, d_k, generator=gen) * 0.1
            bg = torch.rand(b, t, h, d_k, generator=gen).sigmoid()
            wg = torch.rand(b, t, h, d_v, generator=gen).sigmoid()
            do = torch.randn(b, t, h, d_v, generator=gen)
            return tuple(
                x.to(device="cuda", dtype=torch.bfloat16).contiguous()
                for x in (q, k, v, g, bg, wg, do)
            )

        x1, x2 = _mk(1), _mk(2)
        ref1 = adapter(*x1)
        ref2 = adapter(*x2)

        graphed = GraphedBackward(adapter)
        calls["n"] = 0
        out1 = graphed(*x1)  # build (warmup runs + capture) then replay
        res["build_plus_replay_l3_calls"] = calls["n"]

        calls["n"] = 0
        out2 = graphed(*x2)  # pure replay with FRESH inputs
        res["replay_l3_calls"] = calls["n"]

        def worst(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> float:
            return max(
                (
                    (x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-12)
                ).item()
                for x, y in zip(a, b, strict=True)
            )

        res["replay1_vs_eager1_scale_rel"] = worst(out1, ref1)
        res["replay2_vs_eager2_scale_rel"] = worst(out2, ref2)
        res["replay2_vs_eager1_scale_rel"] = worst(out2, ref1)  # small => STALE replay
        res["replay_fresh_inputs_correct"] = res["replay2_vs_eager2_scale_rel"] < 5e-3
        res["GO"] = bool(
            res["eager_selector_l3_calls"] >= 1 and res["replay_fresh_inputs_correct"]
        )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()
        res["GO"] = False

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
