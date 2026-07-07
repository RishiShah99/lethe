"""#48 box gate — graph the WHOLE channel-wise GDN-2 backward, grade + time on B200.

Wraps ``native_gdn2_backward`` (cw regime) in :class:`scratch.gdn2_graph.GraphedBackward`
and, per crown shape, checks:
  * capture succeeded (no "operation not permitted during stream capture" / SIGSEGV),
  * graphed output == oracle (the crown credential, <= 5e-3) and == eager (graph is transparent),
  * determinism (two replays bit-identical),
  * timing: eager backward vs graph replay -> the host-orchestration speedup this lever buys.

Runs ``use_qk_l2norm`` BOTH True (the real contract — includes the torch.autograd.grad L2-norm
VJP, the one unprobed capture risk) and False (isolates that risk). GO requires the True path to
capture + match the oracle. If True fails but False passes, the L2-norm autograd is the blocker
and the fix is to capture pre-L2-norm + apply the VJP eagerly (#48 fallback).

Run on box:
  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_graph_box.py --out results/gdn2_graph_box.json
"""

from __future__ import annotations

import argparse
import json
import time

import torch

D_K = 128
D_V = 128
CHUNK = 64
GRID = [(1, 64, 1), (1, 128, 2), (2, 128, 1), (1, 256, 1)]
ATOL_BF16 = 5e-3
GRAD_FIELDS = ("grad_q", "grad_k", "grad_v", "grad_g", "grad_b", "grad_w")


def _inputs(b: int, t: int, h: int, dtype: torch.dtype, dev: torch.device, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()
    do = torch.randn(b, t, h, D_V, generator=gen)

    def cast(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=dev, dtype=dtype).contiguous()

    return tuple(cast(x) for x in (q, k, v, g, bg, wg, do))


def _scale_rel(a: torch.Tensor, e: torch.Tensor) -> float:
    return ((a.float() - e.float()).abs().max() / e.float().abs().max().clamp_min(1e-12)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--atol", type=float, default=ATOL_BF16)
    ap.add_argument("--eager-trials", type=int, default=3)
    ap.add_argument("--graph-trials", type=int, default=50)
    args = ap.parse_args()

    verdict: dict[str, object] = {"gate": "gdn2_graph_box", "atol": args.atol}
    if not torch.cuda.is_available():
        verdict["GO"] = False
        verdict["reason"] = "no CUDA (desk)"
        print("GO=False")
        print(json.dumps(verdict, indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(verdict, f, indent=2)
        return

    from scratch.gdn2_graph import GraphedBackward

    from lethe.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
    from lethe.kernels.cute.gdn2_backward import _load_box_kernels_cw
    from lethe.kernels.references.gdn_backward import reference_gdn2_backward

    # Capture the cw assembly DIRECTLY, not native_gdn2_backward: the dispatcher's
    # _is_scalar_reducible does torch.allclose (a device->host sync) which is illegal during
    # capture. The dispatch decision (cw vs scalar) is made once here, outside the graph.
    k1_cw, k2_cw = _load_box_kernels_cw()

    dev = torch.device("cuda")
    dtype = torch.bfloat16

    def make_adapter(use_l2: bool):
        def _cw_tuple(q, k, v, g, b, w, do):
            gr = assembled_channelwise_gdn2_backward(
                q, k, v, g, b, w, do, use_qk_l2norm=use_l2, k1_fn=k1_cw, k2_fn=k2_cw
            )
            return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

        return _cw_tuple

    overall_go = True
    by_mode: dict[str, object] = {}
    for use_l2 in (True, False):
        mode = f"l2norm={use_l2}"
        adapter = make_adapter(use_l2)
        shapes: list[dict[str, object]] = []
        mode_ok = True
        worst = 0.0
        for b, t, h in GRID:
            inp = _inputs(b, t, h, dtype, dev, seed=b * 100 + t)
            row: dict[str, object] = {"shape": [b, t, h]}
            try:
                eager = adapter(*inp)
                graphed = GraphedBackward(adapter)
                g1 = graphed(*inp)
                g2 = graphed(*inp)
                row["capture_ok"] = True
            except Exception as e:
                row["capture_ok"] = False
                row["capture_err"] = repr(e)[:400]
                shapes.append(row)
                mode_ok = False
                continue

            rel_eager = max(_scale_rel(g1[i], eager[i]) for i in range(len(eager)))
            det = all(bool(torch.equal(g1[i], g2[i])) for i in range(len(g1)))
            row["rel_vs_eager"] = rel_eager
            row["deterministic"] = det

            # Crown credential: graphed native cw == token-serial oracle. The oracle applies the
            # q/k L2-norm, so this is the apples-to-apples grade only for use_l2=True.
            if use_l2:
                orc = reference_gdn2_backward(*(x.float() for x in inp))
                orc_t = tuple(getattr(orc, f) for f in GRAD_FIELDS)
                rel_orc = max(_scale_rel(g1[i], orc_t[i]) for i in range(len(g1)))
                row["rel_vs_oracle"] = rel_orc
                shape_worst = rel_orc
            else:
                shape_worst = rel_eager
            worst = max(worst, shape_worst)

            passed = bool(row["capture_ok"]) and det and shape_worst <= args.atol
            row["passed"] = passed
            mode_ok = mode_ok and passed
            shapes.append(row)
            print(
                f"  {mode} ({b},{t},{h}) worst={shape_worst:.3e} det={det} cap={row['capture_ok']}"
            )

        by_mode[mode] = {"results": shapes, "worst": worst, "ok": mode_ok}
        if use_l2:
            overall_go = overall_go and mode_ok

    # Timing on the real path (use_qk_l2norm=True), per shape: eager vs graph replay.
    timing: list[dict[str, object]] = []
    adapter = make_adapter(True)
    for b, t, h in GRID:
        inp = _inputs(b, t, h, dtype, dev, seed=b * 100 + t)
        try:
            graphed = GraphedBackward(adapter)
            graphed(*inp)  # build + stage inputs into static buffers
        except Exception as e:
            timing.append({"shape": [b, t, h], "error": repr(e)[:200]})
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.eager_trials):
            adapter(*inp)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1e3 / args.eager_trials

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.graph_trials):
            graphed.replay_only()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t0) * 1e3 / args.graph_trials
        timing.append(
            {
                "shape": [b, t, h],
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "speedup": eager_ms / graph_ms if graph_ms > 0 else 0.0,
            }
        )
        print(
            f"  TIME ({b},{t},{h}) eager={eager_ms:.3f}ms graph={graph_ms:.3f}ms "
            f"speedup={eager_ms / graph_ms:.1f}x"
        )

    verdict["by_mode"] = by_mode
    verdict["timing"] = timing
    verdict["GO"] = overall_go
    print(f"\nGO={overall_go}")
    print(json.dumps(verdict, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
