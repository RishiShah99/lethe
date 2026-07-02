"""#52 — headline re-bench: GRAPHED cw GDN-2 backward vs fla at the bench shapes (L512-4096).

#48 graphed the whole channel-wise backward (one ``CUDA-graph`` replay collapses host
dispatch + the per-chunk Python loop + torch glue). This times the graph replay against
fla's scalar fused Triton backward at the headtohead shapes — the "almost-fla parity"
number the speed plan targets. It also times EAGER ours (the #47 stream-threaded + lever-B
batched path) so the graph lever's own speedup is explicit, and reports ours_graph / fla.

Same caveats as src bench gdn2_backward_headtohead: ours is channel-wise (more work than
fla's scalar gate), ours re-stages the forward inside the captured region (conservative).
Correctness is re-gated by gdn2_graph_box / gdn2_integration_box_cw; this is wall-clock only.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/gdn2_graph_bench.py --out results/gdn2_graph_bench.json
"""

from __future__ import annotations

import argparse
import json
import traceback
from functools import partial
from typing import Any

import torch

D_K = 128
D_V = 128
CHUNK = 64
# (batch, seq_len, nheads) — the gdn2_backward_headtohead headline shapes.
SHAPES = [(1, 512, 4), (1, 1024, 8), (2, 2048, 8), (2, 4096, 4)]


def _inputs(b: int, t: int, h: int, dtype: torch.dtype, dev: torch.device, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1
    g_head = g.mean(-1)
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()
    beta = torch.rand(b, t, h, generator=gen).sigmoid()
    do = torch.randn(b, t, h, D_V, generator=gen)

    def cast(x: torch.Tensor) -> torch.Tensor:
        return x.to(device=dev, dtype=dtype).contiguous()

    return tuple(cast(x) for x in (q, k, v, g, g_head, bg, wg, beta, do))


def _env_fingerprint() -> dict[str, Any]:
    # Burst-1 lesson: the failing capture matrix was confounded by an unrecorded env
    # (cu13 .so swap + non-pristine restore). Every bench JSON now pins the DSL runtime
    # bytes + the cuda-* package set so an env delta can never hide again.
    import hashlib
    import importlib.metadata as md
    from pathlib import Path

    fp: dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    pkgs: dict[str, str] = {}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if any(s in name for s in ("cutlass", "cuda", "triton", "nvidia")):
            pkgs[name] = dist.version
    fp["packages"] = dict(sorted(pkgs.items()))
    try:
        import nvidia_cutlass_dsl

        so = Path(nvidia_cutlass_dsl.__file__).parent / "lib" / "libcute_dsl_runtime.so"
        fp["libcute_dsl_runtime_md5"] = hashlib.md5(so.read_bytes()).hexdigest()
    except Exception:
        fp["libcute_dsl_runtime_md5"] = None
    return fp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/gdn2_graph_bench.json")
    ap.add_argument("--trials", type=int, default=30, help="trials for graph + fla")
    ap.add_argument("--eager-trials", type=int, default=3, help="trials for eager ours (slow)")
    ap.add_argument(
        "--shapes", type=str, default="", help="comma list BxLxH (e.g. 1x512x4); default = all"
    )
    args = ap.parse_args()
    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",")]

    report: dict[str, Any] = {"bench": "gdn2_graph_bench", "runs": []}
    if not torch.cuda.is_available():
        report["error"] = "no CUDA (desk)"
        print(json.dumps(report, indent=2))
        return

    from scratch.gdn2_graph import GraphedBackward

    from flash_mamba_rl.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
    from flash_mamba_rl.kernels.cute.gdn2_backward import _load_box_kernels_cw
    from flash_mamba_rl.verifier.timing import benchmark

    report["env"] = _env_fingerprint()
    k1_cw, k2_cw = _load_box_kernels_cw()
    dev = torch.device("cuda")
    dtype = torch.bfloat16

    def adapter(q, k, v, g, bg, wg, do):
        gr = assembled_channelwise_gdn2_backward(
            q, k, v, g, bg, wg, do, use_qk_l2norm=True, k1_fn=k1_cw, k2_fn=k2_cw
        )
        return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

    def adapter_closed(q, k, v, g, bg, wg, do):
        # de-glue Level 1a: the chunk-local closed-form stage-B fed by K#1's dh.
        gr = assembled_channelwise_gdn2_backward(
            q,
            k,
            v,
            g,
            bg,
            wg,
            do,
            use_qk_l2norm=True,
            k1_fn=k1_cw,
            k2_fn=k2_cw,
            stage_b_closed=True,
        )
        return (gr.grad_q, gr.grad_k, gr.grad_v, gr.grad_g, gr.grad_b, gr.grad_w)

    def bench_fla(inp, trials):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        q, k, v, _g, g_head, _bg, _wg, beta, do = inp
        scale = D_K**-0.5
        leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g_head, beta)]
        try:
            o, _ = chunk_gated_delta_rule(
                *leaves, scale=scale, use_qk_l2norm_in_kernel=True, output_final_state=False
            )
        except TypeError:
            o, _ = chunk_gated_delta_rule(*leaves, scale=scale)

        def bwd():
            return torch.autograd.grad(o, leaves, do, retain_graph=True, allow_unused=True)

        bwd()
        return benchmark(bwd, (), warmup=10, trials=trials).median_ms

    for b, t, h in shapes:
        label = f"B{b}xL{t}xH{h}"
        print(f"[graph-bench] {label} ...", flush=True)
        inp = _inputs(b, t, h, dtype, dev, seed=b * 100 + t)
        cw = inp[:4] + inp[5:7] + (inp[8],)  # (q,k,v,g, bg,wg, do) for the cw assembly
        row: dict[str, Any] = {"shape": [b, t, h]}

        # eager ours (assembly direct; same path the graph captures)
        try:
            eager_run = partial(adapter, *cw)
            eager_run()  # warm/JIT
            row["eager_ms"] = benchmark(eager_run, (), warmup=1, trials=args.eager_trials).median_ms
        except Exception:
            row["eager_err"] = traceback.format_exc()

        # graphed ours (build once, time pure replay)
        try:
            graphed = GraphedBackward(adapter)
            graphed(*cw)  # build + stage inputs into static buffers
            row["graph_ms"] = benchmark(
                graphed.replay_only, (), warmup=10, trials=args.trials
            ).median_ms
        except Exception:
            row["graph_err"] = traceback.format_exc()

        # de-glue Level 1a: closed stage-B — parity vs the default path, then eager+graph time
        try:
            base_out = adapter(*cw)
            closed_out = adapter_closed(*cw)
            row["closed_parity_scale_rel"] = max(
                (
                    (a.float() - c.float()).abs().max() / a.float().abs().max().clamp_min(1e-12)
                ).item()
                for a, c in zip(base_out, closed_out, strict=True)
            )
            closed_run = partial(adapter_closed, *cw)
            row["closed_eager_ms"] = benchmark(
                closed_run, (), warmup=1, trials=args.eager_trials
            ).median_ms
            graphed_c = GraphedBackward(adapter_closed)
            graphed_c(*cw)
            row["closed_graph_ms"] = benchmark(
                graphed_c.replay_only, (), warmup=10, trials=args.trials
            ).median_ms
        except Exception:
            row["closed_err"] = traceback.format_exc()

        try:
            row["fla_ms"] = bench_fla(inp, args.trials)
        except Exception:
            row["fla_err"] = traceback.format_exc()

        if "graph_ms" in row and "fla_ms" in row:
            row["graph_over_fla"] = row["graph_ms"] / row["fla_ms"]
        if "graph_ms" in row and "eager_ms" in row:
            row["eager_over_graph"] = row["eager_ms"] / row["graph_ms"]
        if "closed_graph_ms" in row and "fla_ms" in row:
            row["closed_graph_over_fla"] = row["closed_graph_ms"] / row["fla_ms"]
        if "closed_graph_ms" in row and "graph_ms" in row:
            row["graph_over_closed_graph"] = row["graph_ms"] / row["closed_graph_ms"]
        report["runs"].append(row)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        ef = row.get("eager_over_graph")
        gf = row.get("graph_over_fla")
        print(
            f"  {label} eager={row.get('eager_ms')} graph={row.get('graph_ms')} "
            f"fla={row.get('fla_ms')} graph/fla={gf} eager/graph={ef}",
            flush=True,
        )
        torch.cuda.empty_cache()

    print(f"\nwrote {args.out}")
    print(json.dumps(report["runs"], indent=2))


if __name__ == "__main__":
    main()
