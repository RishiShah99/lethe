"""Phase 4 bench: the native channel-wise GDN-2 backward vs fla / cuLA Triton on B200.

Honest wall-clock of OUR native channel-wise GDN-2 training backward against the
existing open kernels, at matched ``(B, L, H)`` with d_k = d_v = 128, C = 64.

What each side measures (read before quoting a ratio):

- ``ours_native_cw`` — ``native_gdn2_backward`` in the channel-wise regime: the assembled
  tcgen05 K#1 + K#2 producing all six per-channel grads. This is the **host-orchestrated**
  form (a Python per-chunk loop launching the proven (128,64,128) GEMMs, each followed by a
  ``cuda.synchronize``); it is launch-bound, NOT a fused kernel. inc-B2/inc-C (the in-kernel
  fused reverse loop + TMEM residency) are the fusion levers that close the gap — this bench
  is the pre-fusion baseline that quantifies it. ours also re-stages the chunkwise forward
  inside the call (conservative against us), whereas the fla/cuLA backward reuses its
  forward-saved activations.
- ``fla_scalar`` — fla ``chunk_gated_delta_rule`` backward via ``autograd.grad`` (forward run
  once outside the timed region). This is the closest existing FUSED Triton kernel — its
  reverse-state ``dhu`` kernel is exactly the one our K#1 replaces. It runs the **scalar**
  gate (g per token, b = w = beta), so it does materially LESS work than our channel-wise op;
  the comparison is "our channel-wise native backward vs the fla scalar fused backward at the
  same shape", not a FLOP-matched race. The claim of this project is correctness + a native
  channel-wise backward where none existed, NOT fastest (PROJECT_PLAN §0).
- ``cula`` — best-effort: cuLA's chunked gated-delta backward if importable on the box
  (a mostly-native B200 backward). Recorded as a skip when absent, never fabricated.

CUDA-event timed, median over trials (``verifier.timing.benchmark``). Off-box / without the
kernels every GPU row is a recorded skip; ``--ref-candidate`` times the eager channel-wise
refs (harness/shape sanity on CPU, NOT a meaningful number).

    PYTHONPATH=$PWD/src:$PWD ~/cuteenv/bin/python -m flash_mamba_rl.bench.gdn2_backward_headtohead \
        --out results/gdn2_bench_cw.json
"""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

import flash_mamba_rl.kernels.cute.gdn2_backward as gdn2_native
from flash_mamba_rl.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
from flash_mamba_rl.verifier.timing import benchmark

D_K = 128
D_V = 128
CHUNK = 64


@dataclass(frozen=True)
class ShapeSpec:
    batch: int
    seq_len: int
    nheads: int

    def label(self) -> str:
        return f"B{self.batch}xL{self.seq_len}xH{self.nheads}xD{D_K}"


# d_k = d_v = 128 (crown), seq_len % 64 == 0. ours is host-orchestrated (launch-bound), so the
# headline shapes stay modest; the long-L row shows the per-chunk-loop scaling.
SHAPES = [
    ShapeSpec(1, 512, 4),
    ShapeSpec(1, 1024, 8),
    ShapeSpec(2, 2048, 8),
    ShapeSpec(2, 4096, 4),
]
QUICK_SHAPES = [ShapeSpec(1, 512, 2), ShapeSpec(1, 1024, 4)]


def _inputs(
    spec: ShapeSpec, dtype: torch.dtype, dev: torch.device, seed: int
) -> tuple[Tensor, ...]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    b, t, h = spec.batch, spec.seq_len, spec.nheads
    q = torch.randn(b, t, h, D_K, generator=gen)
    k = torch.randn(b, t, h, D_K, generator=gen)
    v = torch.randn(b, t, h, D_V, generator=gen)
    g = -torch.rand(b, t, h, D_K, generator=gen) * 0.1  # per-channel decay (key axis)
    g_head = g.mean(-1)  # scalar decay for fla (per token)
    bg = torch.rand(b, t, h, D_K, generator=gen).sigmoid()  # erase gate (key axis)
    wg = torch.rand(b, t, h, D_V, generator=gen).sigmoid()  # write gate (value axis)
    beta = torch.rand(b, t, h, generator=gen).sigmoid()  # scalar gate for fla
    do = torch.randn(b, t, h, D_V, generator=gen)

    def cast(x: Tensor) -> Tensor:
        return x.to(device=dev, dtype=dtype).contiguous()

    return tuple(cast(x) for x in (q, k, v, g, g_head, bg, wg, beta, do))


def _time(fn: Any, *, warmup: int, trials: int) -> dict[str, Any]:
    r = benchmark(fn, (), warmup=warmup, trials=trials)
    return {
        "median_ms": r.median_ms,
        "std_ms": r.std_ms,
        "min_ms": r.min_ms,
        "max_ms": r.max_ms,
        "n_trials": float(r.n_trials),
    }


def _bench_ours(
    inp: tuple[Tensor, ...], dtype: torch.dtype, ref_candidate: bool, trials: int
) -> dict[str, Any]:
    q, k, v, g, _g_head, bg, wg, _beta, do = inp
    backend = (
        assembled_channelwise_gdn2_backward if ref_candidate else gdn2_native.native_gdn2_backward
    )

    def run() -> Any:
        out = backend(q, k, v, g, bg, wg, do)
        if out is None:
            raise RuntimeError("native_gdn2_backward returned None (regime/dims/availability)")
        return out

    # ours is host-orchestrated (tens of seconds/pass): the first call JITs the cutlass kernels,
    # so one warmup post-JIT is enough and trials stay low (std is sub-1% — see results JSON).
    run()  # sanity / JIT / warm
    res: dict[str, Any] = _time(run, warmup=1, trials=trials)
    res["note"] = "host-orchestrated (launch-bound, pre-fusion); re-stages forward; channel-wise"
    return res


def _bench_fla(inp: tuple[Tensor, ...], trials: int) -> dict[str, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    q, k, v, _g, g_head, _bg, _wg, beta, do = inp
    scale = D_K**-0.5
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g_head, beta)]
    ql, kl, vl, gl, bl = leaves
    try:
        o, _ = chunk_gated_delta_rule(
            ql, kl, vl, gl, bl, scale=scale, use_qk_l2norm_in_kernel=True, output_final_state=False
        )
    except TypeError:
        o, _ = chunk_gated_delta_rule(ql, kl, vl, gl, bl, scale=scale)

    def bwd() -> Any:
        return torch.autograd.grad(o, leaves, do, retain_graph=True, allow_unused=True)

    bwd()  # warm
    res: dict[str, Any] = _time(bwd, warmup=10, trials=trials)
    res["note"] = (
        "fla scalar fused Triton backward (autograd.grad; fwd saved); scalar gate = less work"
    )
    return res


def _bench_cula(inp: tuple[Tensor, ...], trials: int) -> dict[str, Any]:
    """Best-effort cuLA chunked gated-delta backward; skip cleanly if not importable."""
    chunk_fn = None
    for modpath, name in (
        ("cula.ops.gated_delta_rule", "chunk_gated_delta_rule"),
        ("cula.ops.kda", "chunk_kda"),
        ("kda.ops", "chunk_kda"),
    ):
        try:
            mod = __import__(modpath, fromlist=[name])
            chunk_fn = getattr(mod, name)
            break
        except Exception:
            continue
    if chunk_fn is None:
        raise ImportError("cuLA chunked gated-delta backward not importable (tried cula/kda)")

    q, k, v, _g, g_head, _bg, _wg, beta, do = inp
    scale = D_K**-0.5
    leaves = [t.detach().requires_grad_(True) for t in (q, k, v, g_head, beta)]
    out = chunk_fn(*leaves, scale=scale)
    o = out[0] if isinstance(out, tuple) else out

    def bwd() -> Any:
        return torch.autograd.grad(o, leaves, do, retain_graph=True, allow_unused=True)

    bwd()
    res: dict[str, Any] = _time(bwd, warmup=10, trials=trials)
    res["note"] = "cuLA native B200 backward (autograd.grad; fwd saved)"
    return res


def _run_shape(
    spec: ShapeSpec,
    dtype: torch.dtype,
    dev: torch.device,
    ref_candidate: bool,
    trials: int,
    ours_trials: int,
    ours_max_seqlen: int,
) -> dict[str, Any]:
    inp = _inputs(spec, dtype, dev, seed=spec.batch * 100 + spec.seq_len)
    row: dict[str, Any] = {
        "shape": {
            "batch": spec.batch,
            "seq_len": spec.seq_len,
            "nheads": spec.nheads,
            "d_k": D_K,
            "d_v": D_V,
            "chunk": CHUNK,
        },
        "dtype": str(dtype).removeprefix("torch."),
        "impls": {},
        "skipped": {},
    }
    impls: dict[str, Any] = row["impls"]
    skipped: dict[str, str] = row["skipped"]

    # ours is host-orchestrated and costs tens of seconds/pass (it scales with NT = seq_len/chunk),
    # so by default time it only on the small shapes — the pre-fusion gap is identical in kind at every
    # size; the fla reference curve still runs on all shapes. Raise --ours-max-seqlen to time them all.
    if spec.seq_len <= ours_max_seqlen:
        try:
            impls["ours_native_cw"] = _bench_ours(inp, dtype, ref_candidate, ours_trials)
        except Exception:
            skipped["ours_native_cw"] = traceback.format_exc(limit=3)
    else:
        skipped["ours_native_cw"] = (
            f"seq_len {spec.seq_len} > --ours-max-seqlen {ours_max_seqlen}: host-orchestrated "
            "pre-fusion too slow to time here (gap quantified at the smaller shapes)"
        )

    for name, runner in (("fla_scalar", _bench_fla), ("cula", _bench_cula)):
        if ref_candidate:
            skipped[name] = "ref-candidate desk dry-run: GPU baselines skipped"
            continue
        try:
            impls[name] = runner(inp, trials)
        except Exception:
            skipped[name] = traceback.format_exc(limit=3)

    if "ours_native_cw" in impls and "median_ms" in impls["ours_native_cw"]:
        ours = impls["ours_native_cw"]["median_ms"]
        row["fla_over_ours"] = (
            impls["fla_scalar"]["median_ms"] / ours if "fla_scalar" in impls else None
        )
        row["cula_over_ours"] = impls["cula"]["median_ms"] / ours if "cula" in impls else None
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/gdn2_bench_cw.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--trials", type=int, default=20, help="trials for the fla/cuLA baselines")
    ap.add_argument(
        "--ours-trials", type=int, default=3, help="trials for ours (host-orchestrated; slow)"
    )
    ap.add_argument(
        "--ours-max-seqlen",
        type=int,
        default=10**9,
        help="time ours only when seq_len <= this (host-orchestrated pre-fusion is slow at long L)",
    )
    ap.add_argument(
        "--ref-candidate", action="store_true", help="desk dry-run: eager cw refs, no GPU baselines"
    )
    cli = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _orig_is_available = gdn2_native.is_available
    if dev.type == "cuda" and not cli.ref_candidate:
        gdn2_native.is_available = lambda device=None: True  # lift the box gate for this run
    try:
        dtype = torch.bfloat16 if dev.type == "cuda" and not cli.ref_candidate else torch.float32

        triton_version: str | None
        try:
            import triton

            triton_version = triton.__version__
        except ImportError:
            triton_version = None

        report: dict[str, Any] = {
            "claim": "native channel-wise GDN-2 backward wall-clock vs fla/cuLA (correctness+"
            "availability, not fastest; ours is host-orchestrated pre-fusion)",
            "env": {
                "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))
                if dev.type == "cuda"
                else None,
                "torch": torch.__version__,
                "triton": triton_version,
                "python": platform.python_version(),
            },
            "dtype": str(dtype).removeprefix("torch."),
            "runs": [],
        }

        cli.out.parent.mkdir(parents=True, exist_ok=True)
        shapes = QUICK_SHAPES if cli.quick else SHAPES
        for spec in shapes:
            print(f"[gdn2-bench] {spec.label()} {dtype} ...", flush=True)
            report["runs"].append(
                _run_shape(
                    spec,
                    dtype,
                    dev,
                    cli.ref_candidate,
                    cli.trials,
                    cli.ours_trials,
                    cli.ours_max_seqlen,
                )
            )
            # write after every shape: ours is host-orchestrated (slow) and the box is spot,
            # so a partial artifact must survive a preemption or an early stop.
            cli.out.write_text(json.dumps(report, indent=2))
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        gdn2_native.is_available = _orig_is_available

    for r in report["runs"]:
        fo = r.get("fla_over_ours")
        print(
            f"  {r['shape']} fla/ours={fo if fo is None else round(fo, 3)} "
            f"skipped={list(r['skipped'])}"
        )
    print(f"[gdn2-bench] wrote {cli.out}")


if __name__ == "__main__":
    main()
