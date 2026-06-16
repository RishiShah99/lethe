"""E2.c driver: config-emitting GRPO over (op, shape) levels on the B200 box.

The policy emits a JSON ``KernelConfig`` for a target (op, shape); the config
is applied to the trusted in-repo kernel and scored at that shape by the
verifier (``score_config_candidate``). Reuses ``GRPOTrainingLoop`` through its
``extractor`` (fenced-JSON) and ``batch_scorer`` (``ParallelConfigScorer``)
hooks — same advantages / clipped-surrogate+KL / degenerate-skip / checkpoint
machinery as the source track.

Generation is tiny here (a few-token config, not a multi-thousand-token
kernel), so the source track's throughput + backward-truncation failures are
gone and scoring is the bottleneck — hence configs are farmed across the
scoring GPUs while the policy lives on one. Levels are weighted to off-default
and long-L shapes, where the shipped default heuristic (tuned near L=2048) is
provably not optimal and real speedup headroom exists (E2.d grid sweep:
fused_block_forward 1.5-1.7x).

Usage (box, detached, all 8 GPUs visible):
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run --no-sync python \
        scratch/e2_config_rl.py --ckpt-dir e2_config_out [--resume]
Poll:
    tail -n 30 e2_config_out/*/metrics.jsonl ; cat e2_config_out/summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

# (op, batch, seq_len, width) training levels. Off-default + long-L by design:
# the shipped heuristics are tuned near L=2048, so these are where a learned
# config can beat the default. C1/C2/C4 are measurement-tuned with no config
# headroom (HANDOFF) and are omitted from the default set.
DEFAULT_LEVELS: tuple[tuple[str, int, int, int], ...] = (
    ("fused_block_forward", 2, 4096, 1024),
    ("fused_block_forward", 2, 16384, 2048),
    ("fused_block_forward", 2, 1024, 4096),
    ("mimo_backward", 2, 4096, 1024),
    ("mimo_backward", 2, 8192, 2048),
    ("fused_block_backward", 2, 4096, 1024),
    ("fused_block_backward", 2, 8192, 2048),
)

SCORE_TIMEOUT_S: dict[str, float] = {
    "forward_chunked_scan": 300.0,
    "complex_scan_rope": 300.0,
    "fused_block_forward": 420.0,
    "backward_selective_scan": 600.0,
    "mimo_backward": 700.0,
    "fused_block_backward": 900.0,
}


def _parse_levels(spec: str) -> tuple[tuple[str, int, int, int], ...]:
    """Parse "op:BxLxW,op:BxLxW" into levels (empty -> DEFAULT_LEVELS)."""
    if not spec.strip():
        return DEFAULT_LEVELS
    out: list[tuple[str, int, int, int]] = []
    for item in spec.split(","):
        op, dims = item.split(":")
        b, length, w = (int(x) for x in dims.lower().split("x"))
        out.append((op.strip(), b, length, w))
    return tuple(out)


def _level_dir(ckpt_dir: str, op: str, b: int, length: int, w: int) -> str:
    return os.path.join(ckpt_dir, f"{op}_B{b}L{length}W{w}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--levels", default="", help='"op:BxLxW,..." (empty = curated default)')
    ap.add_argument("--steps", type=int, default=40, help="GRPO steps per level")
    ap.add_argument("--k", type=int, default=16, help="configs sampled per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="a config is tiny")
    ap.add_argument("--train-gpu", type=int, default=0)
    ap.add_argument("--score-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--ckpt-dir", default="e2_config_out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    from flash_mamba_rl.kernels.autotune import ShapeSpec
    from flash_mamba_rl.rl.config_grpo import extract_config
    from flash_mamba_rl.rl.hf_policy import HFPolicy, SamplingSettings
    from flash_mamba_rl.rl.parallel_scoring import ParallelConfigScorer
    from flash_mamba_rl.rl.prompts import build_config_prompt
    from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig

    levels = _parse_levels(args.levels)
    score_gpus = tuple(int(g) for g in args.score_gpus.split(",") if g.strip())

    adapter_path = args.init_adapter or None
    print(f"adapter: {adapter_path or 'fresh'}; levels: {len(levels)}", flush=True)
    sampling = SamplingSettings(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.k,
    )
    policy = HFPolicy.from_pretrained(
        args.model,
        adapter_path=adapter_path,
        device_map={"": f"cuda:{args.train_gpu}"},
        sampling=sampling,
        gradient_checkpointing=not args.no_grad_ckpt,
    )

    summary: list[dict[str, object]] = []
    for op, b, length, w in levels:
        shape = ShapeSpec(b, length, w)
        level_dir = _level_dir(args.ckpt_dir, op, b, length, w)
        prompt = build_config_prompt(op, shape)
        scorer = ParallelConfigScorer(
            op=op,
            gpu_ids=score_gpus,
            shape=shape,
            timeout_s=SCORE_TIMEOUT_S.get(op, 420.0),
            measure_speedup=True,
        )
        config = TrainLoopConfig(
            op=op,
            n_per_prompt=args.k,
            total_steps=args.steps,
            learning_rate=args.lr,
            kl_coef=args.kl_coef,
            device="cuda",
            measure_speedup=True,
            checkpoint_dir=level_dir,
            save_every=1,
        )
        loop = GRPOTrainingLoop(
            config,
            policy,
            prompt=prompt,
            extractor=extract_config,
            batch_scorer=scorer,
        )
        if args.resume and loop.load_trainer_state():
            print(f"[{op} B{b}L{length}W{w}] resume at step {loop.step_idx}", flush=True)
        if loop.step_idx >= args.steps:
            print(f"[{op} B{b}L{length}W{w}] already complete; skip", flush=True)
        else:
            print(f"[{op} B{b}L{length}W{w}] training {args.steps} steps", flush=True)
            loop.run()
        # Carry the cross-level adapter forward (one policy, sequential levels).
        history_best = _best_speedup(level_dir)
        summary.append(
            {"op": op, "batch": b, "seq_len": length, "width": w, "best_speedup": history_best}
        )
        _write_summary(args.ckpt_dir, summary)

    for row in summary:
        print(f"[final] {row}", flush=True)


def _best_speedup(level_dir: str) -> float:
    """Best speedup recorded across the level's rollout rows (1.0 if none)."""
    path = os.path.join(level_dir, "rollouts.jsonl")
    best = 1.0
    if not os.path.exists(path):
        return best
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            s = row.get("speedup")
            if isinstance(s, (int, float)) and s > best:
                best = float(s)
    return best


def _write_summary(ckpt_dir: str, summary: list[dict[str, object]]) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, "summary.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, os.path.join(ckpt_dir, "summary.json"))


if __name__ == "__main__":
    main()
