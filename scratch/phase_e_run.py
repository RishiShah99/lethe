"""Phase E driver: curriculum GRPO over the six-op suite on the 8x B200 box.

Topology: the policy (Qwen2.5-Coder-32B + LoRA, bf16 ~64 GB) lives on one
B200 — launch with CUDA_VISIBLE_DEVICES=0 — and scoring sandboxes are
pinned per candidate to the remaining GPUs (absolute ids via extra_env
override, so the trainer's mask does not leak into the workers).

Modes:
  curriculum  — six-level CurriculumRunner with promotion gates (default)
  direct      — single-op GRPO for --steps (the curriculum-off ablation arm)

Resumable for spot: adapter + trainer state checkpoint every step;
curriculum_state.json names the level. --resume restores the newest
adapter across level dirs and the schedule position.

Usage (box, detached):
    CUDA_VISIBLE_DEVICES=0 uv run python scratch/phase_e_run.py \
        --ckpt-dir phase_e_out [--resume]
Poll:
    uv run python scratch/phase_e_poll.py --ckpt-dir phase_e_out
"""

from __future__ import annotations

import argparse
import os

# Per-op sandbox timeouts: a passing backward candidate pays one full
# 12-gate battery per gradient view, plus the speedup bench.
SCORE_TIMEOUT_S: dict[str, float] = {
    "forward_chunked_scan": 420.0,
    "complex_scan_rope": 420.0,
    "fused_block_forward": 600.0,
    "backward_selective_scan": 900.0,
    "mimo_backward": 1000.0,
    "fused_block_backward": 1400.0,
}


def latest_adapter_anywhere(ckpt_dir: str) -> str | None:
    """Newest committed adapter across curriculum level dirs (or the root)."""
    from flash_mamba_rl.rl.train import GRPOTrainingLoop

    candidates: list[tuple[float, str]] = []
    roots = [ckpt_dir]
    if os.path.isdir(ckpt_dir):
        roots += [
            os.path.join(ckpt_dir, d)
            for d in os.listdir(ckpt_dir)
            if d.startswith("level") and os.path.isdir(os.path.join(ckpt_dir, d))
        ]
    for root in roots:
        path = GRPOTrainingLoop.latest_adapter_path(root)
        if path is not None:
            candidates.append((os.path.getmtime(os.path.join(root, "trainer_state.pt")), path))
    return max(candidates)[1] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--mode", choices=("curriculum", "direct"), default="curriculum")
    ap.add_argument("--op", default="fused_block_backward", help="direct-mode target op")
    ap.add_argument("--steps", type=int, default=200, help="direct-mode steps / per-level cap")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--promote-threshold", type=float, default=0.35)
    ap.add_argument("--promote-window", type=int, default=8)
    ap.add_argument("--reward-shaping", choices=("none", "view_fraction"), default="view_fraction")
    ap.add_argument("--no-speedup", action="store_true", help="skip the timing stage")
    ap.add_argument("--score-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--ckpt-dir", default="phase_e_out")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from flash_mamba_rl.rl.curriculum import (
        DEFAULT_CURRICULUM,
        CurriculumConfig,
        CurriculumRunner,
    )
    from flash_mamba_rl.rl.hf_policy import HFPolicy, SamplingSettings
    from flash_mamba_rl.rl.parallel_scoring import ParallelScorer
    from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig

    gpu_ids = tuple(int(g) for g in args.score_gpus.split(","))

    def batch_scorer_factory(op: str) -> ParallelScorer:
        return ParallelScorer(
            op=op,
            gpu_ids=gpu_ids,
            timeout_s=SCORE_TIMEOUT_S.get(op, 600.0),
            reward_shaping=args.reward_shaping,
            measure_speedup=not args.no_speedup,
        )

    adapter_path = latest_adapter_anywhere(args.ckpt_dir) if args.resume else None
    print(f"adapter: {adapter_path or 'fresh'}", flush=True)
    policy = HFPolicy.from_pretrained(
        args.model,
        adapter_path=adapter_path,
        sampling=SamplingSettings(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        ),
    )

    base = TrainLoopConfig(
        n_per_prompt=args.k,
        total_steps=args.steps,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        device="cuda",
        reward_shaping=args.reward_shaping,
        measure_speedup=not args.no_speedup,
        checkpoint_dir=args.ckpt_dir,
        save_every=1,
    )

    if args.mode == "direct":
        import dataclasses

        config = dataclasses.replace(
            base,
            op=args.op,
            score_timeout_s=SCORE_TIMEOUT_S.get(args.op, 600.0),
            checkpoint_dir=os.path.join(args.ckpt_dir, f"direct_{args.op}"),
        )
        loop = GRPOTrainingLoop(config, policy, batch_scorer=batch_scorer_factory(args.op))
        if args.resume:
            resumed = loop.load_trainer_state()
            print(f"resume: {'step ' + str(loop.step_idx) if resumed else 'no state'}", flush=True)
        loop.run()
        return

    runner = CurriculumRunner(
        base_config=base,
        policy=policy,
        curriculum=CurriculumConfig(
            ops=DEFAULT_CURRICULUM,
            promote_threshold=args.promote_threshold,
            promote_window=args.promote_window,
            max_steps_per_level=args.steps,
        ),
        batch_scorer_factory=batch_scorer_factory,
    )
    if args.resume and runner.resume():
        print(f"resume: level {runner.schedule.level_idx}", flush=True)
    summary = runner.run()
    for row in summary:
        print(f"[final] {row}", flush=True)


if __name__ == "__main__":
    main()
