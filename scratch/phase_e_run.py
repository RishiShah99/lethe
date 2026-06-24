"""Phase E driver: curriculum GRPO over the six-op suite on the 8x B200 box.

Topology (launch WITHOUT a CUDA_VISIBLE_DEVICES mask so all 8 B200s are
visible — the driver places everything by absolute id):
  - trainer policy (Qwen2.5-Coder-32B + LoRA, optimizer) on --train-gpu
  - generation replicas on --gen-gpus (a data-parallel GenerationPool;
    the K candidates split across them, sampled concurrently). Generation
    is the 32B bottleneck, so most GPUs go here.
  - scoring sandboxes pinned per candidate to --score-gpus (absolute ids
    via extra_env, so the trainer's device choice does not leak in).
Default 8-GPU split: train 0 · generate 1,2,3,4 · score 5,6,7. With
--gen-gpus "" the trainer policy generates inline (single-GPU fallback).

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
    """Newest committed adapter across curriculum level dirs.

    Curriculum mode only — direct mode resolves its adapter from its own
    ``direct_<op>`` dir so the curriculum-off arm can never resume from
    (or contaminate itself with) curriculum-trained weights.
    """
    from flash_mamba_rl.rl.train import GRPOTrainingLoop

    candidates: list[tuple[float, str]] = []
    if os.path.isdir(ckpt_dir):
        for d in os.listdir(ckpt_dir):
            root = os.path.join(ckpt_dir, d)
            if d.startswith("level") and os.path.isdir(root):
                path = GRPOTrainingLoop.latest_adapter_path(root)
                if path is not None:
                    candidates.append(
                        (os.path.getmtime(os.path.join(root, "trainer_state.pt")), path)
                    )
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
    # Forward triton targets run ~3.1-3.4k tokens (complex_scan_rope,
    # fused_block_forward), so 2048 truncated them and they could only promote on
    # shorter correct generations; 4096 clears the longest forward target with
    # margin (#15).
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    # Backward ops emit far longer kernels — the fused_block_backward triton
    # target is 8754 tokens, which left only ~5% headroom under the old 9216 cap
    # (a slightly longer generation truncated -> unparseable -> the hardest level
    # never promoted, the #15 bug). 12288 gives ~40% margin. Curriculum mode
    # raises the cap to this for the backward levels only.
    ap.add_argument("--max-new-tokens-bwd", type=int, default=12288)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--promote-rate", type=float, default=0.5, help="contract-pass rate gate")
    ap.add_argument("--promote-window", type=int, default=8)
    ap.add_argument("--reward-shaping", choices=("none", "view_fraction"), default="view_fraction")
    ap.add_argument("--no-speedup", action="store_true", help="skip the timing stage")
    ap.add_argument("--train-gpu", type=int, default=0, help="absolute id for the trainer policy")
    # Default OFF: a threaded data-parallel pool is GIL-bound for HF
    # generate, and at K=8 decode is weight-bandwidth-bound so batched
    # single-GPU generation is already near-optimal (smoke: pool 487 s/step
    # vs single-GPU 271 s). Set e.g. --gen-gpus 1,2,3,4 only if K is large
    # enough to make a replica's batch compute-bound. "none" = inline.
    ap.add_argument(
        "--gen-gpus",
        default="none",
        help="data-parallel generation replica GPUs; 'none' = trainer generates inline",
    )
    ap.add_argument("--score-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--ckpt-dir", default="phase_e_out")
    ap.add_argument("--resume", action="store_true")
    # Warm start (the cold-start fix): seed the run from an SFT adapter dir
    # (e.g. GRPOTrainingLoop.latest_adapter_path("sft_out")). A run's own
    # resume state always wins so spot preemption keeps RL progress.
    ap.add_argument("--init-adapter", default="")
    # 32B differentiable log-prob pass OOMs a single B200 without it
    # (measured 177 GiB); off only for small bring-up models.
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    from flash_mamba_rl.rl.curriculum import (
        DEFAULT_CURRICULUM,
        CurriculumConfig,
        CurriculumRunner,
    )
    from flash_mamba_rl.rl.gen_pool import GenerationPool
    from flash_mamba_rl.rl.hf_policy import HFPolicy, SamplingSettings
    from flash_mamba_rl.rl.parallel_scoring import ParallelScorer
    from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig

    gpu_ids = tuple(int(g) for g in args.score_gpus.split(","))
    gen_gpus = [
        int(g) for g in args.gen_gpus.split(",") if g.strip() and g.strip().lower() != "none"
    ]

    def batch_scorer_factory(op: str) -> ParallelScorer:
        return ParallelScorer(
            op=op,
            gpu_ids=gpu_ids,
            timeout_s=SCORE_TIMEOUT_S.get(op, 600.0),
            reward_shaping=args.reward_shaping,
            measure_speedup=not args.no_speedup,
        )

    direct_dir = os.path.join(args.ckpt_dir, f"direct_{args.op}")
    if args.resume:
        adapter_path = (
            GRPOTrainingLoop.latest_adapter_path(direct_dir)
            if args.mode == "direct"
            else latest_adapter_anywhere(args.ckpt_dir)
        )
    else:
        adapter_path = None
    if adapter_path is None and args.init_adapter:
        adapter_path = args.init_adapter
    print(f"adapter: {adapter_path or 'fresh'}", flush=True)
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

    gen_pool = None
    if gen_gpus:
        print(f"generation pool on cuda {gen_gpus}", flush=True)
        gen_pool = GenerationPool.from_pretrained(
            args.model,
            gen_gpus,
            sampling=sampling,
            lora=True,
        )
        gen_pool.refresh_from(policy)

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
            checkpoint_dir=direct_dir,
        )
        loop = GRPOTrainingLoop(
            config, policy, batch_scorer=batch_scorer_factory(args.op), gen_pool=gen_pool
        )
        if args.resume:
            resumed = loop.load_trainer_state()
            print(f"resume: {'step ' + str(loop.step_idx) if resumed else 'no state'}", flush=True)
        loop.run()
        return

    import dataclasses as _dc

    backward_ops = {"backward_selective_scan", "mimo_backward", "fused_block_backward"}

    def on_level_start(op: str) -> None:
        mnt = args.max_new_tokens_bwd if op in backward_ops else args.max_new_tokens
        policy.sampling = _dc.replace(policy.sampling, max_new_tokens=mnt)
        print(f"[curriculum] level {op}: max_new_tokens={mnt}", flush=True)

    runner = CurriculumRunner(
        base_config=base,
        policy=policy,
        curriculum=CurriculumConfig(
            ops=DEFAULT_CURRICULUM,
            promote_contract_rate=args.promote_rate,
            promote_window=args.promote_window,
            max_steps_per_level=args.steps,
        ),
        batch_scorer_factory=batch_scorer_factory,
        gen_pool=gen_pool,
        on_level_start=on_level_start,
    )
    if args.resume and runner.resume():
        print(f"resume: level {runner.schedule.level_idx}", flush=True)
    summary = runner.run()
    for row in summary:
        print(f"[final] {row}", flush=True)


if __name__ == "__main__":
    main()
