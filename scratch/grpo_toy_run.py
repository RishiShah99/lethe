"""GRPO toy validation: elementwise SiLU task — reward must move (Phase D gate).

Bring-up model is a small Qwen2.5-Coder sibling (7B cached on the box from
the bakeoff); the real Phase E run uses the 32B. Resumable for spot: the
adapter + trainer state checkpoint every step.

Usage (box, detached):
    CUDA_VISIBLE_DEVICES=0 uv run python scratch/grpo_toy_run.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct --steps 40 \
        --ckpt-dir grpo_toy_out [--resume]

Poll:  tail -5 grpo_toy_out/metrics.jsonl
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--ckpt-dir", default="grpo_toy_out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--score-timeout", type=float, default=300.0)
    args = ap.parse_args()

    from flash_mamba_rl.rl.hf_policy import HFPolicy, SamplingSettings
    from flash_mamba_rl.rl.train import GRPOTrainingLoop, TrainLoopConfig

    adapter_path = GRPOTrainingLoop.latest_adapter_path(args.ckpt_dir) if args.resume else None
    policy = HFPolicy.from_pretrained(
        args.model,
        adapter_path=adapter_path,
        sampling=SamplingSettings(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        ),
    )
    config = TrainLoopConfig(
        op="elementwise_silu",
        n_per_prompt=args.k,
        total_steps=args.steps,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        device="cuda",
        score_timeout_s=args.score_timeout,
        checkpoint_dir=args.ckpt_dir,
        save_every=1,
    )
    loop = GRPOTrainingLoop(config, policy)
    if args.resume:
        resumed = loop.load_trainer_state()
        print(f"resume: {'restored step ' + str(loop.step_idx) if resumed else 'no state found'}")
    loop.run()


if __name__ == "__main__":
    main()
