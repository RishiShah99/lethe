"""Warm-start SFT driver: LoRA SFT on the verified targets, box-side.

Pre-flight gate: every target must already be CUDA-validated
(scratch/sft_validate.py, contracts_passed on all six) — a target that
does not pass teaches nothing.

The resulting adapter seeds the Phase E curriculum:
    uv run python scratch/phase_e_run.py --resume \
        --init-adapter <ckpt-dir resolved via latest_adapter_path>

Usage (box, detached via scratch/detach.sh):
    uv run python scratch/sft_run.py --ckpt-dir sft_out [--resume]
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ops", default="", help="comma-separated subset (default: all six)")
    ap.add_argument("--variants", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--train-gpu", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--ckpt-dir", default="sft_out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    from lethe.rl.hf_policy import HFPolicy, SamplingSettings
    from lethe.rl.sft import SFTConfig, SFTTrainingLoop, build_sft_examples
    from lethe.rl.train import GRPOTrainingLoop

    adapter_path = GRPOTrainingLoop.latest_adapter_path(args.ckpt_dir) if args.resume else None
    print(f"adapter: {adapter_path or 'fresh'}", flush=True)

    # Temperature 1.0 is load-bearing: completion_log_probs tempers logits,
    # and only T=1 makes the SFT objective standard cross-entropy.
    policy = HFPolicy.from_pretrained(
        args.model,
        adapter_path=adapter_path,
        device_map={"": f"cuda:{args.train_gpu}"},
        sampling=SamplingSettings(temperature=1.0),
        gradient_checkpointing=not args.no_grad_ckpt,
    )

    ops = [o for o in args.ops.split(",") if o] or None
    variants = [v for v in args.variants.split(",") if v] or None
    examples = build_sft_examples(ops, variants)
    print(f"examples: {[f'{e.op}[{e.variant}]' for e in examples]}", flush=True)

    config = SFTConfig(
        total_steps=args.steps,
        learning_rate=args.lr,
        checkpoint_dir=args.ckpt_dir,
        save_every=args.save_every,
        seed=args.seed,
    )
    loop = SFTTrainingLoop(config, policy, examples)
    if args.resume and loop.load_trainer_state():
        print(f"resume: step {loop.step_idx}", flush=True)
    loop.run()
    print(f"DONE adapter={GRPOTrainingLoop.latest_adapter_path(args.ckpt_dir)}", flush=True)


if __name__ == "__main__":
    main()
