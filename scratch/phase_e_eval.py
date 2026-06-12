"""Per-op eval: sample N candidates per op, verifier-grade, write a JSON table.

The "RL vs single-shot" ablation arm: run once with --adapter (trained
policy) and once without (base model); compare per-op exec rate /
contract rate / best reward. Scoring farms across --score-gpus like the
training runs.

Usage (box):
    CUDA_VISIBLE_DEVICES=0 uv run python scratch/phase_e_eval.py \
        --adapter phase_e_out/levelK_op/adapter_step_N --out results/phase_e_eval_rl.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--ops", default="")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--score-gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--no-speedup", action="store_true")
    ap.add_argument("--out", default="results/phase_e_eval.json")
    args = ap.parse_args()

    from phase_e_run import SCORE_TIMEOUT_S

    from flash_mamba_rl.rl.curriculum import DEFAULT_CURRICULUM
    from flash_mamba_rl.rl.hf_policy import HFPolicy, SamplingSettings
    from flash_mamba_rl.rl.parallel_scoring import ParallelScorer
    from flash_mamba_rl.rl.prompts import build_op_prompt
    from flash_mamba_rl.rl.train import _OP_ENTRY_POINTS, extract_code

    ops = tuple(args.ops.split(",")) if args.ops else DEFAULT_CURRICULUM
    gpu_ids = tuple(int(g) for g in args.score_gpus.split(","))

    policy = HFPolicy.from_pretrained(
        args.model,
        adapter_path=args.adapter,
        sampling=SamplingSettings(temperature=args.temperature, max_new_tokens=args.max_new_tokens),
    )

    table: dict[str, Any] = {
        "model": args.model,
        "adapter": args.adapter,
        "n_per_op": args.n,
        "temperature": args.temperature,
        "ops": {},
    }
    for op in ops:
        completions = policy.generate(build_op_prompt(op), args.n)
        sources = [extract_code(c, _OP_ENTRY_POINTS[op]) for c in completions]
        scorer = ParallelScorer(
            op=op,
            gpu_ids=gpu_ids,
            timeout_s=SCORE_TIMEOUT_S.get(op, 600.0),
            measure_speedup=not args.no_speedup,
        )
        scored = scorer.score_batch([s for s in sources if s is not None])
        rewards = [r["reward"] for r in scored] + [0.0] * sum(1 for s in sources if s is None)
        row = {
            "n_no_code": sum(1 for s in sources if s is None),
            "n_compiled": sum(1 for r in scored if r.get("compiled")),
            "n_contracts_passed": sum(1 for r in scored if r.get("contracts_passed")),
            "best_reward": max(rewards) if rewards else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_speedup": max((r.get("speedup") or 0.0) for r in scored) if scored else 0.0,
            "views": sorted((r.get("views_passed", 0), r.get("views_total", 0)) for r in scored),
        }
        table["ops"][op] = row
        print(f"[{op}] {row}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
