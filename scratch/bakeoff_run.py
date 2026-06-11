"""Phase D base-model bakeoff: single-shot Triton generation, verifier-graded.

For one HF coder model: sample N completions of the C1 task prompt,
extract the final ```python block, score each through the sandboxed
candidate-scoring bridge on GPU, write one JSONL row per candidate plus a
summary row. Resumable per model via --out (existing rows count toward N).

Usage (box):
    CUDA_VISIBLE_DEVICES=0,1 uv run python scratch/bakeoff_run.py \
        --model Qwen/Qwen2.5-Coder-7B-Instruct --n 16 \
        --out bakeoff_out/qwen25_7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str | None:
    blocks = _CODE_BLOCK.findall(completion)
    for block in reversed(blocks):
        if "def forward_chunked_scan" in block:
            return block
    return blocks[-1] if blocks else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--score-timeout", type=float, default=420.0)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n_existing = 0
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            n_existing = sum(1 for line in f if '"summary"' not in line)
    n_needed = args.n - n_existing
    print(f"{args.model}: {n_existing} existing, generating {max(0, n_needed)}", flush=True)

    rows: list[dict[str, Any]] = []
    if n_needed > 0:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from flash_mamba_rl.rl.prompts import build_op_prompt

        prompt = build_op_prompt("forward_chunked_scan")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        messages = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        t0 = time.time()
        completions: list[str] = []
        batch = 4
        remaining = n_needed
        while remaining > 0:
            k = min(batch, remaining)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=k,
                    pad_token_id=tok.eos_token_id,
                )
            for seq in out:
                completions.append(tok.decode(seq[input_ids.shape[1] :], skip_special_tokens=True))
            remaining -= k
            print(f"generated {len(completions)}/{n_needed}", flush=True)
        gen_time = time.time() - t0

        del model
        torch.cuda.empty_cache()

        from flash_mamba_rl.verifier.candidate_scoring import score_candidate_source

        with open(args.out, "a", encoding="utf-8") as out_f:
            for i, completion in enumerate(completions):
                source = extract_code(completion)
                if source is None:
                    row: dict[str, Any] = {
                        "model": args.model,
                        "idx": n_existing + i,
                        "status": "no_code_block",
                        "reward": 0.0,
                        "compiled": False,
                        "contracts_passed": False,
                        "gates": {},
                    }
                else:
                    t1 = time.time()
                    score = score_candidate_source(
                        source,
                        device=args.device,
                        timeout_s=args.score_timeout,
                    )
                    row = {
                        "model": args.model,
                        "idx": n_existing + i,
                        "source": source,
                        "uses_triton": "@triton.jit" in source,
                        "score_s": round(time.time() - t1, 1),
                        **score,
                    }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                print(
                    f"[{n_existing + i}] {row['status']} reward={row['reward']}",
                    flush=True,
                )
        print(f"generation {gen_time:.0f}s", flush=True)

    with open(args.out, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if '"summary"' not in line]
    n = len(rows)
    summary = {
        "summary": True,
        "model": args.model,
        "n": n,
        "no_code_block": sum(1 for r in rows if r["status"] == "no_code_block"),
        "uses_triton": sum(1 for r in rows if r.get("uses_triton")),
        "compiled": sum(1 for r in rows if r.get("compiled")),
        "contracts_passed": sum(1 for r in rows if r.get("contracts_passed")),
        "mean_reward": round(sum(r["reward"] for r in rows) / max(1, n), 4),
        "status_counts": {
            s: sum(1 for r in rows if r["status"] == s) for s in {r["status"] for r in rows}
        },
    }
    with open(args.out, "a", encoding="utf-8") as out_f:
        out_f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
