"""Smoke-test HF causal-LM inference on the box (validates the sm_100 path
the RL rollout phases depend on). Small coder model, one generation, timing.

Run:  CUDA_VISIBLE_DEVICES=0 uv run python scratch/smoke_hf_generate.py [model_id]
Writes out/smoke_hf_generate.json.
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

PROMPT = (
    "Write a Triton kernel that adds two float32 vectors. "
    "Reply with only the Python code."
)


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    print(f"loading {model_id} (bf16, single GPU)", flush=True)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).cuda()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s", flush=True)

    messages = [{"role": "user", "content": PROMPT}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].cuda()
    attention_mask = encoded["attention_mask"].cuda()

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
        )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    n_new = output.shape[1] - input_ids.shape[1]
    text = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)
    tok_per_s = n_new / gen_s

    print(f"\n--- generation ({n_new} tokens, {gen_s:.2f}s, {tok_per_s:.1f} tok/s) ---")
    print(text[:1500], flush=True)

    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "smoke_hf_generate.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "load_s": load_s,
                "gen_s": gen_s,
                "new_tokens": n_new,
                "tok_per_s": tok_per_s,
                "contains_triton_jit": "@triton.jit" in text,
            },
            indent=2,
        )
    )
    print("\nOK: hf generate smoke passed")


if __name__ == "__main__":
    main()
