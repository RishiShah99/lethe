"""Re-score the last scored toy candidate in-process and dump full gate details."""

from __future__ import annotations

import json
import sys


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "grpo_toy_out/rollouts.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    row = next(r for r in reversed(rows) if r["status"] == "scored")
    source = row["source"]
    print("=== SOURCE (first 2000 chars) ===")
    print(source[:2000])

    import importlib.util
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="dbg_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("_dbg_candidate", tmp)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dbg_candidate"] = mod
    spec.loader.exec_module(mod)
    fn = mod.elementwise_silu

    import torch

    x = torch.randn(4, 64, 32, device="cuda")
    print("=== direct call ===")
    try:
        out = fn(x)
        ref = x * torch.sigmoid(x)
        print("ok, max_err:", (out - ref).abs().max().item(), "dtype:", out.dtype)
    except Exception as exc:
        print(f"RAISED {type(exc).__name__}: {exc}")

    from flash_mamba_rl.verifier.op_harness import verify_elementwise_op

    results = verify_elementwise_op(fn, device="cuda")
    for name, r in results.items():
        if not r.passed:
            print(f"--- {name}: {r.reason}")
            failures = r.details.get("failures")
            if failures:
                for f_ in failures[:3]:
                    print("    ", str(f_)[:200])


if __name__ == "__main__":
    main()
