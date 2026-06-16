"""Confirm the speed-only winners through the FULL contract-gated reward path.

The speed-only probe re-checks correctness only at the bench shape (loose
tolerance). This re-scores the headline winners through the real
score_candidate_config path (full 12-gate battery, per-grad-view for the
backward ops) to confirm they pass every contract AND keep their speedup — the
rigorous check that block_d=32 (C5/C6) and chunk_k=8 (C3) are gate-clean wins,
not just bench-shape-correct fast configs. Writes results/e2_confirm_winners.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from flash_mamba_rl.kernels.autotune import KernelConfig, ShapeSpec
from flash_mamba_rl.verifier.candidate_scoring import score_candidate_config

_SHAPE = ShapeSpec(2, 2048, 1024)

# (op, config) — each paired with its shipped default for a same-shape anchor.
# The deeper E2.d grid sweep refined the C5 forward winner from block_d=32 to
# block_d=16 / num_warps=4 (1.45-1.66x, shape-robust across width 256-4096 x
# seq 1024-16384); both are confirmed here so the launcher default change rests
# on a gate-clean reward, not a speed-only probe.
_WINNERS = [
    ("fused_block_forward", KernelConfig()),
    ("fused_block_forward", KernelConfig(block_d=32)),
    ("fused_block_forward", KernelConfig(block_d=16)),
    ("fused_block_forward", KernelConfig(block_d=16, num_warps=4)),
    ("mimo_backward", KernelConfig()),
    ("mimo_backward", KernelConfig(chunk_k=8)),
    ("fused_block_backward", KernelConfig()),
    ("fused_block_backward", KernelConfig(block_d=32)),
]


def main() -> None:
    out = []
    for op, cfg in _WINNERS:
        res = score_candidate_config(
            cfg, op=op, device="cuda", shape=_SHAPE, measure_speedup=True, timeout_s=1800
        )
        row = {
            "op": op,
            "config": cfg.searched(),
            "shape": [_SHAPE.batch, _SHAPE.seq_len, _SHAPE.width],
            "status": res.get("status"),
            "contracts": res.get("contracts_passed"),
            "reward": res.get("reward"),
            "speedup": res.get("speedup"),
        }
        out.append(row)
        print(
            f"{op} {cfg.searched() or 'DEFAULT'}: contracts={row['contracts']} "
            f"reward={row['reward']} speedup={row['speedup']}",
            flush=True,
        )
    Path("results").mkdir(exist_ok=True)
    Path("results/e2_confirm_winners.json").write_text(json.dumps(out, indent=2))
    print("WROTE results/e2_confirm_winners.json", flush=True)


if __name__ == "__main__":
    main()
