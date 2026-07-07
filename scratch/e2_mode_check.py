"""Best-tuned serial vs best-tuned chunk_parallel at the saturated shape.

The E2.c policy converged to chunk_parallel at every level (never sampling
serial) and scored 1.2-1.5x vs the shipped serial DEFAULT even at the saturated
shape b8/L2048/d2048 -- where the fixed-config rigor had chunk_parallel at 0.85x.
This scores both modes (best-tuned) through the SAME scoring path the RL used, to
settle whether a real serial>chunk_parallel crossover exists at saturation that
the policy missed, or whether tuned chunk_parallel genuinely wins everywhere.
"""

from __future__ import annotations

from lethe.kernels.autotune import ShapeSpec
from lethe.rl.config_grpo import score_config_candidate

SHAPE = ShapeSpec(8, 2048, 2048)
CONFIGS = {
    "default(serial)": "{}",
    "serial_nw4_bd128_ck4": '{"scan_mode":"serial","num_warps":4,"block_d":128,"chunk_k":4}',
    "serial_nw8_bd128_ck8": '{"scan_mode":"serial","num_warps":8,"block_d":128,"chunk_k":8}',
    "serial_nw4_bd64_ck4": '{"scan_mode":"serial","num_warps":4,"block_d":64,"chunk_k":4}',
    "serial_nw8_bd64_ck8": '{"scan_mode":"serial","num_warps":8,"block_d":64,"chunk_k":8}',
    "cp_winner": '{"scan_mode":"chunk_parallel","chunk_len":256,"num_warps":4,"block_d":128,"chunk_k":4}',
    "cp_bd64": '{"scan_mode":"chunk_parallel","chunk_len":256,"num_warps":4,"block_d":64,"chunk_k":4}',
}

for op in ("backward_selective_scan", "fused_block_backward"):
    print(f"=== {op} @ b{SHAPE.batch}/L{SHAPE.seq_len}/d{SHAPE.width} ===", flush=True)
    for name, txt in CONFIGS.items():
        r = score_config_candidate(
            txt, op=op, device="cuda", shape=SHAPE, timeout_s=900.0, measure_speedup=True
        )
        b = r.get("bench") or {}
        print(
            f"  {name:22} pass={r.get('contracts_passed')} "
            f"su={r.get('speedup')} tc={b.get('t_candidate_ms')} tb={b.get('t_baseline_ms')}",
            flush=True,
        )
print("E2_MODE_CHECK_DONE", flush=True)
