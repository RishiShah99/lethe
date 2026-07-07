"""Readiness gate for the boundary sweep: imports + one cheap GPU scoring call.

Confirms the deployed code, the GPU stack, and the full score_config_candidate
path all work before the multi-hour sweep commits to them. The smoke scores one
serial-default config at the smallest shape (seconds), so a broken deploy or a
pruned mamba_ssm surfaces here, not 3 shapes into the sweep.
"""

from __future__ import annotations

import torch
import triton

from lethe.kernels.autotune import ShapeSpec
from lethe.rl.config_grpo import score_config_candidate

print("torch", torch.__version__, "triton", triton.__version__, flush=True)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)

r = score_config_candidate(
    "{}", op="forward_chunked_scan", device="cuda", shape=ShapeSpec(1, 512, 256), timeout_s=300.0
)
print(
    "smoke: contracts_passed=",
    r.get("contracts_passed"),
    "speedup=",
    r.get("speedup"),
    "status=",
    r.get("status"),
    flush=True,
)
print("BOUNDARY_ENV_OK", flush=True)
