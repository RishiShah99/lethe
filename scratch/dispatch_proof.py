"""#18: prove the medical model trains on our Triton kernels (not eager).

Builds a real ``Mamba3ECGClassifier``, spies the C5 forward launcher, the C6
backward launchers, and the eager fallback, then runs one forward + backward at
cuda+{fp32,bf16}. The claim "the PTB-XL model dispatches to the C5/C6 Triton
path" is proven iff the Triton launcher fires once per block and the eager path
fires zero times — recorded with the compiled kernel's ptxas resource_meta
(which only exists if a Triton kernel actually compiled on-device).

    CUDA_VISIBLE_DEVICES=0 uv run python scratch/dispatch_proof.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from flash_mamba_rl.kernels.ops import (
    _triton_chunk_parallel_fused_bwd,
    _triton_fused_block,
    _triton_fused_block_bwd,
)
from flash_mamba_rl.kernels.ops import fused_block_forward as fbf_mod
from flash_mamba_rl.kernels.ops.fused_block_forward import triton_fused_block_resource_meta
from flash_mamba_rl.medical.model import Mamba3Config, Mamba3ECGClassifier

OUT = Path("results/dispatch_proof.json")


class _Counter:
    def __init__(self, target: Any, attr: str) -> None:
        self.target = target
        self.attr = attr
        self.real = getattr(target, attr)
        self.count = 0

    def __enter__(self) -> _Counter:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return self.real(*args, **kwargs)

        setattr(self.target, self.attr, wrapper)
        return self

    def __exit__(self, *exc: Any) -> None:
        setattr(self.target, self.attr, self.real)


def _run(dtype: torch.dtype) -> dict[str, Any]:
    cfg = Mamba3Config(d_model=256, n_layers=3, d_state=16, conv_kernel_size=4, chunk_size=64)
    model = Mamba3ECGClassifier(cfg).cuda().to(dtype)
    t = 256  # T % chunk_size == 0
    ecg = torch.randn(2, cfg.n_leads, t, device="cuda", dtype=dtype)

    with (
        _Counter(_triton_fused_block, "launch_fused_block_forward") as fwd_triton,
        _Counter(fbf_mod, "_fused_eager") as fwd_eager,
        _Counter(_triton_fused_block_bwd, "launch_fused_block_backward") as bwd_serial,
        _Counter(
            _triton_chunk_parallel_fused_bwd, "launch_fused_block_backward_chunk_parallel"
        ) as bwd_cp,
    ):
        logits = model(ecg)
        loss = logits.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        fwd_t, fwd_e = fwd_triton.count, fwd_eager.count
        bwd_s, bwd_c = bwd_serial.count, bwd_cp.count

    dispatched = fwd_t == cfg.n_layers and fwd_e == 0 and (bwd_s + bwd_c) == cfg.n_layers
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "n_layers": cfg.n_layers,
        "forward_triton_launches": fwd_t,
        "forward_eager_launches": fwd_e,
        "backward_triton_serial_launches": bwd_s,
        "backward_triton_chunk_parallel_launches": bwd_c,
        "dispatched_to_triton_c5_c6": dispatched,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    report: dict[str, Any] = {
        "claim": "#18 — Mamba3ECGClassifier forward+backward dispatches to the C5/C6 Triton path",
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "torch": torch.__version__,
        },
        "runs": [_run(torch.float32), _run(torch.bfloat16)],
        "resource_meta": triton_fused_block_resource_meta(),
    }
    report["all_dispatched"] = all(r["dispatched_to_triton_c5_c6"] for r in report["runs"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}; all_dispatched={report['all_dispatched']}")


if __name__ == "__main__":
    main()
