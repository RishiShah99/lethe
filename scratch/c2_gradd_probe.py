"""Probe: does the public C2 op pass the grad_D view battery on CUDA today?

Disambiguates the triton SFT target's PRC-02 grad_D failure: same kernel
bytes through the public wrapper vs the candidate module. If the public
op fails too, the gate's grad_D calibration has drifted on this stack;
if it passes, the difference is in the candidate packaging.
"""

from __future__ import annotations

from flash_mamba_rl.kernels.ops.backward_selective_scan import backward_selective_scan
from flash_mamba_rl.verifier import op_harness


def main() -> None:
    results = op_harness.verify_bwd_scan_op(
        backward_selective_scan, grad_field="grad_D", device="cuda"
    )
    for name, r in results.items():
        print(f"{name}: passed={r.passed} {'' if r.passed else r.reason[:160]}", flush=True)


if __name__ == "__main__":
    main()
