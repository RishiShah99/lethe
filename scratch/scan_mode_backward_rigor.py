"""GPU rigor for the chunk_parallel scan_mode on the BACKWARD ops (C2 + C6).

Mirror of ``scratch/scan_mode_rigor.py`` (forward) for the backward kernels: runs
``score_candidate_config`` — the same path config-RL uses — on
``backward_selective_scan`` (C2) and ``fused_block_backward`` (C6) for the serial
default (control) and scan_mode=chunk_parallel at long-L shapes. chunk_len left
None so it inherits ``_auto_chunk_len`` (largest divisor ≤ cap), which divides
every gate seq_len, so the per-grad-view battery runs with no divisibility
failure. Prints contracts_passed / reward / speedup / first_failed_view per
config — the honest verdict on whether the reverse-carry reassociation is
contract-clean AND beats the serial backward at long L. Saves
results/scan_mode_backward_rigor.json.
"""

from __future__ import annotations

import json

from flash_mamba_rl.kernels.autotune import KernelConfig, ShapeSpec
from flash_mamba_rl.verifier.candidate_scoring import score_candidate_config

_OPS = ("backward_selective_scan", "fused_block_backward")
_SHAPES = [
    ShapeSpec(2, 16384, 1024),
    ShapeSpec(2, 4096, 1024),
    ShapeSpec(8, 2048, 2048),
]


def _run(op: str, tag: str, cfg: KernelConfig, shape: ShapeSpec) -> dict:  # type: ignore[type-arg]
    res = score_candidate_config(
        cfg,
        op=op,
        device="cuda",
        shape=shape,
        measure_speedup=True,
        timeout_s=900.0,
    )
    return {
        "op": op,
        "tag": tag,
        "shape": f"b{shape.batch}_l{shape.seq_len}_d{shape.width}",
        "status": res.get("status"),
        "contracts_passed": res.get("contracts_passed"),
        "reward": res.get("reward"),
        "speedup": res.get("speedup"),
        "views_passed": res.get("views_passed"),
        "views_total": res.get("views_total"),
        "first_failed_view": res.get("first_failed_view"),
        "failed_gates": [
            g
            for g, v in (res.get("gates") or {}).items()
            if isinstance(v, dict) and not v.get("passed", True)
        ],
    }


def main() -> None:
    rows = []
    for op in _OPS:
        for shape in _SHAPES:
            rows.append(_run(op, "serial_default", KernelConfig(), shape))
            rows.append(_run(op, "cp_default_cl", KernelConfig(scan_mode="chunk_parallel"), shape))
            for cl in (128, 256, 512):
                if shape.seq_len % cl == 0:
                    rows.append(
                        _run(
                            op,
                            f"cp_cl{cl}",
                            KernelConfig(scan_mode="chunk_parallel", chunk_len=cl),
                            shape,
                        )
                    )
    print("SCAN_MODE_BWD_RIGOR_JSON", json.dumps(rows))
    for r in rows:
        print(
            f"{r['op']:24s} {r['tag']:16s} {r['shape']:18s} "
            f"contracts={r['contracts_passed']} reward={r['reward']} speedup={r['speedup']} "
            f"views={r['views_passed']}/{r['views_total']} "
            f"failed={r['first_failed_view'] or r['failed_gates']}"
        )


if __name__ == "__main__":
    main()
