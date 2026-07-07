"""Task #3 GPU rigor: the formal 12-gate battery for the chunk_parallel scan_mode.

Runs ``score_candidate_config`` (the same path config-RL uses) on
forward_chunked_scan for: the serial default (control) and scan_mode=chunk_parallel
at several bench shapes. chunk_len is left None so it inherits the harness chunk_size
(which divides every gate seq_len), so the chunk-parallel path runs at every gate
shape with no divisibility failure. Prints contracts_passed / reward / speedup /
first_failed_view per config — the honest verdict on whether the long-L lever is
contract-clean, not just fast.
"""

from __future__ import annotations

import json

from lethe.kernels.autotune import KernelConfig, ShapeSpec
from lethe.verifier.candidate_scoring import score_candidate_config

_SHAPES = [
    ShapeSpec(2, 16384, 1024),
    ShapeSpec(2, 4096, 1024),
    ShapeSpec(8, 2048, 2048),
]


def _run(tag: str, cfg: KernelConfig, shape: ShapeSpec) -> dict:  # type: ignore[type-arg]
    res = score_candidate_config(
        cfg,
        op="forward_chunked_scan",
        device="cuda",
        shape=shape,
        measure_speedup=True,
        timeout_s=600.0,
    )
    out = {
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
            g for g, v in (res.get("gates") or {}).items() if isinstance(v, dict) and not v.get("passed", True)
        ],
    }
    return out


def main() -> None:
    rows = []
    for shape in _SHAPES:
        rows.append(_run("serial_default", KernelConfig(), shape))
        rows.append(_run("cp_default_cl", KernelConfig(scan_mode="chunk_parallel"), shape))
        for cl in (128, 256, 512):
            if shape.seq_len % cl == 0:
                rows.append(
                    _run(f"cp_cl{cl}", KernelConfig(scan_mode="chunk_parallel", chunk_len=cl), shape)
                )
    print("SCAN_MODE_RIGOR_JSON", json.dumps(rows))
    for r in rows:
        print(
            f"{r['tag']:16s} {r['shape']:18s} contracts={r['contracts_passed']} "
            f"reward={r['reward']} speedup={r['speedup']} "
            f"views={r['views_passed']}/{r['views_total']} "
            f"failed={r['first_failed_view'] or r['failed_gates']}"
        )


if __name__ == "__main__":
    main()
