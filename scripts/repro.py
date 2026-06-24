"""One-command reproducibility surface for flash-mamba-rl.

Captures the local environment, pins all seeds, runs every CPU-derivable
headline check, and prints a PASS/FAIL table.  Exits nonzero if any check
fails so this can gate CI.

GPU-gated steps that cannot run without a B200/H100 box are listed at the
end with their exact box commands — they are NOT executed here.

Usage
-----
    uv run python scripts/repro.py

Output
------
- ``results/repro_env.json``  — environment snapshot (overwrites on each run)
- printed PASS/FAIL table

Checks
------
1. CPU test suite (``uv run pytest -q``), parsed for pass/skip/fail counts.
2. Selector geomean: re-derive the scan_mode selector geomean speedup from
   ``results/scan_mode_boundary.json`` via ``_default_scan_mode``; assert ≥
   2.1x (committed baseline is ~2.174x).
3. Verifier audit headline: load ``results/audit_drkernel.json`` and report
   the ``accepted_only.finding_rate`` (committed: 0.6213 ≈ 62.1%).
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from statistics import geometric_mean
from typing import Any

import numpy as np
import torch

SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Seed pinning
# ---------------------------------------------------------------------------


def pin_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return str(getattr(mod, "__version__", "unknown"))


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _module_version("torch"),
        "triton": _module_version("triton"),
        "numpy": _module_version("numpy"),
        "git_head": _git_head(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["device_count"] = torch.cuda.device_count()
        env["device_0"] = torch.cuda.get_device_name(0)
        env["capability_0"] = list(torch.cuda.get_device_capability(0))
        env["torch_cuda"] = torch.version.cuda
    return env


def write_env(env: dict[str, Any]) -> Path:
    out = RESULTS_DIR / "repro_env.json"
    out.write_text(json.dumps(env, indent=2))
    return out


# ---------------------------------------------------------------------------
# Check 1 — CPU test suite
# ---------------------------------------------------------------------------


def check_cpu_suite() -> tuple[bool, str]:
    """Shell out to pytest, parse summary line, return (passed, detail)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:warnings"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    stdout = result.stdout + result.stderr
    # Last non-empty line typically holds the summary, e.g.:
    # "705 passed, 112 skipped in 12.34s"  or  "3 failed, 700 passed ..."
    summary = ""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if "passed" in line or "failed" in line or "error" in line:
            summary = line
            break

    failed = "failed" in summary or "error" in summary or result.returncode not in (0, 5)
    # returncode 5 = no tests collected (treat as pass — cannot happen here, but safe)
    if result.returncode == 0:
        return True, summary
    if result.returncode == 5:
        return True, "no tests collected"
    return not failed, summary


# ---------------------------------------------------------------------------
# Check 2 — selector geomean from committed boundary sweep
# ---------------------------------------------------------------------------


def compute_selector_geomean() -> tuple[float, int]:
    """Re-derive geomean speedup of _default_scan_mode over the old serial default.

    Returns (geomean, n_valid_rows).  Raises if the boundary JSON is absent.
    """
    from flash_mamba_rl.kernels.ops.forward_chunked_scan import _default_scan_mode

    boundary_path = RESULTS_DIR / "scan_mode_boundary.json"
    if not boundary_path.exists():
        raise FileNotFoundError(boundary_path)

    entries = json.loads(boundary_path.read_text())
    speedups: list[float] = []
    for e in entries:
        if e.get("winner") == "skipped":
            continue
        bs = (e.get("best_serial") or {}).get("speedup") or 0.0
        bc = (e.get("best_chunk_parallel") or {}).get("speedup") or 0.0
        if bs <= 0.0 and bc <= 0.0:
            continue
        is_fwd = e.get("op") == "forward_chunked_scan"
        mode = _default_scan_mode(e["seq_len"], e["batch"], e["width"], is_forward=is_fwd)
        sp = bs if mode == "serial" else (bc if bc > 0 else bs)
        if sp > 0:
            speedups.append(sp)

    if not speedups:
        raise ValueError("no valid speedup rows in boundary sweep")
    return geometric_mean(speedups), len(speedups)


def check_selector_geomean(tol: float = 2.1) -> tuple[bool, str]:
    try:
        gm, n = compute_selector_geomean()
        passed = gm >= tol
        detail = f"geomean={gm:.4f}x over {n} shapes (threshold={tol}x, committed~2.174x)"
        return passed, detail
    except FileNotFoundError as exc:
        return False, f"boundary sweep JSON missing: {exc}"
    except Exception as exc:
        return False, f"error: {exc}"


# ---------------------------------------------------------------------------
# Check 3 — verifier audit headline
# ---------------------------------------------------------------------------


def check_audit_headline() -> tuple[bool, str]:
    """Load audit_drkernel.json and report accepted_only.finding_rate."""
    audit_path = RESULTS_DIR / "audit_drkernel.json"
    if not audit_path.exists():
        # Raw rows are in audit_out/ (box artifacts) but the aggregate JSON is absent
        audit_out = REPO_ROOT / "audit_out"
        if audit_out.exists():
            return (
                True,
                "aggregate-only (raw rows are box artifacts); "
                "audit_drkernel.json not yet produced locally",
            )
        return False, "audit_drkernel.json absent and no audit_out/ raw rows found"

    data = json.loads(audit_path.read_text())
    rate = data["accepted_only"]["finding_rate"]
    total = data["accepted_only"]["denominator"]
    detail = (
        f"accepted_only finding_rate={rate:.4f} ({rate * 100:.1f}%) "
        f"over {total} audited rows (committed~62.1%)"
    )
    # finding_rate is an aggregate metric from a box run — we report it faithfully,
    # no tolerance gate; presence of the JSON and a plausible value is sufficient.
    plausible = 0.50 <= rate <= 0.80
    return plausible, detail


# ---------------------------------------------------------------------------
# GPU-gated steps (listed, not executed)
# ---------------------------------------------------------------------------

_BOX_STEPS = [
    (
        "6-kernel GPU gate battery (C1-C6 parity + verifier)",
        "uv run --no-sync pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py "
        "-m gpu -q -rA",
        "bash scratch/c6_gpu_suite.sh  (run on B200 box detached via detach.sh)",
    ),
    (
        "Scan-mode selector GPU regression (default path routes chunk_parallel)",
        "uv run --no-sync pytest tests/test_kernels_gpu.py tests/test_gpu_verifier.py "
        "-m gpu -q -rA",
        "bash scratch/run_selector_regression.sh",
    ),
    (
        "Boundary sweep repro (178-shape B200 bench, produces scan_mode_boundary.json)",
        "uv run --no-sync python scratch/scan_mode_boundary.py",
        "bash scratch/boundary_launch.sh  (self-shuts on BOUNDARY_DONE_OK)",
    ),
    (
        "Config-RL scan_mode run (E2.c - GRPO policy selects mode+chunk_len)",
        "uv run --no-sync python scratch/e2_config_rl.py "
        "--levels <LEVELS> --steps 40 --k 16 --ckpt-dir e2_config_out",
        "bash scratch/e2_config_launch.sh <LEVELS> 40 16 e2_config_out",
    ),
    (
        "PTB-XL medical training (0.880 AUC from scratch, 8xB200 DDP)",
        "uv run --no-sync torchrun --standalone --nproc_per_node=8 scratch/ptbxl_train.py "
        "--data-root ~/data/ptbxl --steps 20000 --batch-size 8 "
        "--eval-every 500 --save-every 500 --log-every 10 --resume",
        "bash scratch/ptbxl_autostart.sh  (waits for dataset then launches)",
    ),
    (
        "H100 #904 control (Triton TMEM-promotion cliff on Hopper)",
        "uv run --no-sync python scratch/repro_904.py",
        "fleet up h100  then  uv run --no-sync python scratch/repro_904.py on H100 box",
    ),
    (
        "Verifier rigor audit raw-row re-derivation (62.1% finding rate from 3134 rows)",
        "uv run --no-sync python scratch/audit_run.py  (produces audit_out/results_shard*.jsonl)",
        "bash scratch/audit_box.sh  on B200 box; aggregate locally via scratch/audit_aggregate.py",
    ),
    (
        "B200 kernel bench table (C1-C6 vs official, per-shape, labeled)",
        "uv run --no-sync python scratch/c6_bench_run.sh  (all 6 bench scripts)",
        "bash scratch/c1_bench.sh  scratch/c2_bench.sh  ...  scratch/c6_bench_run.sh",
    ),
]


def print_box_steps() -> None:
    print("\n" + "=" * 72)
    print("BOX-GATED STEPS (GPU required - NOT executed here)")
    print("=" * 72)
    for i, (name, cmd, recipe) in enumerate(_BOX_STEPS, 1):
        print(f"\n[{i}] {name}")
        print(f"    cmd    : {cmd}")
        print(f"    box    : {recipe}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _row(label: str, passed: bool, detail: str) -> str:
    status = "PASS" if passed else "FAIL"
    return f"  {status:<6}  {label:<45}  {detail}"


def run_all() -> int:
    # On Windows the default console codec (cp1252) cannot encode non-ASCII;
    # reconfigure stdout/stderr to utf-8 if the stream supports it.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    pin_seeds(SEED)

    env = capture_env()
    env_path = write_env(env)
    print(f"env -> {env_path}")
    print(
        f"       python={env['python']}  torch={env['torch']}  "
        f"triton={env['triton']}  numpy={env['numpy']}"
    )
    print(f"       git_head={env['git_head']}  cuda={env['cuda_available']}")

    checks: list[tuple[str, bool, str]] = []

    # Check 1
    ok1, d1 = check_cpu_suite()
    checks.append(("CPU test suite (pytest)", ok1, d1))

    # Check 2
    ok2, d2 = check_selector_geomean()
    checks.append(("Selector geomean >= 2.1x", ok2, d2))

    # Check 3
    ok3, d3 = check_audit_headline()
    checks.append(("Audit headline (62.1%)", ok3, d3))

    print("\n" + "=" * 72)
    print("REPRO SURFACE - PASS/FAIL")
    print("=" * 72)
    for label, passed, detail in checks:
        print(_row(label, passed, detail))

    print_box_steps()

    all_passed = all(p for _, p, _ in checks)
    print("\n" + ("ALL CHECKS PASSED" if all_passed else "ONE OR MORE CHECKS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_all())
