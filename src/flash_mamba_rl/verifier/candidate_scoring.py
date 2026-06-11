"""Source-string candidate scoring: generated code → sandboxed gates → reward.

The bridge both the Phase D bakeoff and the GRPO trainer call per generated
candidate: import the source in an isolated subprocess, find the target
op's entry point, run the op-harness gate battery, and map the outcome
through the staged reward table.

Reward boundary semantics under Triton's lazy PTX compile: a module that
fails to exec (syntax/import errors) is *not compiled* (reward 0.0); a
module that execs but whose kernel dies at first launch inside the gates
(ptxas errors surface there) has JIT-registered — it scores as a contract
failure (0.1), matching ``compile.py``'s ``__warmup__`` rationale.

CMP-02 (gradcheck) is excluded by default for generated forward kernels:
the task prompts ask for a forward op, backward ops are their own
curriculum tasks, and an autograd wrapper is orthogonal boilerplate to the
kernel-quality signal. Override via ``config["exclude_gates"]``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from typing import Any

from flash_mamba_rl.verifier.sandbox import run_in_subprocess

DEFAULT_EXCLUDE_GATES: tuple[str, ...] = ("gate_cmp_02_gradient_correctness",)

# op name -> (entry-point callable, op_harness verify driver)
_OP_VERIFIERS: dict[str, tuple[str, str]] = {
    "forward_chunked_scan": ("forward_chunked_scan", "verify_scan_op"),
}


def _trunc(obj: Any, limit: int = 300) -> str:
    text = str(obj)
    return text if len(text) <= limit else text[:limit] + "..."


def score_source_worker(source: str, config: dict[str, Any]) -> dict[str, Any]:
    """Subprocess body: import *source*, gate it, return a picklable score dict.

    stdout is swapped to stderr for the duration — the sandbox marshals the
    return value as pickled stdout and candidate prints would corrupt it.
    """
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return _score_source_body(source, config)
    finally:
        sys.stdout = real_stdout


def _score_source_body(source: str, config: dict[str, Any]) -> dict[str, Any]:
    from flash_mamba_rl.verifier import op_harness
    from flash_mamba_rl.verifier.reward import compute_reward

    op_name = str(config.get("op", "forward_chunked_scan"))
    device = str(config.get("device", "cpu"))
    exclude = set(config.get("exclude_gates", DEFAULT_EXCLUDE_GATES))
    entry_point, verify_name = _OP_VERIFIERS[op_name]

    # Real temp file: @triton.jit refuses exec'd pseudo-modules and the lazy
    # PTX compile re-reads the source during gating.
    fd, path = tempfile.mkstemp(suffix=".py", prefix="cand_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("_scored_candidate", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_scored_candidate"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return {
            "status": "exec_fail",
            "error": f"{type(exc).__name__}: {_trunc(exc)}",
            "compiled": False,
            "contracts_passed": False,
            "reward": compute_reward(
                compiled=False, contracts_passed=False, speedup_vs_handwritten=None
            ),
            "gates": {},
        }

    fn = getattr(mod, entry_point, None)
    if fn is None or not callable(fn):
        return {
            "status": "no_entrypoint",
            "error": f"module defines no callable {entry_point}",
            "compiled": False,
            "contracts_passed": False,
            "reward": compute_reward(
                compiled=False, contracts_passed=False, speedup_vs_handwritten=None
            ),
            "gates": {},
        }

    verify = getattr(op_harness, verify_name)
    results = verify(fn, device=device)
    required = [name for name in results if name not in exclude]
    contracts_passed = all(results[name].passed for name in required)
    reward = compute_reward(
        compiled=True,
        contracts_passed=contracts_passed,
        speedup_vs_handwritten=None,
    )
    return {
        "status": "scored",
        "error": "",
        "compiled": True,
        "contracts_passed": contracts_passed,
        "reward": reward,
        "gates": {
            name: {"passed": r.passed, "reason": "" if r.passed else _trunc(r.reason, 200)}
            for name, r in results.items()
        },
    }


def score_candidate_source(
    source: str,
    *,
    op: str = "forward_chunked_scan",
    device: str = "cpu",
    timeout_s: float = 300.0,
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES,
) -> dict[str, Any]:
    """Score one candidate source in an isolated subprocess (parent-side API).

    Sandbox-level failures (hangs, segfaults, CUDA context kills) normalize
    to reward 0.0 — a kernel that takes the process down cannot earn the
    contract-failure floor.
    """
    res = run_in_subprocess(
        "flash_mamba_rl.verifier.candidate_scoring",
        "score_source_worker",
        (source, {"op": op, "device": device, "exclude_gates": list(exclude_gates)}),
        timeout_s=timeout_s,
        memory_limit_mb=0,
    )
    if res.success and isinstance(res.output, dict):
        return dict(res.output)
    return {
        "status": f"sandbox_{res.error_class.name.lower()}",
        "error": _trunc(res.stderr[-400:] if res.stderr else res.error_class.name),
        "compiled": False,
        "contracts_passed": False,
        "reward": 0.0,
        "gates": {},
    }
