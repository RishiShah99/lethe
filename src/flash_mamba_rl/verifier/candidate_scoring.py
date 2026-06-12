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

Backward ops are graded one gradient view at a time (the op-harness
per-view drivers); the contract verdict is ALL views passing. Views run
in field order with fail-fast: the first failing view ends the battery —
bad candidates (the common case) pay for one view, near-passing ones pay
for all. Optional ``view_fraction`` shaping redistributes the
contract-fail band as 0.1 + 0.35 * views_passed/views_total, strictly
below the 0.5 full-pass floor — the speedup term still pays only after
every view passes.

Candidate sources that mention the project package, the official Mamba
kernels, or dynamic-import machinery are rejected before exec
(``forbidden_import``, reward 0.0): wrapping the reference or the
hand-written ops would otherwise pass every gate without writing a
kernel — a 0.5-reward fixed point the policy must not be able to reach.

CMP-02 (gradcheck) is excluded by default for generated kernels: forward
tasks ask for a forward op, backward tasks ARE the gradient computation
(an autograd wrapper over them is orthogonal boilerplate), and a raw
Triton kernel carries no grad_fn for the gate to differentiate through.
Override via ``config["exclude_gates"]``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

from flash_mamba_rl.verifier.sandbox import run_in_subprocess

DEFAULT_EXCLUDE_GATES: tuple[str, ...] = ("gate_cmp_02_gradient_correctness",)

# Substring screen, not a security boundary: it stops the realistic RL
# reward hacks (importing the reference/official kernels, dynamic-import
# or exec evasion) for sources the policy emits. "selective_scan_cuda" is
# the official extension module; the bare entry-point name
# "backward_selective_scan" does not contain it.
FORBIDDEN_SOURCE_TOKENS: tuple[str, ...] = (
    "flash_mamba_rl",
    "mamba_ssm",
    "causal_conv1d",
    "selective_scan_cuda",
    "importlib",
    "__import__",
    "exec(",
    "eval(",
)


@dataclass(frozen=True)
class OpSpec:
    """Scoring wiring for one curriculum op.

    ``view_fields_attr`` names the op_harness tuple of gradient views for
    backward ops (the per-view driver is then called once per field with
    ``grad_field=``); None means a single-view forward driver.
    """

    entry_point: str
    verify_name: str
    view_fields_attr: str | None = None


_OP_VERIFIERS: dict[str, OpSpec] = {
    "forward_chunked_scan": OpSpec("forward_chunked_scan", "verify_scan_op"),
    "elementwise_silu": OpSpec("elementwise_silu", "verify_elementwise_op"),
    "complex_scan_rope": OpSpec("complex_scan_rope", "verify_rope_op"),
    "fused_block_forward": OpSpec("fused_block_forward", "verify_fused_block_op"),
    "backward_selective_scan": OpSpec(
        "backward_selective_scan", "verify_bwd_scan_op", "BWD_GRAD_FIELDS"
    ),
    "mimo_backward": OpSpec("mimo_backward", "verify_mimo_bwd_op", "MIMO_BWD_GRAD_FIELDS"),
    "fused_block_backward": OpSpec(
        "fused_block_backward", "verify_fused_bwd_op", "FUSED_BWD_GRAD_FIELDS"
    ),
}


def scoreable_ops() -> tuple[str, ...]:
    return tuple(_OP_VERIFIERS)


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


def _failure(status: str, error: str) -> dict[str, Any]:
    from flash_mamba_rl.verifier.reward import compute_reward

    return {
        "status": status,
        "error": error,
        "compiled": False,
        "contracts_passed": False,
        "reward": compute_reward(
            compiled=False, contracts_passed=False, speedup_vs_handwritten=None
        ),
        "gates": {},
        "views_passed": 0,
        "views_total": 0,
        "first_failed_view": None,
    }


def _score_source_body(source: str, config: dict[str, Any]) -> dict[str, Any]:
    from flash_mamba_rl.verifier import op_harness
    from flash_mamba_rl.verifier.reward import compute_reward

    op_name = str(config.get("op", "forward_chunked_scan"))
    device = str(config.get("device", "cpu"))
    exclude = set(config.get("exclude_gates", DEFAULT_EXCLUDE_GATES))
    fail_fast = bool(config.get("fail_fast", True))
    reward_shaping = str(config.get("reward_shaping", "none"))
    spec = _OP_VERIFIERS[op_name]

    hits = [tok for tok in FORBIDDEN_SOURCE_TOKENS if tok in source]
    if hits:
        return _failure("forbidden_import", f"forbidden tokens: {hits}")

    # Real temp file: @triton.jit refuses exec'd pseudo-modules and the lazy
    # PTX compile re-reads the source during gating.
    fd, path = tempfile.mkstemp(suffix=".py", prefix="cand_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    mod_spec = importlib.util.spec_from_file_location("_scored_candidate", path)
    assert mod_spec is not None and mod_spec.loader is not None
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["_scored_candidate"] = mod
    try:
        mod_spec.loader.exec_module(mod)
    except Exception as exc:
        return _failure("exec_fail", f"{type(exc).__name__}: {_trunc(exc)}")

    fn = getattr(mod, spec.entry_point, None)
    if fn is None or not callable(fn):
        return _failure("no_entrypoint", f"module defines no callable {spec.entry_point}")

    verify = getattr(op_harness, spec.verify_name)
    gates: dict[str, dict[str, Any]] = {}

    if spec.view_fields_attr is None:
        results = verify(fn, device=device)
        required = [name for name in results if name not in exclude]
        contracts_passed = all(results[name].passed for name in required)
        gates = {
            name: {"passed": r.passed, "reason": "" if r.passed else _trunc(r.reason, 200)}
            for name, r in results.items()
        }
        views_total = 1
        views_passed = 1 if contracts_passed else 0
        first_failed_view = None if contracts_passed else op_name
    else:
        fields: tuple[str, ...] = getattr(op_harness, spec.view_fields_attr)
        views_total = len(fields)
        views_passed = 0
        first_failed_view = None
        for field in fields:
            results = verify(fn, grad_field=field, device=device)
            for name, r in results.items():
                gates[f"{field}/{name}"] = {
                    "passed": r.passed,
                    "reason": "" if r.passed else _trunc(r.reason, 200),
                }
            required = [name for name in results if name not in exclude]
            if all(results[name].passed for name in required):
                views_passed += 1
            else:
                first_failed_view = field
                if fail_fast:
                    break
        contracts_passed = views_passed == views_total

    speedup: float | None = None
    bench: dict[str, float] = {}
    bug_routing = False
    if (
        contracts_passed
        and bool(config.get("measure_speedup", False))
        and device.startswith("cuda")
    ):
        from flash_mamba_rl.verifier import op_bench

        bench = op_bench.measure_speedup(fn, op_name, device)
        speedup = bench["speedup"]
        bug_routing = op_bench.bug_routing_active(op_name, device)

    reward = compute_reward(
        compiled=True,
        contracts_passed=contracts_passed,
        speedup_vs_handwritten=speedup,
        bug_routing=bug_routing,
    )
    if reward_shaping == "view_fraction" and not contracts_passed and views_total > 1:
        reward = 0.1 + 0.35 * (views_passed / views_total)
    return {
        "status": "scored",
        "error": "",
        "compiled": True,
        "contracts_passed": contracts_passed,
        "reward": reward,
        "gates": gates,
        "views_passed": views_passed,
        "views_total": views_total,
        "first_failed_view": first_failed_view,
        "speedup": speedup,
        "bench": bench,
        "bug_routing": bug_routing,
    }


def score_candidate_source(
    source: str,
    *,
    op: str = "forward_chunked_scan",
    device: str = "cpu",
    timeout_s: float = 300.0,
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES,
    fail_fast: bool = True,
    reward_shaping: str = "none",
    measure_speedup: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score one candidate source in an isolated subprocess (parent-side API).

    Sandbox-level failures (hangs, segfaults, CUDA context kills) normalize
    to reward 0.0 — a kernel that takes the process down cannot earn the
    contract-failure floor. ``extra_env`` pins the sandbox (e.g.
    ``CUDA_VISIBLE_DEVICES``) for multi-GPU scoring workers.
    """
    res = run_in_subprocess(
        "flash_mamba_rl.verifier.candidate_scoring",
        "score_source_worker",
        (
            source,
            {
                "op": op,
                "device": device,
                "exclude_gates": list(exclude_gates),
                "fail_fast": fail_fast,
                "reward_shaping": reward_shaping,
                "measure_speedup": measure_speedup,
            },
        ),
        timeout_s=timeout_s,
        memory_limit_mb=0,
        extra_env=extra_env,
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
        "views_passed": 0,
        "views_total": 0,
        "first_failed_view": None,
    }
