"""Source-string candidate scoring: generated code → sandboxed gates → reward."""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib.util
import os
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any

from lethe.verifier.sandbox import run_in_subprocess


@contextlib.contextmanager
def _hidden_project_modules() -> Iterator[None]:
    """Hide already-imported project modules from sys.modules."""
    prefixes = ("lethe", "mamba_ssm", "causal_conv1d", "selective_scan_cuda")
    hidden = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name in prefixes or name.startswith(tuple(p + "." for p in prefixes))
    }
    try:
        yield
    finally:
        sys.modules.update(hidden)


DEFAULT_EXCLUDE_GATES: tuple[str, ...] = ("gate_cmp_02_gradient_correctness",)

# Candidate sources may not reach the oracle.
_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "lethe",
        "mamba_ssm",
        "causal_conv1d",
        "selective_scan_cuda",
        "importlib",
        "sys",
        "builtins",
    }
)

# Reflection / eval entry points a wrap reaches for once direct imports screen.
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {"__import__", "__builtins__", "importlib", "builtins", "eval", "exec"}
)

# Oracle packages are blocked via the __import__ guard and _OracleImportBlocker (meta-path finder).
_GUARDED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"lethe", "mamba_ssm", "causal_conv1d", "selective_scan_cuda"}
)


def _ast_screen(source: str) -> list[str]:
    """Forbidden import / reflection constructs in *source*, parse-tree level."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    violations: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS:
                    violations.add(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS:
                violations.add(f"from {module} import ...")
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            violations.add(node.id)
    return sorted(violations)


@contextlib.contextmanager
def _import_guard() -> Iterator[None]:
    """Block fresh imports of the oracle packages at resolved-name level."""
    real_import = builtins.__import__

    def guarded(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name.split(".", 1)[0] in _GUARDED_IMPORT_ROOTS:
            raise ImportError(f"import of {name!r} blocked: candidate may not reach the oracle")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = real_import


@contextlib.contextmanager
def _oracle_cache_evicted(keys: list[str]) -> Iterator[None]:
    """Evict a precomputed set of oracle module keys from sys.modules."""
    saved = {name: sys.modules.pop(name) for name in keys if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(saved)


class _OracleImportBlocker:
    """meta-path finder that rejects the oracle packages by resolved root."""

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".", 1)[0] in _GUARDED_IMPORT_ROOTS:
            raise ImportError(f"import of {fullname!r} blocked: candidate may not reach the oracle")
        return None


@contextlib.contextmanager
def _oracle_import_blocker() -> Iterator[None]:
    """Install ``_OracleImportBlocker`` at the front of sys.meta_path."""
    finder = _OracleImportBlocker()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)


@dataclass(frozen=True)
class OpSpec:
    """Scoring wiring for one curriculum op."""

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
    """Subprocess body: import *source*, gate it, return a picklable score dict."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return _score_source_body(source, config)
    finally:
        sys.stdout = real_stdout


def _failure(status: str, error: str) -> dict[str, Any]:
    from lethe.verifier.reward import compute_reward

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
    op_name = str(config.get("op", "forward_chunked_scan"))
    device = str(config.get("device", "cpu"))
    exclude = set(config.get("exclude_gates", DEFAULT_EXCLUDE_GATES))
    fail_fast = bool(config.get("fail_fast", True))
    reward_shaping = str(config.get("reward_shaping", "none"))
    # Under fail-fast, views_passed is a prefix count; shaping needs full passes for gradient signal.
    if reward_shaping == "view_fraction":
        fail_fast = False
    spec = _OP_VERIFIERS[op_name]
    raw_shape = config.get("shape")
    bench_shape = (int(raw_shape[0]), int(raw_shape[1]), int(raw_shape[2])) if raw_shape else None

    violations = _ast_screen(source)
    if violations:
        return _failure("forbidden_import", f"forbidden constructs: {violations}")

    # @triton.jit refuses exec'd pseudo-modules; PTX compile re-reads the source during gating.
    fd, path = tempfile.mkstemp(suffix=".py", prefix="cand_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        mod_spec = importlib.util.spec_from_file_location("_scored_candidate", path)
        assert mod_spec is not None and mod_spec.loader is not None
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules["_scored_candidate"] = mod
        try:
            with _hidden_project_modules(), _import_guard(), _oracle_import_blocker():
                mod_spec.loader.exec_module(mod)
        except Exception as exc:
            return _failure("exec_fail", f"{type(exc).__name__}: {_trunc(exc)}")

        fn = getattr(mod, spec.entry_point, None)
        if fn is None or not callable(fn):
            return _failure("no_entrypoint", f"module defines no callable {spec.entry_point}")
        return _gate_and_reward(
            fn, spec, config, op_name, device, exclude, fail_fast, bench_shape=bench_shape
        )
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _gate_and_reward(
    raw_fn: Any,
    spec: OpSpec,
    config: dict[str, Any],
    op_name: str,
    device: str,
    exclude: set[str],
    fail_fast: bool,
    *,
    trusted: bool = False,
    bench_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    import lethe.kernels.ops  # noqa: F401  # force-load the speedup baseline into the evicted set
    from lethe.verifier import op_harness
    from lethe.verifier.reward import compute_reward

    reward_shaping = str(config.get("reward_shaping", "none"))

    # Entry point is invoked HERE (gates + speedup bench) after exec, once guards are down.
    if trusted:
        fn = raw_fn
    else:
        oracle_keys = [
            name for name in list(sys.modules) if name.split(".", 1)[0] in _GUARDED_IMPORT_ROOTS
        ]

        def guarded_fn(*args: Any, **kwargs: Any) -> Any:
            with _oracle_cache_evicted(oracle_keys), _oracle_import_blocker():
                return raw_fn(*args, **kwargs)

        fn = guarded_fn
    verify = getattr(op_harness, spec.verify_name)
    gates: dict[str, dict[str, Any]] = {}

    if spec.view_fields_attr is None:
        results = verify(fn, device=device)
        required = [name for name in results if name not in exclude]
        # bool(required) avoids a vacuous pass: an exclude set emptying required=[] must not unlock speedup.
        contracts_passed = bool(required) and all(results[name].passed for name in required)
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
            if required and all(results[name].passed for name in required):
                views_passed += 1
            else:
                if first_failed_view is None:
                    first_failed_view = field
                if fail_fast:
                    break
        contracts_passed = views_passed == views_total

    speedup: float | None = None
    bench: dict[str, Any] = {}
    bug_routing = False
    if (
        contracts_passed
        and bool(config.get("measure_speedup", False))
        and device.startswith("cuda")
    ):
        from lethe.verifier import op_bench

        if bench_shape is not None:
            b_sz, s_len, w = bench_shape
            bench = op_bench.measure_speedup(
                fn, op_name, device, batch=b_sz, seq_len=s_len, width=w
            )
        else:
            bench = op_bench.measure_speedup(fn, op_name, device)
        if bench.get("correct_at_bench", True):
            speedup = float(bench["speedup"])
            bug_routing = op_bench.bug_routing_active(op_name, device)
        else:
            # Wrong at the bench width (gates cap d_model at 64): a real failure, not slow, so no speedup credit.
            contracts_passed = False
            first_failed_view = "bench_shape_correctness"
            gates["bench_shape_correctness"] = {
                "passed": False,
                "reason": "candidate diverges from the hand-written baseline at the bench shape",
            }

    reward = compute_reward(
        compiled=True,
        contracts_passed=contracts_passed,
        speedup_vs_handwritten=speedup,
        bug_routing=bug_routing,
    )
    if (
        reward_shaping == "view_fraction"
        and not contracts_passed
        and views_total > 1
        and bench.get("correct_at_bench", True)
    ):
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
    shape: tuple[int, int, int] | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score one candidate source in an isolated subprocess (parent-side API)."""
    res = run_in_subprocess(
        "lethe.verifier.candidate_scoring",
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
                "shape": list(shape) if shape is not None else None,
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


def _score_config_body(
    op: str, kernel_config: dict[str, Any], opts: dict[str, Any]
) -> dict[str, Any]:
    from lethe.kernels import autotune

    device = str(opts.get("device", "cpu"))
    exclude = set(opts.get("exclude_gates", DEFAULT_EXCLUDE_GATES))
    fail_fast = bool(opts.get("fail_fast", True))
    reward_shaping = str(opts.get("reward_shaping", "none"))
    if reward_shaping == "view_fraction":
        fail_fast = False
    spec = _OP_VERIFIERS[op]

    cfg = autotune.KernelConfig(**kernel_config)
    shape_dict = opts.get("shape")
    shape = autotune.ShapeSpec(**shape_dict) if shape_dict else None
    violations = autotune.validate(op, cfg, shape=shape)
    if violations:
        return _failure("invalid_config", f"illegal config {cfg.searched()}: {violations}")

    fn = autotune.make_configured_op(op, cfg)
    bench_shape = (shape.batch, shape.seq_len, shape.width) if shape is not None else None
    gr_config = {
        "measure_speedup": bool(opts.get("measure_speedup", False)),
        "reward_shaping": reward_shaping,
    }
    return _gate_and_reward(
        fn, spec, gr_config, op, device, exclude, fail_fast, trusted=True, bench_shape=bench_shape
    )


def score_config_worker(
    op: str, kernel_config: dict[str, Any], opts: dict[str, Any]
) -> dict[str, Any]:
    """Subprocess body for config scoring: build the configured trusted op, gate it, score it."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return _score_config_body(op, kernel_config, opts)
    finally:
        sys.stdout = real_stdout


def score_candidate_config(
    kernel_config: Any,
    *,
    op: str = "forward_chunked_scan",
    device: str = "cpu",
    shape: Any = None,
    timeout_s: float = 300.0,
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES,
    fail_fast: bool = True,
    reward_shaping: str = "none",
    measure_speedup: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score one ``KernelConfig`` applied to the trusted in-repo op, in a subprocess."""
    payload_config = asdict(kernel_config)
    opts: dict[str, Any] = {
        "device": device,
        "exclude_gates": list(exclude_gates),
        "fail_fast": fail_fast,
        "reward_shaping": reward_shaping,
        "measure_speedup": measure_speedup,
    }
    if shape is not None:
        opts["shape"] = asdict(shape)
    res = run_in_subprocess(
        "lethe.verifier.candidate_scoring",
        "score_config_worker",
        (op, payload_config, opts),
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
