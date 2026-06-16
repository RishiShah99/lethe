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
every view passes. Shaping disables fail-fast: a prefix count would
zero the signal for improving later views whenever an early one fails.

Candidate sources that import the project package or the official Mamba
kernels, or reach for the dynamic-import / builtins-reflection machinery a
wrap needs, are blocked by three layers (see ``_ast_screen`` / ``_import_guard``
/ ``_OracleImportBlocker``): an AST screen rejects them before exec
(``forbidden_import``, reward 0.0); a ``builtins.__import__`` guard, a
``sys.meta_path`` finder, and sys.modules eviction of the oracle block it across
every import pathway, armed both during module exec AND around each
candidate-entry-point call (the gates and the speedup bench invoke the entry point
*after* exec, so a wrap deferred — or an import callable captured — into the body
must be caught there too; the eviction forces a cache miss so even a captured-real
``import_module`` re-resolves into the finder). Wrapping the reference or the
hand-written ops would otherwise pass every gate without writing a kernel — the
0.5-reward fixed point the policy must not be able to reach.

CMP-02 (gradcheck) is excluded by default for generated kernels: forward
tasks ask for a forward op, backward tasks ARE the gradient computation
(an autograd wrapper over them is orthogonal boilerplate), and a raw
Triton kernel carries no grad_fn for the gate to differentiate through.
Override via ``config["exclude_gates"]``.
"""

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

from flash_mamba_rl.verifier.sandbox import run_in_subprocess


@contextlib.contextmanager
def _hidden_project_modules() -> Iterator[None]:
    """Hide already-imported project modules from sys.modules.

    The worker imports op_harness (and transitively the references)
    before the candidate's module body runs; without this, a candidate
    could fish the oracle out of sys.modules with a string-built key and
    wrap it. Local references held by the worker stay valid; the entries
    are restored afterwards so in-process callers (tests) see identical
    module objects.
    """
    prefixes = ("flash_mamba_rl", "mamba_ssm", "causal_conv1d", "selective_scan_cuda")
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

# Candidate sources may not reach the oracle. The realistic RL reward-hack is
# wrapping the reference (or the official kernels) so every gate passes without
# a kernel — the 0.5 fixed point. Three layers stop it. (1) An AST screen rejects
# the package imports, the dynamic-import machinery (importlib/__import__), and
# the builtins reflection a wrap needs: a name built at runtime
# (``"flash"+"_mamba_rl"``, ``__builtins__["__im"+"port__"]``) still leaves an
# ``__import__``/``__builtins__``/``importlib`` reference in the parse tree,
# which substring scanning could not see — and substring scanning also
# false-matched ``eval(`` inside ``retrieval(``, so it is gone. (2) A
# ``builtins.__import__`` guard active during candidate exec blocks the oracle
# packages by resolved root name, catching the direct-import path at the point
# it resolves. (3) An ``_OracleImportBlocker`` on ``sys.meta_path`` blocks the
# same roots through EVERY import pathway — ``importlib.import_module`` and
# ``_gcd_import`` bypass ``builtins.__import__``, so the guard alone could not
# see them; the finder is also why importlib is no longer guard-blocked (doing
# so scored every real Triton kernel as a non-compile, since Triton pulls in
# importlib transitively at jit time — the finder closes the importlib-as-gadget
# path without the false positive). At exec, project modules are additionally
# hidden from sys.modules (see ``_hidden_project_modules``) so a candidate import
# re-consults the finder rather than hitting a cached oracle. The guards span two
# windows, because the gates and the speedup bench invoke the entry point AFTER
# exec (a wrap deferred into the body, or an import callable captured at exec, is
# what (1)-(3) would otherwise miss): module exec uses all three layers; each
# candidate-entry-point call pairs the finder with sys.modules eviction of a
# bounded, pre-scanned oracle key set (4) (see ``_oracle_cache_evicted``).
# Eviction is load-bearing — a candidate that saved the *real* ``import_module``
# before the guards installed bypasses any live-attribute patch, but the forced
# cache miss still routes its import through the finder; the key set is scanned
# once so the cost stays out of the timed region (a full per-call scan dragged
# parity kernels to ~0.7x). Object-graph gadget chains (gc-walking to the
# already-imported reference object directly, no import at all) stay possible by
# construction — a documented arms-race boundary, not a contract.
_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "flash_mamba_rl",
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

# Resolved-root oracle packages, blocked at every import pathway: the
# ``builtins.__import__`` guard for the direct path and ``_OracleImportBlocker``
# (a meta-path finder) for ``importlib.import_module`` / ``_gcd_import``.
# importlib is deliberately NOT a member — a real Triton candidate pulls it in
# transitively at jit time, so guard-blocking it scored every kernel as a
# non-compile; the finder closes the importlib-as-gadget path instead. sys and
# builtins are always cached, so a fresh re-import is a no-op (the AST screen
# handles their source-level use).
_GUARDED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {"flash_mamba_rl", "mamba_ssm", "causal_conv1d", "selective_scan_cuda"}
)


def _ast_screen(source: str) -> list[str]:
    """Forbidden import / reflection constructs in *source*, parse-tree level.

    A syntactically invalid source returns no violations — exec surfaces the
    SyntaxError and it scores as a non-compile (0.0), not a screen rejection.
    """
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
    """Block fresh imports of the oracle packages at resolved-name level.

    Belt to the AST screen's suspenders: a name assembled at runtime can slip
    the parse tree, but the import it ultimately drives passes through this
    patched ``builtins.__import__`` and is rejected by its resolved root.
    Process-isolated in the scoring worker, so patching builtins is safe; the
    verifier's own project imports happen outside this window.
    """
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
    """Evict a precomputed set of oracle module keys from sys.modules.

    Forces any oracle import during the window — including one driven by a real
    ``import_module`` / ``__import__`` reference the candidate *captured at exec*,
    before any live-attribute patch was installed — to miss the sys.modules cache
    and re-resolve through ``sys.meta_path``, where ``_OracleImportBlocker``
    rejects it. Late-binding patches alone could not stop an early-bound
    reference; the cache miss is what closes it.

    ``keys`` is scanned once per scoring (see ``_gate_and_reward``), so the
    per-call cost is O(oracle modules) — not the O(sys.modules) full scan that
    landed inside the speedup timing and dragged parity kernels to ~0.7x.
    op_harness holds direct references to the reference functions, so evicting
    their sys.modules entries does not break the gates' own oracle calls — only a
    fresh candidate import re-resolves and meets the finder.
    """
    saved = {name: sys.modules.pop(name) for name in keys if name in sys.modules}
    try:
        yield
    finally:
        sys.modules.update(saved)


class _OracleImportBlocker:
    """meta-path finder that rejects the oracle packages by resolved root.

    The ``_import_guard`` only sees imports routed through
    ``builtins.__import__``; ``importlib.import_module`` and
    ``importlib._bootstrap._gcd_import`` bypass it. Every import pathway,
    those included, consults ``sys.meta_path``, so a finder that raises for the
    oracle roots is the pathway-complete block — and the reason importlib need
    not be guard-blocked (which broke real Triton candidates). With
    ``_hidden_project_modules`` evicting the oracle from sys.modules, a candidate
    import re-consults the finders and is rejected regardless of how it is
    spelled; unrelated stdlib / third-party imports return None and fall through
    to the real finders.
    """

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split(".", 1)[0] in _GUARDED_IMPORT_ROOTS:
            raise ImportError(f"import of {fullname!r} blocked: candidate may not reach the oracle")
        return None


@contextlib.contextmanager
def _oracle_import_blocker() -> Iterator[None]:
    """Install ``_OracleImportBlocker`` at the front of sys.meta_path.

    Used both around module exec and around each candidate-entry-point call.
    """
    finder = _OracleImportBlocker()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.meta_path.remove(finder)


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
    op_name = str(config.get("op", "forward_chunked_scan"))
    device = str(config.get("device", "cpu"))
    exclude = set(config.get("exclude_gates", DEFAULT_EXCLUDE_GATES))
    fail_fast = bool(config.get("fail_fast", True))
    reward_shaping = str(config.get("reward_shaping", "none"))
    # Under fail-fast, views_passed is the longest passing prefix, not the
    # pass count — shaping over a prefix would zero the gradient signal for
    # improving later views whenever an early one fails. Shaping therefore
    # always runs the full battery.
    if reward_shaping == "view_fraction":
        fail_fast = False
    spec = _OP_VERIFIERS[op_name]

    violations = _ast_screen(source)
    if violations:
        return _failure("forbidden_import", f"forbidden constructs: {violations}")

    # Real temp file: @triton.jit refuses exec'd pseudo-modules and the lazy
    # PTX compile re-reads the source during gating. The file is removed by
    # the caller's lifecycle only after scoring completes (the subprocess
    # exits; in-process callers rely on the finally below).
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
        return _gate_and_reward(fn, spec, config, op_name, device, exclude, fail_fast)
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
    import flash_mamba_rl.kernels.ops  # noqa: F401  # force-load the speedup baseline into the evicted set
    from flash_mamba_rl.verifier import op_harness
    from flash_mamba_rl.verifier.reward import compute_reward

    reward_shaping = str(config.get("reward_shaping", "none"))

    # The entry point is invoked HERE (gates + speedup bench), after exec — the
    # exec-time guards are down. A candidate that deferred the oracle wrap into
    # its body, or captured a real import_module/__import__ at exec, would reach
    # the reference at call time (the 0.5 fixed point). Wrap every candidate call
    # in a cache-eviction (forces a miss) + the meta-path finder, so the import
    # re-resolves into the finder regardless of how the callable was bound. The
    # oracle key set is scanned once, now that op_harness (references) and the
    # hand-written ops are loaded — per-call cost is O(oracle modules), keeping
    # the eviction out of the timed region. The config-scoring path passes
    # trusted=True: its callable IS an in-repo kernel (no generated source), so
    # the eviction must NOT run — it would hide the very modules the trusted op
    # imports, breaking the call it is meant to protect.
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
        from flash_mamba_rl.verifier import op_bench

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
            # Value-incorrect at the bench width the gates don't reach (they cap
            # d_model at 64): a real correctness failure, not "not faster" — demote
            # to the contract-fail reward so no speedup credit is paid.
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


def _score_config_body(
    op: str, kernel_config: dict[str, Any], opts: dict[str, Any]
) -> dict[str, Any]:
    from flash_mamba_rl.kernels import autotune

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
    """Subprocess body for config scoring: build the configured trusted op, gate it, score it.

    stdout is swapped to stderr for the duration — the sandbox marshals the
    return value as pickled stdout and a kernel printf would corrupt it (the
    ``score_source_worker`` convention).
    """
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
    """Score one ``KernelConfig`` applied to the trusted in-repo op, in a subprocess.

    Unlike :func:`score_candidate_source`, there is no generated source — the
    config is bound into the already-correct kernel via
    :func:`flash_mamba_rl.kernels.autotune.make_configured_op` — so the
    AST/oracle screens do not run (nothing to game). The subprocess still
    isolates OOM / ptxas-ICE / timeout, which normalise to reward 0.0; an
    out-of-grid or shape-incompatible config scores the ``invalid_config`` 0.0.
    ``shape`` (a ``ShapeSpec``) sets the speedup bench shape — this is the lever
    the autotuner is rewarded on, since the shipped default is optimal only near
    the training shape; ``None`` uses op_bench's default shape. Only
    ``(batch, seq_len, width)`` reach the bench — ``ShapeSpec.n_state`` is NOT
    consumed here (each op's state dim is fixed by its harness aux builder), so a
    non-None ``n_state`` is silently ignored by scoring.
    """
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
        "flash_mamba_rl.verifier.candidate_scoring",
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
