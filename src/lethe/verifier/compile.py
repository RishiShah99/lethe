"""Kernel compilation sandbox: compile Triton source or syntax-check on CPU-only hosts."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill *proc* and its descendants (e.g. ptxas / CUDA grandchildren)."""
    # sys.platform (not os.name) so mypy narrows the POSIX-only calls away on Windows.
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()
    proc.wait()


class ErrorClass(Enum):
    OK = auto()
    SYNTAX = auto()
    TYPE = auto()
    OOM = auto()
    TIMEOUT = auto()
    PTXAS_C7907 = auto()
    TMEM_BUDGET = auto()
    OTHER = auto()


_C7907_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"C7907", re.IGNORECASE),
    re.compile(r"internal compiler error", re.IGNORECASE),
    re.compile(r"ptxas.*error.*C\d{4}", re.IGNORECASE),
]

# TMEM overflow (mamba#904): OutOfResources Required=544 > Hardware limit=512; C7907 pre-3.7.
_TMEM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"out of resource:\s*tensor memory", re.IGNORECASE),
    # Anchored on the OutOfResources prefix plus a structured tail so it can't match incidentally.
    re.compile(
        r"out of resource.*tensor memory.*Required:\s*\d+.*Hardware limit:\s*\d+",
        re.IGNORECASE,
    ),
]

_OOM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"ResourceExhausted", re.IGNORECASE),
]

_TYPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"TypeError", re.IGNORECASE),
    re.compile(r"type error", re.IGNORECASE),
    re.compile(r"IncompatibleType", re.IGNORECASE),
]

# Subprocess template: exec + optional __warmup__ under triton, else ast.parse on CPU-only hosts.
_COMPILE_SCRIPT = textwrap.dedent(
    """\
    # -*- coding: utf-8 -*-
    import sys, ast

    source = sys.stdin.read()

    try:
        import triton  # noqa: F401
        _have_triton = True
    except ImportError:
        _have_triton = False

    if _have_triton:
        # Real path: exec the module (JIT registration), then run __warmup__
        # if defined so PTX actually compiles. Candidate errors, including
        # the candidate's own ImportError, must propagate, not fall through
        # to the AST check.
        import tempfile, importlib.util, os
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(source)
            fname = f.name
        spec = importlib.util.spec_from_file_location("_candidate", fname)
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so dataclasses/pickle/inspect paths that
        # re-resolve the module by name work, and keep the source file
        # alive through __warmup__ so launch-time tracebacks (the
        # C7907/TMEM evidence) reference real source lines.
        sys.modules["_candidate"] = mod
        try:
            spec.loader.exec_module(mod)
            warmup = getattr(mod, "__warmup__", None)
            if warmup is not None:
                warmup()
        finally:
            os.unlink(fname)
        print("OK", flush=True)
        sys.exit(0)

    # --- CPU-only fallback: pure syntax check via ast.parse ---
    try:
        ast.parse(source)
        print("OK", flush=True)
        sys.exit(0)
    except SyntaxError as e:
        print(f"SYNTAX:{e}", file=sys.stderr, flush=True)
        sys.exit(1)
    """
)


@dataclass(frozen=True)
class CompileResult:
    success: bool
    error_class: ErrorClass
    stderr: str
    ptxas_c7907: bool
    compile_time_s: float
    tmem_budget: bool = False

    @property
    def blackwell_failure(self) -> bool:
        """True if either TMEM-overflow signature fired: ptxas C7907 (older) or OutOfResources (triton>=3.7)."""
        return self.ptxas_c7907 or self.tmem_budget


def _scan_for_c7907(text: str) -> bool:
    """Return True if *text* contains any ptxas C7907 / ICE pattern."""
    return any(p.search(text) for p in _C7907_PATTERNS)


def _scan_for_tmem(text: str) -> bool:
    """Return True if *text* contains the TMEM-budget OutOfResources pattern."""
    return any(p.search(text) for p in _TMEM_PATTERNS)


def _classify_stderr(stderr: str, return_code: int) -> ErrorClass:
    """Map subprocess stderr + return code to an ErrorClass."""
    if return_code == 0:
        return ErrorClass.OK
    # Order matters: the Blackwell-failure signatures outrank generic OOM, which outranks OTHER.
    if _scan_for_c7907(stderr):
        return ErrorClass.PTXAS_C7907
    if _scan_for_tmem(stderr):
        return ErrorClass.TMEM_BUDGET
    if any(p.search(stderr) for p in _OOM_PATTERNS):
        return ErrorClass.OOM
    if any(p.search(stderr) for p in _TYPE_PATTERNS):
        return ErrorClass.TYPE
    if "SyntaxError" in stderr or "SYNTAX:" in stderr:
        return ErrorClass.SYNTAX
    return ErrorClass.OTHER


def compile_kernel(source: str, *, timeout_s: float = 30.0) -> CompileResult:
    """Compile *source* as a Triton kernel (or syntax-check on CPU-only hosts)."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="fmrl_compile_",
        delete=False,
        encoding="utf-8",
    ) as script_file:
        script_file.write(_COMPILE_SCRIPT)
        script_path = Path(script_file.name)

    t0 = time.perf_counter()
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _stdout_bytes, stderr_bytes = proc.communicate(input=source.encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            compile_time = time.perf_counter() - t0
            return CompileResult(
                success=False,
                error_class=ErrorClass.TIMEOUT,
                stderr="",
                ptxas_c7907=False,
                compile_time_s=compile_time,
            )
    finally:
        script_path.unlink(missing_ok=True)

    compile_time = time.perf_counter() - t0
    stderr_text = stderr_bytes.decode(errors="replace")
    rc = proc.returncode

    ptxas_c7907 = _scan_for_c7907(stderr_text)
    tmem_budget = _scan_for_tmem(stderr_text)

    if rc == 0:
        return CompileResult(
            success=True,
            error_class=ErrorClass.OK,
            stderr=stderr_text,
            ptxas_c7907=ptxas_c7907,
            compile_time_s=compile_time,
            tmem_budget=tmem_budget,
        )

    error_class = _classify_stderr(stderr_text, rc)
    return CompileResult(
        success=False,
        error_class=error_class,
        stderr=stderr_text,
        ptxas_c7907=ptxas_c7907,
        compile_time_s=compile_time,
        tmem_budget=tmem_budget,
    )
