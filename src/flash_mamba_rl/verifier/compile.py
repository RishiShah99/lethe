"""Kernel compilation sandbox: compile Triton source or syntax-check on CPU-only hosts."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class ErrorClass(Enum):
    OK = auto()
    SYNTAX = auto()
    TYPE = auto()
    OOM = auto()
    TIMEOUT = auto()
    PTXAS_C7907 = auto()
    OTHER = auto()


_C7907_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"C7907", re.IGNORECASE),
    re.compile(r"internal compiler error", re.IGNORECASE),
    re.compile(r"ptxas.*error.*C\d{4}", re.IGNORECASE),
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

# Subprocess script template: try triton compile, fall back to ast.parse.
_COMPILE_SCRIPT = textwrap.dedent(
    """\
    # -*- coding: utf-8 -*-
    import sys, ast, time

    source = sys.stdin.read()

    # --- Attempt real Triton compilation ---
    try:
        import triton
        import triton.language as tl
        import tempfile, importlib.util, os
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(source)
            fname = f.name
        spec = importlib.util.spec_from_file_location("_candidate", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # triggers JIT registration
        os.unlink(fname)
        print("OK", flush=True)
        sys.exit(0)
    except ImportError:
        pass  # triton not available - fall through to AST check

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


def _scan_for_c7907(text: str) -> bool:
    """Return True if *text* contains any ptxas C7907 / ICE pattern."""
    return any(p.search(text) for p in _C7907_PATTERNS)


def _classify_stderr(stderr: str, return_code: int) -> ErrorClass:
    """Map subprocess stderr + return code to an ErrorClass."""
    if return_code == 0:
        return ErrorClass.OK
    # Order matters: ptxas C7907 before generic OTHER.
    if _scan_for_c7907(stderr):
        return ErrorClass.PTXAS_C7907
    if any(p.search(stderr) for p in _OOM_PATTERNS):
        return ErrorClass.OOM
    if any(p.search(stderr) for p in _TYPE_PATTERNS):
        return ErrorClass.TYPE
    if "SyntaxError" in stderr or "SYNTAX:" in stderr:
        return ErrorClass.SYNTAX
    return ErrorClass.OTHER


def compile_kernel(source: str, *, timeout_s: float = 30.0) -> CompileResult:
    """Compile *source* as a Triton kernel (or syntax-check on CPU-only hosts).

    Spawns a child process so that compiler crashes cannot kill the caller.
    On hosts where ``triton`` is not importable, falls back to ``ast.parse``.

    Parameters
    ----------
    source:
        Python/Triton source code string.
    timeout_s:
        Wall-clock timeout for the subprocess.

    Returns
    -------
    CompileResult
        Frozen dataclass with outcome details.
    """
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
        )
        try:
            _stdout_bytes, stderr_bytes = proc.communicate(input=source.encode(), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
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

    if rc == 0:
        return CompileResult(
            success=True,
            error_class=ErrorClass.OK,
            stderr=stderr_text,
            ptxas_c7907=ptxas_c7907,
            compile_time_s=compile_time,
        )

    error_class = _classify_stderr(stderr_text, rc)
    return CompileResult(
        success=False,
        error_class=error_class,
        stderr=stderr_text,
        ptxas_c7907=ptxas_c7907,
        compile_time_s=compile_time,
    )
