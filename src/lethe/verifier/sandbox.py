"""Subprocess sandbox: run a kernel callable in an isolated child process."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from lethe.verifier.compile import ErrorClass, kill_process_tree

# Return codes we care about
_RC_TIMEOUT = -998  # synthetic sentinel

# POSIX segfault return code
_RC_SEGFAULT_POSIX = -11

# Windows access violation exit code (as signed 32-bit: 0xC0000005 → -1073741819)
_RC_ACCESS_VIOLATION_WIN = -1073741819
# Also check unsigned representation that some shells report
_RC_ACCESS_VIOLATION_WIN_U = 0xC0000005

_CUDA_IMA_PATTERNS = [
    "an illegal memory access",
    "CUDA error: device-side assert triggered",
    "cudaErrorIllegalAddress",
    "CUDA_ERROR_ILLEGAL_ADDRESS",
]

_OOM_PATTERNS = [
    "out of memory",
    "CUDA out of memory",
    "ResourceExhausted",
    "std::bad_alloc",
    "MemoryError",
]

# Child worker script template.
_WORKER_SCRIPT = textwrap.dedent(
    """\
    import os, sys

    # The result travels to the parent over fd 1. Duplicate it to a private fd
    # and repoint fd 1 (+ Python stdout) at stderr BEFORE anything else runs:
    # the stdin read, the task unpickle, and the heavy imports below can each
    # trigger a C-level write to fd 1 (a CUDA/native banner, an os.write(1))
    # that the Python-level stdout swap alone cannot intercept, and one stray
    # byte corrupts the torch.save payload. The candidate keeps a working
    # stdout (now stderr); only the result travels the private fd.
    sys.stdout.flush()
    result_fd = os.dup(1)
    os.dup2(2, 1)

    import importlib, pickle, platform

    # --- Memory limit (POSIX only) ---
    limit_mb = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit_mb > 0:
        if platform.system() != "Windows":
            try:
                import resource
                limit_bytes = limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            except Exception as e:
                print(f"[sandbox] WARNING: could not set memory limit: {e}", file=sys.stderr)
        else:
            print(
                "[sandbox] WARNING: memory_limit_mb not enforced on Windows "
                f"(requested {limit_mb} MB)",
                file=sys.stderr,
            )

    # --- Read task ---
    raw = sys.stdin.buffer.read()
    module_path, callable_name, inputs = pickle.loads(raw)

    # Import the module containing the callable
    if module_path.endswith(".py"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_sandbox_mod", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)

    fn = getattr(mod, callable_name)
    result = fn(*inputs)

    # Write result to the private channel (a pipe may take partial writes).
    # torch.save (not pickle.dumps) so the parent can reload with
    # weights_only=True; pin protocol 2 to match torch.load's default and avoid
    # its protocol-5 weights_only warning.
    import io as _io
    import torch as _torch
    _rbuf = _io.BytesIO()
    _torch.save(result, _rbuf, pickle_protocol=2)
    payload = memoryview(_rbuf.getvalue())
    while payload:
        payload = payload[os.write(result_fd, payload):]
    os.close(result_fd)
    """
)


@dataclass(frozen=True)
class SubprocessResult:
    success: bool
    output: Any
    error_class: ErrorClass
    stderr: str
    exit_code: int


def _classify_subprocess_failure(stderr: str, rc: int) -> ErrorClass:
    """Map return code + stderr to an ErrorClass."""
    if rc == _RC_TIMEOUT:
        return ErrorClass.TIMEOUT
    if rc in (_RC_SEGFAULT_POSIX, _RC_ACCESS_VIOLATION_WIN, _RC_ACCESS_VIOLATION_WIN_U):
        return ErrorClass.OTHER  # segfault; "OTHER" since ErrorClass has no SEGFAULT variant

    stderr_lower = stderr.lower()
    if any(p.lower() in stderr_lower for p in _OOM_PATTERNS):
        return ErrorClass.OOM
    if any(p.lower() in stderr_lower for p in _CUDA_IMA_PATTERNS):
        return ErrorClass.OTHER  # CUDA IMA

    # Non-zero but unclassified
    return ErrorClass.OTHER


def _deserialize_child_output(data: bytes) -> Any:
    """Load the child's ``torch.save`` result WITHOUT executing candidate code."""
    import io

    import torch

    return torch.load(io.BytesIO(data), weights_only=True)


def run_in_subprocess(
    callable_module: str,
    callable_name: str,
    inputs: tuple[Any, ...],
    *,
    timeout_s: float = 60.0,
    memory_limit_mb: int = 8192,
    extra_env: dict[str, str] | None = None,
) -> SubprocessResult:
    """Run ``<callable_module>.<callable_name>(*inputs)`` in an isolated subprocess."""
    import tempfile
    from pathlib import Path

    # Write the worker script to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="fmrl_sandbox_",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(_WORKER_SCRIPT)
        worker_path = Path(f.name)

    task_bytes = pickle.dumps((callable_module, callable_name, inputs))

    # Propagate sys.path via PYTHONPATH so module paths resolve regardless of the child's cwd.
    env = dict(os.environ)
    parent_paths = [p for p in sys.path if p]
    if env.get("PYTHONPATH"):
        parent_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parent_paths)
    if extra_env:
        env.update(extra_env)

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(worker_path), str(memory_limit_mb)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(input=task_bytes, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            return SubprocessResult(
                success=False,
                output=None,
                error_class=ErrorClass.TIMEOUT,
                stderr="",
                exit_code=_RC_TIMEOUT,
            )
    finally:
        worker_path.unlink(missing_ok=True)

    rc = proc.returncode
    stderr_text = stderr_bytes.decode(errors="replace")

    if rc != 0:
        error_class = _classify_subprocess_failure(stderr_text, rc)
        return SubprocessResult(
            success=False,
            output=None,
            error_class=error_class,
            stderr=stderr_text,
            exit_code=rc,
        )

    # Safely deserialize the result (never executes candidate code; see helper).
    try:
        output = _deserialize_child_output(stdout_bytes)
    except Exception as exc:
        return SubprocessResult(
            success=False,
            output=None,
            error_class=ErrorClass.OTHER,
            stderr=f"deserialize error: {exc}",
            exit_code=rc,
        )

    return SubprocessResult(
        success=True,
        output=output,
        error_class=ErrorClass.OK,
        stderr=stderr_text,
        exit_code=rc,
    )
