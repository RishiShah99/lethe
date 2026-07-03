"""Kernel candidate discovery + loading.

Candidate kernels live in directories on disk, one ``.py`` file per
kernel. Each file declares one callable (the kernel) plus optional
``__candidate_op__`` metadata identifying which reference op it
implements.

The loader scans a directory for ``kernel_*.py`` files, reads their
source, and bundles each into a ``KernelCandidate`` for the verifier
to compile, run, and score.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KernelCandidate:
    """A single candidate kernel ready for verification.

    Attributes
    ----------
    source_path:
        Absolute path to the candidate's ``.py`` file.
    source_code:
        The full text of the candidate source (read eagerly so the
        verifier's compile sandbox can submit it without re-reading).
    callable_name:
        Name of the kernel callable inside the module. Defaults to the
        file's basename without ``kernel_`` prefix and ``.py`` suffix.
    target_op:
        Name of the reference op this candidate implements, if declared
        via ``__candidate_op__`` in the module. ``None`` when not declared.
    metadata:
        Free-form metadata read from a module-level ``__candidate_meta__``
        dict, if present.
    """

    source_path: Path
    source_code: str
    callable_name: str
    target_op: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_module_attrs(source: str) -> dict[str, Any]:
    """Extract literal module-level assignments (constants only).

    Walks the AST and picks up assignments where the RHS is a literal
    (string, number, list, dict, tuple). Skips anything that requires
    execution. This lets us read ``__candidate_op__ = "forward_chunked_scan"``
    without importing the module.
    """
    attrs: dict[str, Any] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return attrs

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        attrs[target.id] = value
    return attrs


def _infer_callable_name(path: Path, attrs: dict[str, Any]) -> str:
    """Pick the callable name. Honour ``__candidate_callable__`` if set."""
    declared = attrs.get("__candidate_callable__")
    if isinstance(declared, str):
        return declared
    stem = path.stem
    if stem.startswith("kernel_"):
        return stem[len("kernel_") :]
    return stem


def load_candidate(path: str | Path) -> KernelCandidate:
    """Load a single candidate file into a ``KernelCandidate``.

    Parameters
    ----------
    path:
        Path to the candidate ``.py`` file.

    Returns
    -------
    KernelCandidate
        Frozen dataclass with source and metadata populated.

    Raises
    ------
    FileNotFoundError
        If *path* does not point to an existing file.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"candidate file not found: {p}")

    source = p.read_text(encoding="utf-8")
    attrs = _parse_module_attrs(source)

    target_op = attrs.get("__candidate_op__")
    if not isinstance(target_op, str):
        target_op = None

    metadata_raw = attrs.get("__candidate_meta__", {})
    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

    return KernelCandidate(
        source_path=p,
        source_code=source,
        callable_name=_infer_callable_name(p, attrs),
        target_op=target_op,
        metadata=metadata,
    )


def discover_candidates(directory: str | Path) -> list[KernelCandidate]:
    """Find all ``kernel_*.py`` candidates in *directory* (non-recursive).

    Parameters
    ----------
    directory:
        Directory to scan. Files matching ``kernel_*.py`` are loaded.

    Returns
    -------
    list[KernelCandidate]
        Candidates sorted by file name for deterministic ordering.

    Raises
    ------
    NotADirectoryError
        If *directory* does not exist or is not a directory.
    """
    d = Path(directory).resolve()
    if not d.is_dir():
        raise NotADirectoryError(f"not a directory: {d}")

    paths = sorted(d.glob("kernel_*.py"))
    return [load_candidate(p) for p in paths]


def import_candidate(candidate: KernelCandidate) -> Callable[..., Any]:
    """Import the candidate's module and return its kernel callable.

    Each call produces a fresh module instance (no caching), so RL
    sweeps that regenerate kernels under the same path get the new
    version.

    Raises
    ------
    AttributeError
        If the named callable is not present in the loaded module.
    ImportError
        If the module fails to load.
    """
    spec = importlib.util.spec_from_file_location(
        f"_fmrl_candidate_{candidate.source_path.stem}",
        candidate.source_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {candidate.source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, candidate.callable_name, None)
    if fn is None or not callable(fn):
        raise AttributeError(
            f"{candidate.source_path}: no callable named {candidate.callable_name!r}"
        )
    return fn  # type: ignore[no-any-return]
