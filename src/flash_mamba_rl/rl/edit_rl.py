"""E2.f — edit-RL: the action is a correctness-preserving edit of the kernel.

config-RL (E2.c) is correctness-invariant by construction — it only re-tunes
launch knobs, so it can never *discover* a structurally different kernel. Edit-RL
hands the policy the source of the trusted kernel plus a SEARCH/REPLACE edit
action; the edited source is graded through the *untrusted* path
(:func:`flash_mamba_rl.verifier.candidate_scoring.score_candidate_source`: the
AST/oracle screens, the full gate battery, then the speedup bench at the target
shape). Any edit that changes the result, fails to compile, spills, or OOMs is
caught and scores at or below the contract-fail floor; an edit earns speedup
credit only after every gate re-passes. This is the only action space that can
find structural wins (memory layout, vectorization, fusion) while staying
reference-faithful — the verifier, never the policy, is the correctness oracle.

The base is the self-contained SFT target (``triton`` on the box, where the
kernel actually runs; ``eager`` for CPU plumbing tests). The edit is one or more
git-conflict-style blocks::

    <<<<<<< SEARCH
    <exact lines from the base>
    =======
    <replacement lines>
    >>>>>>> REPLACE

Each SEARCH must match the base exactly once (an absent or ambiguous SEARCH is an
illegal action, scored 0.0 — the analog of the source track's no_code_block). A
completion carrying no SEARCH marker at all is a no_code_block; one with a marker
but a malformed/unmatched block reaches the scorer and lands at the
``illegal_edit`` 0.0, so a botched edit stays distinguishable from no attempt.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from flash_mamba_rl.kernels.autotune import ShapeSpec
from flash_mamba_rl.rl.sft_targets import target_source
from flash_mamba_rl.verifier.candidate_scoring import (
    DEFAULT_EXCLUDE_GATES,
    score_candidate_source,
)

_SEARCH_MARK = "<<<<<<< SEARCH"
_DIVIDER_MARK = "======="
_REPLACE_MARK = ">>>>>>> REPLACE"
_MARKERS = frozenset({_SEARCH_MARK, _DIVIDER_MARK, _REPLACE_MARK})


def extract_edits(completion: str) -> str | None:
    """The completion verbatim if it carries a SEARCH marker, else None.

    None is a ``no_code_block`` (0.0); the scorer re-parses the returned text,
    so a present-but-malformed block still reaches it (``illegal_edit`` 0.0).
    """
    return completion if _SEARCH_MARK in completion else None


def parse_edits(text: str) -> list[tuple[str, str]] | None:
    """Parse the SEARCH/REPLACE blocks in *text*; None if none are well-formed.

    A block is ``<<<<<<< SEARCH`` … ``=======`` … ``>>>>>>> REPLACE`` on their
    own lines. An unterminated block (a SEARCH with no matching divider or
    REPLACE) makes the whole emission illegal — a partial edit must not apply.
    A body line that is itself a bare marker (e.g. source containing a
    ``=======`` line) is unrepresentable in this grammar — it would silently
    shift the block boundaries — so such an emission is rejected outright
    rather than mis-parsed into a wrong edit.
    """
    lines = text.splitlines()
    edits: list[tuple[str, str]] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() != _SEARCH_MARK:
            i += 1
            continue
        i += 1
        search: list[str] = []
        while i < n and lines[i].strip() != _DIVIDER_MARK:
            search.append(lines[i])
            i += 1
        if i >= n:
            return None
        i += 1
        replace: list[str] = []
        while i < n and lines[i].strip() != _REPLACE_MARK:
            replace.append(lines[i])
            i += 1
        if i >= n:
            return None
        i += 1
        if any(line.strip() in _MARKERS for line in (*search, *replace)):
            return None
        edits.append(("\n".join(search), "\n".join(replace)))
    return edits or None


def apply_edits(base: str, edits: list[tuple[str, str]]) -> str | None:
    """Apply each ``(search, replace)`` to *base*; None on any non-unique match.

    A SEARCH must occur exactly once: zero means it doesn't match the base, more
    than one means the replacement is ambiguous — both reject so the edit is
    deterministic. An empty SEARCH (a whole-file insertion) is rejected for the
    same reason. Replacements apply in order, so a later block sees the text the
    earlier ones produced.
    """
    src = base
    for search, replace in edits:
        if search == "" or src.count(search) != 1:
            return None
        src = src.replace(search, replace, 1)
    return src


def _illegal_action_result(error: str) -> dict[str, Any]:
    """A 0.0-reward score dict for an unparseable / non-matching edit.

    Mirrors ``candidate_scoring._failure``'s shape so rollout rows stay uniform
    across the source, config, and edit tracks.
    """
    return {
        "status": "illegal_edit",
        "error": error,
        "compiled": False,
        "contracts_passed": False,
        "reward": 0.0,
        "gates": {},
        "views_passed": 0,
        "views_total": 0,
        "first_failed_view": None,
    }


def score_edit_candidate(
    text: str,
    *,
    op: str,
    base_variant: str = "triton",
    device: str = "cuda",
    shape: ShapeSpec | None = None,
    timeout_s: float = 300.0,
    exclude_gates: tuple[str, ...] = DEFAULT_EXCLUDE_GATES,
    reward_shaping: str = "none",
    measure_speedup: bool = True,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply one edit emission to the base kernel and score the result.

    Malformed or non-matching edits score 0.0; otherwise the edited source runs
    the full untrusted gate battery + speedup bench at ``shape`` via
    :func:`score_candidate_source`, so correctness is re-checked from scratch.
    """
    edits = parse_edits(text)
    if not edits:
        return _illegal_action_result(f"no SEARCH/REPLACE block: {text[:120]!r}")
    base = target_source(op, base_variant)
    edited = apply_edits(base, edits)
    if edited is None:
        return _illegal_action_result("a SEARCH block did not match the base exactly once")
    bench_shape = (shape.batch, shape.seq_len, shape.width) if shape is not None else None
    return score_candidate_source(
        edited,
        op=op,
        device=device,
        timeout_s=timeout_s,
        exclude_gates=exclude_gates,
        reward_shaping=reward_shaping,
        measure_speedup=measure_speedup,
        shape=bench_shape,
        extra_env=extra_env,
    )


def build_edit_scorer(
    op: str,
    shape: ShapeSpec,
    *,
    base_variant: str = "triton",
    device: str = "cuda",
    timeout_s: float = 300.0,
    measure_speedup: bool = True,
    extra_env: dict[str, str] | None = None,
) -> Callable[[str], dict[str, Any]]:
    """A :class:`GRPOTrainingLoop` ``scorer`` closure for (op, shape) edits."""

    def scorer(text: str) -> dict[str, Any]:
        return score_edit_candidate(
            text,
            op=op,
            base_variant=base_variant,
            shape=shape,
            device=device,
            timeout_s=timeout_s,
            measure_speedup=measure_speedup,
            extra_env=extra_env,
        )

    return scorer
