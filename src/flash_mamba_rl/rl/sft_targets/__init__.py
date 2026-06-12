"""Verified warm-start targets: one self-contained eager solution per curriculum op.

Option-A SFT data for the cold-start fix: the base policy never samples a
contract-passing kernel, so GRPO groups are degenerate and the update is
skipped — there is no gradient to follow. These modules are correct
eager-torch candidates that pass the full gate battery through
``score_candidate_source`` at reward 0.5; SFT on prompt→target pairs puts
contract passes inside the sampling distribution so the speedup term can
take over from a live gradient.

Sources are consumed as *text*, never imported by the trainer — the SFT
completion is exactly the bytes the verifier scored. Every target must
stay free of ``candidate_scoring.FORBIDDEN_SOURCE_TOKENS`` (no package /
official-kernel / dynamic-import references) and must mirror the
reference math statement-for-statement: the gates compare against
autograd through the references, so replication fidelity IS the
correctness argument.
"""

from pathlib import Path

# Two variants per op: "eager" (Option A — correct torch, lands the 0.5
# contract floor) and "triton" (Option B — the hand-written kernels made
# self-contained, the neighborhood the speedup term explores from).
_OPS: tuple[str, ...] = (
    "forward_chunked_scan",
    "backward_selective_scan",
    "mimo_backward",
    "complex_scan_rope",
    "fused_block_forward",
    "fused_block_backward",
)

_VARIANT_SUFFIX: dict[str, str] = {"eager": "", "triton": "_triton"}


def available_targets() -> tuple[str, ...]:
    return _OPS


def target_variants(op_name: str) -> tuple[str, ...]:
    if op_name not in _OPS:
        raise KeyError(op_name)
    return tuple(
        v
        for v, suffix in _VARIANT_SUFFIX.items()
        if (Path(__file__).parent / f"{op_name}{suffix}.py").exists()
    )


def target_source(op_name: str, variant: str = "eager") -> str:
    """Return the verified candidate source for *(op_name, variant)*."""
    if op_name not in _OPS:
        raise KeyError(op_name)
    path = Path(__file__).parent / f"{op_name}{_VARIANT_SUFFIX[variant]}.py"
    return path.read_text(encoding="utf-8")
