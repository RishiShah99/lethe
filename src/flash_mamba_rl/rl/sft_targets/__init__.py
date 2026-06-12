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

_TARGETS: dict[str, str] = {
    "forward_chunked_scan": "forward_chunked_scan.py",
    "backward_selective_scan": "backward_selective_scan.py",
    "mimo_backward": "mimo_backward.py",
    "complex_scan_rope": "complex_scan_rope.py",
    "fused_block_forward": "fused_block_forward.py",
    "fused_block_backward": "fused_block_backward.py",
}


def available_targets() -> tuple[str, ...]:
    return tuple(_TARGETS)


def target_source(op_name: str) -> str:
    """Return the verified candidate source for *op_name* (KeyError if unknown)."""
    return (Path(__file__).parent / _TARGETS[op_name]).read_text(encoding="utf-8")
