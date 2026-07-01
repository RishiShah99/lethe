"""Regression: the C1 forward bench reports scale-normalized parity.

c1 previously stored a bare absolute ``max_err`` (bf16 included, no scale
normalization), unlike c2-c6's ``_parity_stats``. The bench parity block runs
only under mamba_ssm + CUDA, so this is a source-level check that c1 routes both
parity comparands through the shared ``_parity_stats`` helper.
"""

from __future__ import annotations

import inspect

from flash_mamba_rl.bench import c1_forward_chunked_scan


def test_c1_parity_uses_scale_normalized_stats() -> None:
    src = inspect.getsource(c1_forward_chunked_scan)
    assert "_parity_stats(" in src
    assert "ours_vs_official_max_err" not in src  # the bare-max_err keys are gone
    assert "ours_vs_reference_max_err" not in src
