"""Regression: the C1 forward bench reports scale-normalized parity."""

from __future__ import annotations

import inspect

from lethe.bench import c1_forward_chunked_scan


def test_c1_parity_uses_scale_normalized_stats() -> None:
    src = inspect.getsource(c1_forward_chunked_scan)
    assert "_parity_stats(" in src
    assert "ours_vs_official_max_err" not in src  # the bare-max_err keys are gone
    assert "ours_vs_reference_max_err" not in src
