"""CPU tests for the autotuner config action space (kernels/autotune.py)."""

from __future__ import annotations

import pytest

from flash_mamba_rl.kernels.autotune import (
    KernelConfig,
    ShapeSpec,
    grid_size,
    iter_configs,
    tunable_ops,
    validate,
)


def test_searched_omits_none() -> None:
    cfg = KernelConfig(block_d=64, num_warps=8)
    assert cfg.searched() == {"block_d": 64, "num_warps": 8}


def test_empty_config_is_the_shipped_default_and_always_valid() -> None:
    # The all-None config means "use every shipped heuristic" — it must pass
    # for every op, since it is exactly the pre-autotune launch path.
    for op in tunable_ops():
        assert validate(op, KernelConfig()) == []


def test_validate_accepts_grid_point() -> None:
    cfg = KernelConfig(block_d=64, num_warps=8, num_stages=2)
    assert validate("forward_chunked_scan", cfg) == []


def test_validate_rejects_out_of_grid_value() -> None:
    v = validate("forward_chunked_scan", KernelConfig(num_warps=3))
    assert v and any("num_warps" in s for s in v)


def test_validate_rejects_wrong_knob_for_op() -> None:
    # block_d is not tunable for the head-structured rope op (block_p is).
    v = validate("complex_scan_rope", KernelConfig(block_d=64))
    assert v and any("block_d" in s for s in v)


def test_validate_rejects_unknown_op() -> None:
    assert validate("not_an_op", KernelConfig()) != []


def test_validate_chunk_k_must_divide_seq_len() -> None:
    cfg = KernelConfig(chunk_k=16)
    assert validate("backward_selective_scan", cfg, shape=ShapeSpec(2, 2048, 1024)) == []
    bad = validate("backward_selective_scan", cfg, shape=ShapeSpec(2, 1000, 1024))
    assert bad and any("chunk_k" in s for s in bad)


@pytest.mark.parametrize("op", tunable_ops())
def test_iter_configs_matches_grid_size_and_all_validate(op: str) -> None:
    configs = list(iter_configs(op))
    assert len(configs) == grid_size(op)
    assert all(validate(op, cfg) == [] for cfg in configs)


def test_searched_includes_scan_mode_and_chunk_len() -> None:
    cfg = KernelConfig(scan_mode="chunk_parallel", chunk_len=128)
    assert cfg.searched() == {"scan_mode": "chunk_parallel", "chunk_len": 128}


def test_scan_mode_is_tunable_for_forward_scan() -> None:
    assert validate("forward_chunked_scan", KernelConfig(scan_mode="chunk_parallel")) == []
    assert validate("forward_chunked_scan", KernelConfig(scan_mode="serial")) == []


def test_scan_mode_is_tunable_for_backward_scan() -> None:
    # The chunk-parallel lever also reassociates the SISO backward (C2).
    assert validate("backward_selective_scan", KernelConfig(scan_mode="chunk_parallel")) == []
    assert validate("backward_selective_scan", KernelConfig(scan_mode="serial")) == []
    cfg = KernelConfig(scan_mode="chunk_parallel", chunk_len=256)
    assert validate("backward_selective_scan", cfg, shape=ShapeSpec(2, 4096, 1024)) == []


def test_validate_rejects_unknown_scan_mode() -> None:
    v = validate("forward_chunked_scan", KernelConfig(scan_mode="parallel"))
    assert v and any("scan_mode" in s for s in v)


def test_block_p_not_tunable_for_mimo_backward() -> None:
    # The mimo grid has no p axis (one masked BLOCK_P tile covers headdim), so
    # block_p is a correctness floor there — searched only for the rope op.
    v = validate("mimo_backward", KernelConfig(block_p=16))
    assert v and any("block_p" in s for s in v)
    assert validate("complex_scan_rope", KernelConfig(block_p=16)) == []


def test_scan_mode_not_tunable_for_other_ops() -> None:
    # The chunk-parallel lever is for the SISO scan (forward + C2 backward); the
    # head-structured ops (mimo) don't carry it.
    v = validate("mimo_backward", KernelConfig(scan_mode="chunk_parallel"))
    assert v and any("scan_mode" in s for s in v)


def test_validate_chunk_len_must_divide_seq_len() -> None:
    cfg = KernelConfig(scan_mode="chunk_parallel", chunk_len=128)
    assert validate("forward_chunked_scan", cfg, shape=ShapeSpec(2, 4096, 1024)) == []
    bad = validate("forward_chunked_scan", cfg, shape=ShapeSpec(2, 1000, 1024))
    assert bad and any("chunk_len" in s for s in bad)
