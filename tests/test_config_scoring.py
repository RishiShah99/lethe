"""CPU tests for the config-scoring path (score_candidate_config).

These exercise the plumbing — config validation, the trusted-op bridge, the
gate battery — on CPU, where the configured op runs the eager path (the config
is a CUDA-launch knob). The config's performance *effect* and the speedup
reward are GPU-only and validated on the box.
"""

from __future__ import annotations

from flash_mamba_rl.kernels.autotune import KernelConfig, ShapeSpec
from flash_mamba_rl.verifier.candidate_scoring import score_candidate_config


def test_valid_config_passes_contracts_on_cpu() -> None:
    # Contracts pass; no speedup is measured on CPU -> the 0.5 correct-but-slow
    # floor (the config only changes the CUDA launch, which CPU does not take).
    res = score_candidate_config(
        KernelConfig(num_warps=8, block_d=64),
        op="forward_chunked_scan",
        device="cpu",
    )
    assert res["status"] == "scored"
    assert res["contracts_passed"] is True
    assert res["reward"] == 0.5


def test_invalid_config_scores_zero() -> None:
    # 3 is not in the num_warps grid -> illegal action, reward 0.0, no kernel run.
    res = score_candidate_config(
        KernelConfig(num_warps=3),
        op="forward_chunked_scan",
        device="cpu",
    )
    assert res["status"] == "invalid_config"
    assert res["contracts_passed"] is False
    assert res["reward"] == 0.0


def test_shape_incompatible_chunk_k_rejected() -> None:
    # chunk_k must divide the bench sequence length; 1000 % 16 != 0.
    res = score_candidate_config(
        KernelConfig(chunk_k=16),
        op="backward_selective_scan",
        device="cpu",
        shape=ShapeSpec(2, 1000, 1024),
    )
    assert res["status"] == "invalid_config"
    assert res["reward"] == 0.0


def test_default_config_passes_on_cpu() -> None:
    # The all-None config (shipped default) must score exactly like a valid
    # explicit one — it is the pre-autotune path.
    res = score_candidate_config(KernelConfig(), op="forward_chunked_scan", device="cpu")
    assert res["status"] == "scored"
    assert res["contracts_passed"] is True
