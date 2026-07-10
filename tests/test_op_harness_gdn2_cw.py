"""Harness tests for the GDN-2 channel-wise crown gate (Phase 3)."""

import pytest
import torch

from lethe.kernels.cute.gdn2_assemble import assembled_channelwise_gdn2_backward
from lethe.verifier.op_harness import (
    GDN2_BWD_GRAD_FIELDS,
    gdn2_bwd_candidate_adapter,
    gdn2_channelwise_reference_adapter,
    verify_gdn2_channelwise_op,
    verify_gdn2_channelwise_op_all_grads,
)

_ALL_GATES_COUNT = 12


class TestGdn2ChannelwiseGate:
    def test_reference_adapter_deterministic(self) -> None:
        do = torch.randn(2, 16, 8)
        for field in GDN2_BWD_GRAD_FIELDS:
            adapted = gdn2_channelwise_reference_adapter(field)
            assert torch.equal(adapted(do), adapted(do.clone())), field

    def test_candidate_equals_reference_on_cpu(self) -> None:
        """On CPU the assembly IS the refs assembly, so the views coincide exactly."""
        do = torch.randn(2, 16, 8)
        for field in GDN2_BWD_GRAD_FIELDS:
            cand = gdn2_bwd_candidate_adapter(assembled_channelwise_gdn2_backward, field)
            ref = gdn2_channelwise_reference_adapter(field)
            assert torch.equal(cand(do), ref(do)), field

    def test_all_gates_pass_on_cpu_grad_v_view(self) -> None:
        results = verify_gdn2_channelwise_op(
            assembled_channelwise_gdn2_backward,
            grad_field="grad_v",
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: r.reason for name, r in results.items() if not r.passed}
        assert not failed, f"channel-wise grad_v gates failed on CPU: {failed}"

    @pytest.mark.slow
    def test_all_gates_pass_all_views(self) -> None:
        all_results = verify_gdn2_channelwise_op_all_grads(
            assembled_channelwise_gdn2_backward,
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert set(all_results) == set(GDN2_BWD_GRAD_FIELDS)
        for field, results in all_results.items():
            assert len(results) == _ALL_GATES_COUNT, field
            failed = {name: r.reason for name, r in results.items() if not r.passed}
            assert not failed, f"channel-wise {field} gates failed on CPU: {failed}"
