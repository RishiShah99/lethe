"""Harness tests for the GDN-2 backward verifier wiring.

Mirrors TestMimoBackwardHarness: adapter factoring/determinism/shapes/dtype,
reference-adapter equivalence, a full 12-gate CPU run on one view, and a PRC-02
discrimination check (fp16-state cheat must fail where the honest path passes).

All tests run on CPU only.
"""

import pytest
import torch

from lethe.kernels.ops import gdn2_backward
from lethe.kernels.references.gdn_backward import reference_gdn2_backward
from lethe.verifier.contracts import gate_prc_02_mixed_precision_accumulation
from lethe.verifier.op_harness import (
    GDN2_BWD_GATE_OVERRIDES,
    GDN2_BWD_GRAD_FIELDS,
    GDN2_HEADDIM,
    _gdn2_reduce_candidate,
    gdn2_bwd_candidate_adapter,
    gdn2_bwd_reference_adapter,
    verify_gdn2_bwd_op,
)

_ALL_GATES_COUNT = 12


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _low_precision_state_gdn2_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    *,
    state_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Cheating GDN-2 backward: per-step math in fp32, running state re-rounded.

    Autograd through the low-precision-state forward keeps the backward carry at
    ``state_dtype`` too — the missing-fp32-accumulator failure PRC-02 exists to
    reject. GDN-2's decay-limited memory makes an fp16 state only ~2x over the
    honest fp16 input-rounding floor (uncatchable at a safe atol — measured); a
    bf16 state is the robust adversary that proves the gate's teeth.
    """
    inputs = (q, k, v, g, b, w)
    with torch.enable_grad():
        leaves = [t.detach().float().requires_grad_(True) for t in inputs]
        qf, kf, vf, gf, bf, wf = leaves
        batch, seqlen, nheads, d_k = qf.shape
        d_v = vf.shape[-1]
        qn = _l2norm(qf) * (d_k**-0.5)
        kn = _l2norm(kf)
        S = torch.zeros(batch, nheads, d_k, d_v, dtype=state_dtype)
        outs: list[torch.Tensor] = []
        for t in range(seqlen):
            S = (S.float() * gf[:, t].exp().unsqueeze(-1)).to(state_dtype)
            erase = (S.float() * (bf[:, t] * kn[:, t]).unsqueeze(-1)).sum(-2)
            v_new = wf[:, t] * vf[:, t] - erase
            S = (S.float() + kn[:, t].unsqueeze(-1) * v_new.unsqueeze(-2)).to(state_dtype)
            outs.append((S.float() * qn[:, t].unsqueeze(-1)).sum(-2))
        o = torch.stack(outs, dim=1)
        grads = torch.autograd.grad(o, leaves, do.float())
    return tuple(grad.to(q.dtype) for grad in grads)


def _bf16_state_gdn2_bwd(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return _low_precision_state_gdn2_bwd(*args, state_dtype=torch.bfloat16)


class TestGdn2BackwardHarness:
    def test_factoring_covers_every_gate_d_model(self) -> None:
        for d_model in (4, 8, 16, 32, 64, 1024):
            assert d_model % GDN2_HEADDIM == 0
            if d_model <= 64:
                out = gdn2_bwd_candidate_adapter(reference_gdn2_backward, "grad_q")(
                    torch.randn(1, 8, d_model)
                )
                assert out.shape == (1, 8, d_model // GDN2_HEADDIM, GDN2_HEADDIM)

    def test_indivisible_d_model_rejected(self) -> None:
        adapted = gdn2_bwd_candidate_adapter(reference_gdn2_backward, "grad_q")
        with pytest.raises(ValueError, match="not divisible"):
            adapted(torch.randn(1, 8, 6))

    def test_adapters_deterministic_per_view(self) -> None:
        do = torch.randn(2, 16, 8)
        for field in GDN2_BWD_GRAD_FIELDS:
            adapted = gdn2_bwd_candidate_adapter(gdn2_backward, field)
            assert torch.equal(adapted(do), adapted(do.clone())), field

    def test_reference_adapter_matches_direct_reference(self) -> None:
        do = torch.randn(2, 16, 8)
        for field in GDN2_BWD_GRAD_FIELDS:
            ref_adapted = gdn2_bwd_reference_adapter(field)
            cand_adapted = gdn2_bwd_candidate_adapter(reference_gdn2_backward, field)
            assert torch.equal(ref_adapted(do), cand_adapted(do)), field

    def test_view_outputs_have_expected_shapes(self) -> None:
        b, seq, d_model = 2, 16, 8
        h = d_model // GDN2_HEADDIM
        do = torch.randn(b, seq, d_model)
        expected = {
            "grad_q": (b, seq, h, GDN2_HEADDIM),
            "grad_k": (b, seq, h, GDN2_HEADDIM),
            "grad_v": (b, seq, h, GDN2_HEADDIM),
            "grad_g": (b, seq, h, GDN2_HEADDIM),
            "grad_b": (b, seq, h, GDN2_HEADDIM),
            "grad_w": (b, seq, h, GDN2_HEADDIM),
        }
        for field, shape in expected.items():
            out = gdn2_bwd_candidate_adapter(gdn2_backward, field)(do)
            assert out.shape == shape, field

    def test_adapter_follows_input_dtype(self) -> None:
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            out = gdn2_bwd_candidate_adapter(gdn2_backward, "grad_q")(
                torch.randn(1, 8, 4, dtype=dtype)
            )
            assert out.dtype == dtype

    def test_all_gates_pass_on_cpu_grad_v_view(self) -> None:
        results = verify_gdn2_bwd_op(
            gdn2_backward,
            grad_field="grad_v",
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: r.reason for name, r in results.items() if not r.passed}
        assert not failed, f"grad_v gates failed on CPU: {failed}"
        assert results["gate_res_02_resource_limits"].details.get("applicable") is True


class TestGdn2Prc02Discrimination:
    def test_scale_aware_prc02_separates_honest_from_coarse_accumulator(self) -> None:
        # Calibrated floor (scratch/gdn2_prc02_floor.py): honest fp16 input-rounding
        # floor ~6e-4 of scale; bf16-state cheat ~6.7e-3 (11x) on grad_v; unit atol
        # 2e-3 sits between. (An fp16-state cheat is only ~2x over the floor — GDN-2's
        # decay-limited memory makes it uncatchable at a safe atol; the bf16 cheat is
        # the robust adversary. See _gdn2_prc02.)
        torch.manual_seed(11)
        kwargs = dict(GDN2_BWD_GATE_OVERRIDES["grad_v"]["gate_prc_02_mixed_precision_accumulation"])
        honest = gate_prc_02_mixed_precision_accumulation(
            gdn2_bwd_candidate_adapter(gdn2_backward, "grad_v"),
            gdn2_bwd_reference_adapter("grad_v"),
            **kwargs,
        )
        assert honest.passed, f"honest fp32-accumulator backward rejected: {honest.reason}"
        cheat = gate_prc_02_mixed_precision_accumulation(
            gdn2_bwd_candidate_adapter(_bf16_state_gdn2_bwd, "grad_v"),
            gdn2_bwd_reference_adapter("grad_v"),
            **kwargs,
        )
        assert not cheat.passed, f"bf16-state cheat slipped past PRC-02: {cheat.details}"


class TestGdn2ReduceCandidate:
    def test_grad_beta_combines_erase_and_write(self) -> None:
        grads = tuple(torch.randn(2, 4, 3, 5) for _ in range(6))
        out = _gdn2_reduce_candidate("grad_beta", grads)
        assert torch.equal(out, grads[4].sum(-1) + grads[5].sum(-1))

    def test_unknown_view_raises(self) -> None:
        # A typo'd view must NOT fall through to the beta reduction (a green
        # result from an operator-wiring error); it must raise.
        grads = tuple(torch.randn(2, 4, 3, 5) for _ in range(6))
        with pytest.raises(ValueError, match="grad_typo"):
            _gdn2_reduce_candidate("grad_typo", grads)
