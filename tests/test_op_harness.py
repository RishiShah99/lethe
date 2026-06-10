"""CPU-side validation of the C1 op and the op-harness gate adapters.

The Triton path needs the box; everything here exercises the eager path,
the adapter layer, and the contracts plumbing (device threading + per-gate
overrides) so the GPU run only has to answer GPU-specific questions.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from flash_mamba_rl.kernels.ops import backward_selective_scan, forward_chunked_scan
from flash_mamba_rl.kernels.references import (
    reference_backward_selective_scan,
    reference_forward_chunked_scan,
)
from flash_mamba_rl.verifier.contracts import (
    gate_cmp_01_input_variation,
    gate_ord_03_noncommutative_reduction,
    gate_prc_02_mixed_precision_accumulation,
    run_all_gates,
)
from flash_mamba_rl.verifier.op_harness import (
    BWD_GRAD_FIELDS,
    SCAN_BWD_GATE_OVERRIDES,
    SCAN_CHUNK_SIZE,
    bwd_scan_candidate_adapter,
    bwd_scan_reference_adapter,
    scan_candidate_adapter,
    scan_reference_adapter,
    verify_bwd_scan_op,
    verify_scan_op,
)

# Every gate is implemented as of RES-02; on CPU, RES-02 passes as
# not-applicable (no compiled artifact) and the other 11 run for real.
_ALL_GATES_COUNT = 12


def _fp16_accumulator_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor,
    *,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Cheating scan: discretisation in fp32 but the running state in fp16."""
    import torch.nn.functional as F

    delta_bar = F.softplus(delta.float())
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * a.float().unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * b_proj.float().unsqueeze(2)
    batch, seq_len, d_model = u.shape
    h = torch.zeros(batch, d_model, a.shape[1], dtype=torch.float16, device=u.device)
    ys: list[torch.Tensor] = []
    for t in range(seq_len):
        bu = b_bar[:, t] * u[:, t].float().unsqueeze(-1)
        h = (a_bar[:, t].half() * h + bu.half()).half()
        y_t = (h * c_proj[:, t].half().unsqueeze(1)).sum(-1).float()
        ys.append(y_t + d_skip.float() * u[:, t].float())
    return torch.stack(ys, dim=1).to(u.dtype)


def _unguarded_softplus_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor,
    *,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Cheating scan: softplus written naively as log1p(exp(x)).

    The single most natural way to write softplus, and exactly correct for
    every delta below ~89 — but exp overflows to Inf above that, so the
    saturated aux entries blow up where torch's thresholded softplus stays
    linear. Without the saturation probe this kernel passes every gate.
    """
    import torch.nn.functional as F  # noqa: F401  (parallel structure with reference)

    delta_bar = torch.log1p(torch.exp(delta.float()))  # no large-x guard
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * a.float().unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * b_proj.float().unsqueeze(2)
    batch, seq_len, d_model = u.shape
    h = torch.zeros(batch, d_model, a.shape[1], dtype=torch.float32, device=u.device)
    ys: list[torch.Tensor] = []
    for t in range(seq_len):
        h = a_bar[:, t] * h + b_bar[:, t] * u[:, t].float().unsqueeze(-1)
        ys.append(
            (h * c_proj[:, t].float().unsqueeze(1)).sum(-1) + d_skip.float() * u[:, t].float()
        )
    return torch.stack(ys, dim=1).to(u.dtype)


def _cumprod_trick_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor,
    *,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Cheating scan: 'parallelises' the recurrence as P_t * cumsum(bu_t / P_t).

    Algebraically exact, numerically catastrophic: P_t underflows over long
    near-integrator sequences and the ratio cumsum amplifies tiny terms.
    The classic wrong way to vectorise a selective scan — ORD-03's target.
    """
    import torch.nn.functional as F

    delta_bar = F.softplus(delta.float())
    log_a_bar = delta_bar.unsqueeze(-1) * a.float().unsqueeze(0).unsqueeze(0)  # [B,L,D,N]
    b_bar = delta_bar.unsqueeze(-1) * b_proj.float().unsqueeze(2)
    bu = b_bar * u.float().unsqueeze(-1)
    p = torch.exp(torch.cumsum(log_a_bar, dim=1))  # running decay products
    h = p * torch.cumsum(bu / p, dim=1)
    y = (h * c_proj.float().unsqueeze(2)).sum(-1) + d_skip.float() * u.float()
    return y.to(u.dtype)


def _scan_inputs(
    b: int = 2,
    seq: int = 8,
    d: int = 4,
    n: int = 8,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    u = torch.randn(b, seq, d, dtype=dtype)
    delta = torch.randn(b, seq, d, dtype=dtype)
    a = (-torch.rand(d, n)).to(dtype)
    b_proj = torch.randn(b, seq, n, dtype=dtype)
    c_proj = torch.randn(b, seq, n, dtype=dtype)
    d_skip = torch.randn(d, dtype=dtype)
    return u, delta, a, b_proj, c_proj, d_skip


class TestForwardChunkedScanCpu:
    def test_bitwise_matches_reference_fp32(self) -> None:
        args = _scan_inputs()
        y_ours = forward_chunked_scan(*args, chunk_size=4)
        y_ref = reference_forward_chunked_scan(*args, chunk_size=4)
        assert torch.equal(y_ours, y_ref), "eager path must replicate the reference bitwise"

    def test_fp16_close_to_fp32(self) -> None:
        args32 = _scan_inputs(seq=16, d=8)
        args16 = tuple(t.to(torch.float16) for t in args32)
        y16 = forward_chunked_scan(*args16, chunk_size=8)
        y32 = reference_forward_chunked_scan(*(t.to(torch.float32) for t in args16), chunk_size=8)
        assert y16.dtype == torch.float16
        assert torch.allclose(y16.float(), y32, atol=1e-3, rtol=1e-3)

    def test_fp64_native(self) -> None:
        args64 = tuple(t.to(torch.float64) for t in _scan_inputs())
        y = forward_chunked_scan(*args64, chunk_size=4)
        assert y.dtype == torch.float64
        assert torch.isfinite(y).all()

    def test_chunk_divisibility_error_parity(self) -> None:
        args = _scan_inputs(seq=8)
        with pytest.raises(ValueError, match="divisible"):
            forward_chunked_scan(*args, chunk_size=3)
        with pytest.raises(ValueError, match="divisible"):
            reference_forward_chunked_scan(*args, chunk_size=3)

    def test_gradients_match_reference_autograd(self) -> None:
        args_ours = _scan_inputs(seed=3)
        args_ref = tuple(t.clone() for t in args_ours)
        for t in (*args_ours, *args_ref):
            t.requires_grad_(True)

        y_ours = forward_chunked_scan(*args_ours, chunk_size=4)
        y_ref = reference_forward_chunked_scan(*args_ref, chunk_size=4)
        grad_out = torch.ones_like(y_ours)
        y_ours.backward(grad_out)
        y_ref.backward(grad_out)

        for ours, ref in zip(args_ours, args_ref, strict=True):
            assert ours.grad is not None and ref.grad is not None
            assert torch.allclose(ours.grad, ref.grad, atol=1e-6), "gradient mismatch"

    def test_gradcheck_fp64(self) -> None:
        args = tuple(
            t.to(torch.float64).requires_grad_(True) for t in _scan_inputs(b=1, seq=4, d=2, n=3)
        )
        assert torch.autograd.gradcheck(
            lambda *xs: forward_chunked_scan(*xs, chunk_size=2), args, eps=1e-6, atol=1e-4
        )


class TestScanHarness:
    def test_adapter_is_deterministic(self) -> None:
        adapted = scan_candidate_adapter(forward_chunked_scan)
        u = torch.randn(2, 16, 4)
        assert torch.equal(adapted(u), adapted(u.clone()))

    def test_reference_adapter_matches_direct_reference(self) -> None:
        ref_adapted = scan_reference_adapter()
        cand_adapted = scan_candidate_adapter(reference_forward_chunked_scan)
        u = torch.randn(2, 16, 4)
        assert torch.equal(ref_adapted(u), cand_adapted(u))

    def test_adapter_follows_input_dtype(self) -> None:
        adapted = scan_candidate_adapter(forward_chunked_scan)
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            y = adapted(torch.randn(1, 8, 4, dtype=dtype))
            assert y.dtype == dtype

    def test_all_gates_pass_on_cpu(self) -> None:
        results = verify_scan_op(forward_chunked_scan)
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: result.reason for name, result in results.items() if not result.passed}
        assert not failed, f"gates failed on CPU: {failed}"

    def test_raw_reference_as_candidate_fails_precision_gate(self) -> None:
        # The raw oracle rejects non-fp32 inputs, so as a *candidate* it must
        # fail PRC-01 — candidates own their mixed-precision handling.
        results = verify_scan_op(reference_forward_chunked_scan)
        assert not results["gate_prc_01_precision_regime"].passed

    def test_fp16_accumulator_cheat_caught_by_prc02(self) -> None:
        # Discriminative-power check for PRC-02 at the scan's override shape:
        # a scan that keeps h (and the N-dot) in fp16 must be rejected, while
        # the honest fp32-accumulating op passes (test_all_gates_pass_on_cpu).
        # With the Mamba-realistic delta in the aux distribution the measured
        # floors are ~6e-3 honest vs ~1.5e-1 cheat against atol=2e-2.
        results = verify_scan_op(_fp16_accumulator_scan)
        prc02 = results["gate_prc_02_mixed_precision_accumulation"]
        assert not prc02.passed, "PRC-02 lost its discriminative power for the scan op"

    def test_unguarded_softplus_caught(self) -> None:
        # Review finding (C1 phase): without saturation-regime delta in the
        # aux distribution, a softplus with no large-x guard — the most
        # natural way to write it — passed all 12 gates. The saturated
        # entries (delta=95 > fp32 exp overflow at ~89) force it to Inf
        # where the reference stays linear.
        results = verify_scan_op(_unguarded_softplus_scan)
        cmp01 = results["gate_cmp_01_input_variation"]
        assert not cmp01.passed, "saturation probe lost: unguarded softplus passed CMP-01"

    def test_cumprod_trick_caught_by_ord03_with_scan_tolerances(self) -> None:
        # ORD-03 runs with a tolerance for the scan op (hardware kernels
        # cannot be bitwise against torch eager); this pins that the
        # relaxation keeps rejecting numerically unstable scan orderings.
        # The cumprod-ratio trick tracks the oracle to ~3e-5 up to L=4096,
        # then its decay products underflow: at the override length L=8192
        # it NaNs out while honest reorder noise stays ~1e-4 vs atol 1e-3.
        results = verify_scan_op(_cumprod_trick_scan)
        ord03 = results["gate_ord_03_noncommutative_reduction"]
        assert not ord03.passed, "ORD-03 lost its discriminative power for the scan op"

    def test_gate_overrides_reach_named_gate_only(self) -> None:
        candidate = scan_candidate_adapter(forward_chunked_scan)
        reference = scan_reference_adapter()
        baseline = run_all_gates(candidate, reference)
        assert baseline["gate_cmp_01_input_variation"].passed

        overridden = run_all_gates(
            candidate,
            reference,
            gate_overrides={"gate_cmp_01_input_variation": {"atol": -1.0, "rtol": 0.0}},
        )
        assert not overridden["gate_cmp_01_input_variation"].passed
        assert overridden["gate_cmp_03_shape_polymorphism"].passed, (
            "override leaked beyond the named gate"
        )

    def test_resource_meta_override_activates_res02(self) -> None:
        results = verify_scan_op(
            forward_chunked_scan, resource_meta={"n_regs": 64, "shared_bytes": 1024}
        )
        res02 = results["gate_res_02_resource_limits"]
        assert res02.passed
        assert res02.details.get("applicable") is True

    def test_chunk_size_divides_all_gate_sequence_lengths(self) -> None:
        # Gate default shapes: CMP-02 L=8 is the smallest; PRC-02 override
        # L=1024 the largest. The harness chunk size must divide them all,
        # or gates fail on the ValueError instead of testing the kernel.
        for seq_len in (8, 16, 32, 64, 128, 256, 512, 1024):
            assert seq_len % SCAN_CHUNK_SIZE == 0


def _bwd_via_autograd(
    fwd_fn: Callable[..., torch.Tensor],
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Backward-signature wrapper differentiating through a (cheating) forward.

    The forward cheats above carry their pathologies into their gradients —
    fp16 state means an fp16 backward carry, an unguarded softplus NaNs the
    whole graph at saturated delta — so autograd through them yields the
    matching *backward* cheats for the per-gradient gate views.
    """

    def bwd(
        u: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b_proj: torch.Tensor,
        c_proj: torch.Tensor,
        d_skip: torch.Tensor,
        dy: torch.Tensor,
        *,
        chunk_size: int = 8,
    ) -> tuple[torch.Tensor, ...]:
        with torch.enable_grad():
            leaves = [
                t.detach().requires_grad_(True) for t in (u, delta, a, b_proj, c_proj, d_skip)
            ]
            y = fwd_fn(*leaves, chunk_size=chunk_size)
            return torch.autograd.grad(y, leaves, dy)

    return bwd


class TestBackwardSelectiveScanCpu:
    def test_matches_reference_fp32(self) -> None:
        args = _scan_inputs()
        dy = torch.randn(2, 8, 4)
        ours = backward_selective_scan(*args, dy, chunk_size=4)
        ref = reference_backward_selective_scan(*args, dy, chunk_size=4)
        for field, got, want in zip(BWD_GRAD_FIELDS, ours, ref, strict=True):
            assert torch.equal(got, want), f"{field}: eager path must replicate the reference"

    def test_fp16_grads_close_to_fp32_oracle(self) -> None:
        args32 = _scan_inputs(seq=16, d=8)
        args16 = tuple(t.to(torch.float16) for t in args32)
        dy16 = torch.randn(2, 16, 8, dtype=torch.float16)
        ours = backward_selective_scan(*args16, dy16, chunk_size=8)
        ref = reference_backward_selective_scan(
            *(t.to(torch.float32) for t in args16), dy16.to(torch.float32), chunk_size=8
        )
        for field, got, want in zip(BWD_GRAD_FIELDS, ours, ref, strict=True):
            assert got.dtype == torch.float16, field
            assert torch.allclose(got.float(), want, atol=1e-2, rtol=1e-2), field

    def test_fp64_native(self) -> None:
        args64 = tuple(t.to(torch.float64) for t in _scan_inputs())
        dy = torch.randn(2, 8, 4, dtype=torch.float64)
        grads = backward_selective_scan(*args64, dy, chunk_size=4)
        for field, g in zip(BWD_GRAD_FIELDS, grads, strict=True):
            assert g.dtype == torch.float64, field
            assert torch.isfinite(g).all(), field

    def test_chunk_divisibility_error_parity(self) -> None:
        args = _scan_inputs(seq=8)
        dy = torch.randn(2, 8, 4)
        with pytest.raises(ValueError, match="divisible"):
            backward_selective_scan(*args, dy, chunk_size=3)
        with pytest.raises(ValueError, match="divisible"):
            reference_backward_selective_scan(*args, dy, chunk_size=3)

    def test_consistent_with_public_forward_autograd(self) -> None:
        # The op pair must be self-consistent: differentiating the public
        # forward must give the same gradients the public backward returns.
        args = _scan_inputs(seed=11)
        dy = torch.randn(2, 8, 4)
        direct = backward_selective_scan(*args, dy, chunk_size=4)

        leaves = tuple(t.detach().requires_grad_(True) for t in args)
        y = forward_chunked_scan(*leaves, chunk_size=4)
        via_autograd = torch.autograd.grad(y, leaves, dy)
        for field, got, want in zip(BWD_GRAD_FIELDS, direct, via_autograd, strict=True):
            assert torch.equal(got, want), field

    def test_differentiable_wrt_dy_when_required(self) -> None:
        # CMP-02's gradcheck differentiates the op w.r.t. dy; the eager path
        # must build that graph (the VJP is linear in dy).
        args = _scan_inputs()
        dy = torch.randn(2, 8, 4, requires_grad=True)
        grads = backward_selective_scan(*args, dy, chunk_size=4)
        assert grads.grad_u.requires_grad
        (dd,) = torch.autograd.grad(grads.grad_u.sum(), dy)
        assert dd.shape == dy.shape
        assert torch.isfinite(dd).all()


class TestBackwardScanHarness:
    def test_adapters_deterministic_per_view(self) -> None:
        dy = torch.randn(2, 16, 4)
        for field in BWD_GRAD_FIELDS:
            adapted = bwd_scan_candidate_adapter(backward_selective_scan, field)
            assert torch.equal(adapted(dy), adapted(dy.clone())), field

    def test_reference_adapter_matches_direct_reference(self) -> None:
        dy = torch.randn(2, 16, 4)
        for field in BWD_GRAD_FIELDS:
            ref_adapted = bwd_scan_reference_adapter(field)
            cand_adapted = bwd_scan_candidate_adapter(reference_backward_selective_scan, field)
            assert torch.equal(ref_adapted(dy), cand_adapted(dy)), field

    def test_view_outputs_have_expected_shapes(self) -> None:
        b, seq, d = 2, 16, 4
        dy = torch.randn(b, seq, d)
        expected = {
            "grad_u": (b, seq, d),
            "grad_delta": (b, seq, d),
            "grad_A": (d, 16),
            "grad_B": (b, seq, 16),
            "grad_C": (b, seq, 16),
            "grad_D": (d,),
        }
        for field, shape in expected.items():
            out = bwd_scan_candidate_adapter(backward_selective_scan, field)(dy)
            assert out.shape == shape, field

    def test_all_gates_pass_on_cpu_grad_u_view(self) -> None:
        # One full-suite view exercises every gate against the backward op on
        # CPU (the eager path). The remaining five views run the full suite
        # on the GPU box, where the Triton kernel — the actual target — is
        # live; re-running all six against the eager path here would only
        # re-verify autograd against autograd at ~6x the suite cost.
        # resource_meta rides along to pin that the verify wrapper forwards
        # it into RES-02 (applicable instead of not-applicable).
        results = verify_bwd_scan_op(
            backward_selective_scan,
            grad_field="grad_u",
            resource_meta={"n_regs": 64, "shared_bytes": 1024},
        )
        assert len(results) == _ALL_GATES_COUNT
        failed = {name: result.reason for name, result in results.items() if not result.passed}
        assert not failed, f"grad_u gates failed on CPU: {failed}"
        assert results["gate_res_02_resource_limits"].details.get("applicable") is True

    def test_grad_a_prc02_scale_aware_separates_honest_from_cheat(self) -> None:
        # grad_A's PRC-02 runs scale-aware (error carries the magnitude of
        # large cancelling intermediates; B200-measured honest floors are
        # 4.9e-4 Triton / 9.0e-4 eager of output scale vs 1.25e-2 for an
        # fp16 carry). This pins both sides of the 3e-3 unit atol on the
        # eager path.
        kwargs = dict(SCAN_BWD_GATE_OVERRIDES["grad_A"]["gate_prc_02_mixed_precision_accumulation"])
        honest = gate_prc_02_mixed_precision_accumulation(
            bwd_scan_candidate_adapter(backward_selective_scan, "grad_A", saturate=False),
            bwd_scan_reference_adapter("grad_A", saturate=False),
            **kwargs,
        )
        assert honest.passed, f"honest fp32-accumulator backward rejected: {honest.reason}"
        cheat = gate_prc_02_mixed_precision_accumulation(
            bwd_scan_candidate_adapter(
                _bwd_via_autograd(_fp16_accumulator_scan), "grad_A", saturate=False
            ),
            bwd_scan_reference_adapter("grad_A", saturate=False),
            **kwargs,
        )
        assert not cheat.passed, "scale-aware PRC-02 lost its discriminative power on grad_A"

    def test_fp16_accumulator_bwd_cheat_caught_by_prc02(self) -> None:
        # Autograd through an fp16-state forward keeps the backward carry in
        # fp16 too; PRC-02 on the grad_u view must reject it while the honest
        # op passes (test_all_gates_pass_on_cpu_grad_u_view).
        kwargs = dict(SCAN_BWD_GATE_OVERRIDES["grad_u"]["gate_prc_02_mixed_precision_accumulation"])
        result = gate_prc_02_mixed_precision_accumulation(
            bwd_scan_candidate_adapter(
                _bwd_via_autograd(_fp16_accumulator_scan), "grad_u", saturate=False
            ),
            bwd_scan_reference_adapter("grad_u", saturate=False),
            **kwargs,
        )
        assert not result.passed, "PRC-02 lost its discriminative power for the backward op"

    def test_unguarded_softplus_bwd_cheat_caught_by_cmp01(self) -> None:
        # The saturation probe carries over to the backward: an unguarded
        # softplus turns delta=95 into Inf, and every gradient inherits the
        # NaN where the reference stays finite.
        result = gate_cmp_01_input_variation(
            bwd_scan_candidate_adapter(_bwd_via_autograd(_unguarded_softplus_scan), "grad_delta"),
            bwd_scan_reference_adapter("grad_delta"),
        )
        assert not result.passed, "saturation probe lost for the backward op"

    def test_cumprod_trick_bwd_cheat_caught_by_ord03(self) -> None:
        # The ratio-cumsum 'parallelisation' underflows its decay products at
        # ORD-03's L=8192 and the gradients NaN out, while honest reorder
        # noise stays within the view tolerance.
        kwargs = dict(SCAN_BWD_GATE_OVERRIDES["grad_u"]["gate_ord_03_noncommutative_reduction"])
        result = gate_ord_03_noncommutative_reduction(
            bwd_scan_candidate_adapter(_bwd_via_autograd(_cumprod_trick_scan), "grad_u"),
            bwd_scan_reference_adapter("grad_u"),
            **kwargs,
        )
        assert not result.passed, "ORD-03 lost its discriminative power for the backward op"
