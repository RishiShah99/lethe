"""SFT warm-start targets: token screen, registry coverage, reference parity.

The targets mirror the references statement-for-statement, so fp32 parity
is asserted bitwise wherever the op graph is identical (C1-C5); the fused
backward replaces the conv primitive with a shifted sum (determinism by
construction on CUDA) and is asserted at tight tolerance instead. The
full gate battery runs once on CPU for the cheapest op to pin the whole
score_candidate_source path; all six run on CUDA via scratch/sft_validate.py
before any SFT step consumes them.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from flash_mamba_rl.kernels.references.backward_selective_scan import (
    reference_backward_selective_scan,
)
from flash_mamba_rl.kernels.references.complex_scan_rope import reference_complex_scan_rope
from flash_mamba_rl.kernels.references.forward_chunked_scan import reference_forward_chunked_scan
from flash_mamba_rl.kernels.references.fused_block_backward import reference_fused_block_backward
from flash_mamba_rl.kernels.references.fused_block_forward import reference_fused_block_forward
from flash_mamba_rl.kernels.references.mimo_backward import reference_mimo_backward
from flash_mamba_rl.rl.sft_targets import available_targets, target_source, target_variants
from flash_mamba_rl.rl.sft_targets.backward_selective_scan import backward_selective_scan
from flash_mamba_rl.rl.sft_targets.complex_scan_rope import complex_scan_rope
from flash_mamba_rl.rl.sft_targets.forward_chunked_scan import forward_chunked_scan
from flash_mamba_rl.rl.sft_targets.fused_block_backward import fused_block_backward
from flash_mamba_rl.rl.sft_targets.fused_block_forward import fused_block_forward
from flash_mamba_rl.rl.sft_targets.mimo_backward import mimo_backward
from flash_mamba_rl.verifier.candidate_scoring import (
    FORBIDDEN_SOURCE_TOKENS,
    score_candidate_source,
    scoreable_ops,
)


def _scan_inputs(
    batch: int = 2, seq_len: int = 64, d_model: int = 8, n_state: int = 4
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(7)
    u = torch.randn(batch, seq_len, d_model, generator=gen)
    delta = torch.randn(batch, seq_len, d_model, generator=gen)
    a = -torch.rand(d_model, n_state, generator=gen) - 0.1
    b = torch.randn(batch, seq_len, n_state, generator=gen)
    c = torch.randn(batch, seq_len, n_state, generator=gen)
    d = torch.randn(d_model, generator=gen)
    return u, delta, a, b, c, d


def _fused_inputs(
    batch: int = 2,
    l_out: int = 32,
    d_model: int = 8,
    n_state: int = 4,
    conv_k: int = 4,
) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(11)
    x = torch.randn(batch, l_out + conv_k - 1, d_model, generator=gen)
    conv_w = torch.randn(d_model, 1, conv_k, generator=gen)
    conv_b = torch.randn(d_model, generator=gen)
    delta = torch.randn(batch, l_out, d_model, generator=gen)
    a = -torch.rand(d_model, n_state, generator=gen) - 0.1
    b = torch.randn(batch, l_out, n_state, generator=gen)
    c = torch.randn(batch, l_out, n_state, generator=gen)
    d = torch.randn(d_model, generator=gen)
    norm_w = torch.randn(d_model, generator=gen)
    return x, conv_w, conv_b, delta, a, b, c, d, norm_w


def test_registry_covers_curriculum_ops() -> None:
    expected = set(scoreable_ops()) - {"elementwise_silu"}
    assert set(available_targets()) == expected
    for op in available_targets():
        assert target_variants(op) == ("eager", "triton")
        for variant in target_variants(op):
            assert f"def {op}(" in target_source(op, variant)


def test_no_forbidden_tokens() -> None:
    for op in available_targets():
        for variant in target_variants(op):
            source = target_source(op, variant)
            hits = [tok for tok in FORBIDDEN_SOURCE_TOKENS if tok in source]
            assert not hits, f"{op}[{variant}]: {hits}"


@pytest.fixture()
def stub_triton() -> Iterator[None]:
    """Importable stand-in for triton so the CUDA targets load on CPU.

    Only module-level surface is needed: @triton.jit wraps (never runs),
    tl.constexpr is called once for a module constant, libdevice is an
    attribute. Annotations stay strings (future import in every target).
    Stubs are removed afterwards so the kernels' find_spec dispatch is
    unaffected for the rest of the session.
    """
    if importlib.util.find_spec("triton") is not None:
        yield
        return
    triton_mod = types.ModuleType("triton")
    tl_mod = types.ModuleType("triton.language")
    extra_mod = types.ModuleType("triton.language.extra")
    extra_mod.libdevice = types.SimpleNamespace()  # type: ignore[attr-defined]
    tl_mod.constexpr = lambda v: v  # type: ignore[attr-defined]
    tl_mod.extra = extra_mod  # type: ignore[attr-defined]
    triton_mod.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)  # type: ignore[attr-defined]
    triton_mod.cdiv = lambda a, b: -(-a // b)  # type: ignore[attr-defined]
    triton_mod.next_power_of_2 = lambda n: 1 << max(0, (int(n) - 1).bit_length())  # type: ignore[attr-defined]
    triton_mod.language = tl_mod  # type: ignore[attr-defined]
    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = tl_mod
    sys.modules["triton.language.extra"] = extra_mod
    try:
        yield
    finally:
        for name in ("triton", "triton.language", "triton.language.extra"):
            sys.modules.pop(name, None)


def _load_triton_target(op: str) -> Any:
    import flash_mamba_rl.rl.sft_targets as pkg

    path = Path(pkg.__file__).parent / f"{op}_triton.py"
    spec = importlib.util.spec_from_file_location(f"_sft_tgt_{op}_triton", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTritonVariantCpuFallback:
    """RES-01 always probes a CPU leg; the CUDA targets must dispatch to
    their eager fallback there and match the reference."""

    def test_c1_cpu_fallback_bitwise(self, stub_triton: None) -> None:
        mod = _load_triton_target("forward_chunked_scan")
        inputs = _scan_inputs()
        want = reference_forward_chunked_scan(*inputs, chunk_size=64)
        assert torch.equal(mod.forward_chunked_scan(*inputs), want)

    def test_c2_cpu_fallback_bitwise(self, stub_triton: None) -> None:
        mod = _load_triton_target("backward_selective_scan")
        inputs = _scan_inputs()
        dy = torch.randn(2, 64, 8, generator=torch.Generator().manual_seed(29))
        want = reference_backward_selective_scan(*inputs, dy, chunk_size=64)
        got = mod.backward_selective_scan(*inputs, dy)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)

    def test_c3_cpu_fallback_bitwise(self, stub_triton: None) -> None:
        mod = _load_triton_target("mimo_backward")
        gen = torch.Generator().manual_seed(17)
        x = torch.randn(2, 8, 2, 3, generator=gen)
        b = torch.randn(2, 8, 2, 2, 4, generator=gen)
        c = torch.randn(2, 8, 2, 2, 4, generator=gen)
        dt = torch.rand(2, 8, 2, generator=gen) * 0.1 + 1e-3
        alpha = torch.exp(dt * -torch.rand(2, generator=gen))
        mimo_x = 0.5 + torch.randn(2, 2, 3, generator=gen) * 0.1
        mimo_o = 0.5 + torch.randn(2, 2, 3, generator=gen) * 0.1
        dy = torch.randn(2, 8, 2, 3, generator=gen)
        want = reference_mimo_backward(x, b, c, dt, alpha, mimo_x, mimo_o, dy)
        got = mod.mimo_backward(x, b, c, dt, alpha, mimo_x, mimo_o, dy)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=0.0, atol=0.0)

    def test_c4_cpu_fallback_bitwise(self, stub_triton: None) -> None:
        mod = _load_triton_target("complex_scan_rope")
        gen = torch.Generator().manual_seed(19)
        x = torch.randn(2, 12, 2, 4, generator=gen)
        b = torch.randn(2, 12, 2, 8, generator=gen)
        c = torch.randn(2, 12, 2, 8, generator=gen)
        dt = torch.rand(2, 12, 2, generator=gen) * 0.1 + 1e-3
        a = -torch.rand(2, generator=gen)
        angle_proj = torch.randn(2, 12, 2, 3, generator=gen)
        want = reference_complex_scan_rope(x, b, c, dt, a, angle_proj)
        assert torch.equal(mod.complex_scan_rope(x, b, c, dt, a, angle_proj), want)

    def test_c5_cpu_fallback_bitwise(self, stub_triton: None) -> None:
        mod = _load_triton_target("fused_block_forward")
        inputs = _fused_inputs()
        want = reference_fused_block_forward(*inputs, conv_kernel_size=4, chunk_size=16)
        got = mod.fused_block_forward(*inputs, conv_kernel_size=4, chunk_size=16)
        assert torch.equal(got, want)

    def test_c6_cpu_fallback_close(self, stub_triton: None) -> None:
        mod = _load_triton_target("fused_block_backward")
        inputs = _fused_inputs()
        dy = torch.randn(2, 32, 8, generator=torch.Generator().manual_seed(31))
        want = reference_fused_block_backward(*inputs, dy, conv_kernel_size=4, chunk_size=16)
        got = mod.fused_block_backward(*inputs, dy, conv_kernel_size=4, chunk_size=16)
        assert len(got) == 9
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)


def test_triton_variants_parse_and_self_contained() -> None:
    # Triton is absent on the dev box, so the CUDA targets are pinned at the
    # source level here: they must parse and import only torch/triton/math.
    import ast

    for op in available_targets():
        source = target_source(op, "triton")
        tree = ast.parse(source)
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots <= {"torch", "triton", "math", "__future__"}, f"{op}: {roots}"


def test_c1_matches_reference_bitwise() -> None:
    inputs = _scan_inputs()
    got = forward_chunked_scan(*inputs)
    want = reference_forward_chunked_scan(*inputs, chunk_size=64)
    assert torch.equal(got, want)


def test_c1_half_round_once_convention() -> None:
    inputs = tuple(t.to(torch.float16) for t in _scan_inputs())
    got = forward_chunked_scan(*inputs)
    want = reference_forward_chunked_scan(*(t.to(torch.float32) for t in inputs), chunk_size=64).to(
        torch.float16
    )
    assert got.dtype == torch.float16
    assert torch.equal(got, want)


def test_c2_matches_autograd_bitwise_including_nonfinite_dy() -> None:
    inputs = _scan_inputs()
    gen = torch.Generator().manual_seed(13)
    dy = torch.randn(2, 64, 8, generator=gen)
    dy[0, 3, 2] = float("nan")
    dy[1, 10, 5] = float("inf")
    got = backward_selective_scan(*inputs, dy)
    want = reference_backward_selective_scan(*inputs, dy, chunk_size=64)
    assert len(got) == 6
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, rtol=0.0, atol=0.0, equal_nan=True)


def test_c3_matches_autograd_bitwise() -> None:
    gen = torch.Generator().manual_seed(17)
    batch, seq_len, nheads, headdim, rank, n_state = 2, 8, 2, 3, 2, 4
    x = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    b = torch.randn(batch, seq_len, rank, nheads, n_state, generator=gen)
    c = torch.randn(batch, seq_len, rank, nheads, n_state, generator=gen)
    dt = torch.rand(batch, seq_len, nheads, generator=gen) * 0.1 + 1e-3
    alpha = torch.exp(dt * -torch.rand(nheads, generator=gen))
    mimo_x = 0.5 + torch.randn(nheads, rank, headdim, generator=gen) * 0.1
    mimo_o = 0.5 + torch.randn(nheads, rank, headdim, generator=gen) * 0.1
    dy = torch.randn(batch, seq_len, nheads, headdim, generator=gen)

    got = mimo_backward(x, b, c, dt, alpha, mimo_x, mimo_o, dy)
    want = reference_mimo_backward(x, b, c, dt, alpha, mimo_x, mimo_o, dy)
    assert len(got) == 7
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, rtol=0.0, atol=0.0, equal_nan=True)


def test_c4_matches_reference_bitwise() -> None:
    gen = torch.Generator().manual_seed(19)
    batch, seq_len, nheads, headdim, n_state, num_rope = 2, 12, 2, 4, 8, 3
    x = torch.randn(batch, seq_len, nheads, headdim, generator=gen)
    b = torch.randn(batch, seq_len, nheads, n_state, generator=gen)
    c = torch.randn(batch, seq_len, nheads, n_state, generator=gen)
    dt = torch.rand(batch, seq_len, nheads, generator=gen) * 0.1 + 1e-3
    a = -torch.rand(nheads, generator=gen)
    angle_proj = torch.randn(batch, seq_len, nheads, num_rope, generator=gen)

    got = complex_scan_rope(x, b, c, dt, a, angle_proj)
    want = reference_complex_scan_rope(x, b, c, dt, a, angle_proj)
    assert torch.equal(got, want)


def test_c5_matches_reference_bitwise() -> None:
    inputs = _fused_inputs()
    got = fused_block_forward(*inputs, conv_kernel_size=4, chunk_size=16)
    want = reference_fused_block_forward(*inputs, conv_kernel_size=4, chunk_size=16)
    assert torch.equal(got, want)


def test_c6_matches_autograd_including_nonfinite_dy() -> None:
    # The target's shifted-sum conv reorders the K-term accumulation vs
    # F.conv1d, so parity is tight-tolerance rather than bitwise.
    inputs = _fused_inputs()
    gen = torch.Generator().manual_seed(23)
    dy = torch.randn(2, 32, 8, generator=gen)
    dy[0, 5, 1] = float("nan")
    dy[1, 20, 7] = float("-inf")
    got = fused_block_backward(*inputs, dy, conv_kernel_size=4, chunk_size=16)
    want = reference_fused_block_backward(*inputs, dy, conv_kernel_size=4, chunk_size=16)
    assert len(got) == 9
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6, equal_nan=True)


def test_c6_half_grads_match_dtype() -> None:
    inputs = tuple(t.to(torch.bfloat16) for t in _fused_inputs())
    dy = torch.randn(2, 32, 8).to(torch.bfloat16)
    got = fused_block_backward(*inputs, dy, conv_kernel_size=4, chunk_size=16)
    assert all(g.dtype == torch.bfloat16 for g in got)


def test_c1_target_passes_full_battery() -> None:
    result = score_candidate_source(
        target_source("forward_chunked_scan"), op="forward_chunked_scan", device="cpu"
    )
    failed = {g: r for g, r in result["gates"].items() if not r["passed"]}
    assert result["contracts_passed"] is True, (result["status"], result["error"], failed)
    assert result["reward"] == 0.5
