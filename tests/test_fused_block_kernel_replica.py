"""CPU replica of the C5 Triton kernels' exact algorithm vs the reference.

The fused path computes the same math as the reference through a different
route: the conv as a per-step K-column window dot riding the scan's serial
loop (reference: one F.conv1d over the whole sequence), SiLU inline as
x * 1/(1+exp(-x)), the scan as C1's register recurrence, and the RMSNorm
as a chunked in-program sum of squares (reference: torch mean). This
replica mirrors both kernels statement-for-statement so any algebra
divergence — window indexing, correlation-vs-convolution orientation,
bias placement, ssq chunk boundary — surfaces on CPU without a GPU in the
loop. Tolerances are fp32 reorder noise, not correctness slack.
"""

from __future__ import annotations

import math

import torch

from flash_mamba_rl.kernels.ops.fused_block_forward import _fused_eager
from flash_mamba_rl.kernels.references import reference_fused_block_forward


def _conv_scan_replica(
    x: torch.Tensor,
    conv_w: torch.Tensor,
    conv_b: torch.Tensor,
    delta: torch.Tensor,
    a_mat: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
    d_skip: torch.Tensor,
) -> torch.Tensor:
    """Kernel A's compute order, statement for statement, in torch."""
    batch, seq_in, d_model = x.shape
    conv_k = conv_w.shape[-1]
    l_out = seq_in - (conv_k - 1)
    n_state = a_mat.shape[1]
    w = conv_w[:, 0, :]

    h = torch.zeros(batch, d_model, n_state, dtype=x.dtype)
    y_scan = torch.empty(batch, l_out, d_model, dtype=x.dtype)
    for t in range(l_out):
        xw = x[:, t : t + conv_k, :].permute(0, 2, 1)
        conv = (w.unsqueeze(0) * xw).sum(-1) + conv_b
        z = conv * (1.0 / (1.0 + torch.exp(-conv)))

        dlt = delta[:, t]
        dbar = torch.where(dlt > 20.0, dlt, torch.log1p(torch.exp(dlt)))
        a_bar = torch.exp(dbar.unsqueeze(-1) * a_mat.unsqueeze(0))
        bu = (dbar * z).unsqueeze(-1) * b_proj[:, t].unsqueeze(1)
        h = a_bar * h + bu
        y_scan[:, t] = (h * c_proj[:, t].unsqueeze(1)).sum(-1) + d_skip * z
    return y_scan


def _rmsnorm_replica(
    y_scan: torch.Tensor, norm_w: torch.Tensor, eps: float, block_d: int
) -> torch.Tensor:
    """Kernel B's chunked sum-of-squares order in torch."""
    _batch, _l_out, d_model = y_scan.shape
    ssq = torch.zeros(y_scan.shape[:2], dtype=y_scan.dtype)
    for d0 in range(0, d_model, block_d):
        v = y_scan[:, :, d0 : d0 + block_d]
        ssq = ssq + (v * v).sum(-1)
    rms = torch.sqrt(ssq / d_model + eps)
    return y_scan / rms.unsqueeze(-1) * norm_w


def _inputs(
    b: int, l_out: int, d: int, n: int, k: int, seed: int, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)

    def mk(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=dtype)

    x = mk(b, l_out + k - 1, d)
    conv_w = mk(d, 1, k) / math.sqrt(k)
    conv_b = 0.5 * mk(d)
    delta = mk(b, l_out, d)
    a_mat = -torch.rand(d, n, dtype=dtype)
    b_proj = mk(b, l_out, n)
    c_proj = mk(b, l_out, n)
    d_skip = mk(d)
    norm_w = 1.0 + 0.25 * mk(d)
    return x, conv_w, conv_b, delta, a_mat, b_proj, c_proj, d_skip, norm_w


class TestKernelReplicaParity:
    def _check(
        self,
        b: int,
        l_out: int,
        d: int,
        n: int,
        k: int,
        seed: int = 0,
        block_d: int = 4,
        saturate: bool = False,
    ) -> None:
        args = _inputs(b, l_out, d, n, k, seed)
        if saturate:
            args[3].view(-1)[::7] = 25.0
            args[3].view(-1)[::11] = 95.0
        y_scan = _conv_scan_replica(*args[:8])
        got = _rmsnorm_replica(y_scan, args[8], 1e-5, block_d)
        want = reference_fused_block_forward(*args, conv_kernel_size=k, chunk_size=l_out)
        max_err = (got - want).abs().max().item()
        scale = want.abs().max().clamp(min=1.0).item()
        assert max_err / scale < 1e-5, f"replica diverges: scale_rel={max_err / scale:.3e}"

    def test_standard_window(self) -> None:
        self._check(2, 64, 12, 16, 4)

    def test_window_two(self) -> None:
        self._check(2, 48, 8, 16, 2)

    def test_window_one_degenerate(self) -> None:
        self._check(1, 32, 6, 8, 1)

    def test_window_at_budget(self) -> None:
        # K=8 is MAX_CONV_K, the widest window the launcher admits.
        self._check(1, 32, 6, 8, 8)

    def test_single_output_step(self) -> None:
        self._check(1, 1, 8, 16, 4)

    def test_ssq_chunk_tail(self) -> None:
        # d_model=10 with block_d=4: the chunked sum-of-squares runs a
        # masked tail chunk, the boundary the Triton norm kernel has.
        self._check(2, 32, 10, 8, 4, block_d=4)

    def test_saturated_softplus_branch(self) -> None:
        self._check(2, 32, 8, 8, 4, saturate=True)

    def test_long_sequence_decay(self) -> None:
        self._check(1, 512, 6, 16, 4, seed=7)

    def test_fp64_tight(self) -> None:
        # The reference rejects fp64; _fused_eager is its documented
        # op-for-op fp64 extension (bitwise-equal to the reference on fp32,
        # pinned in test_fused_block_op).
        args = _inputs(2, 64, 12, 16, 4, seed=11, dtype=torch.float64)
        y_scan = _conv_scan_replica(*args[:8])
        got = _rmsnorm_replica(y_scan, args[8], 1e-5, 4)
        want = _fused_eager(*args, eps=1e-5)
        max_err = (got - want).abs().max().item()
        assert max_err < 1e-12, f"fp64 replica diverges: {max_err:.3e}"
