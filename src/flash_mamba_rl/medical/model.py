"""Mamba-3 ECG classifier for PTB-XL (Phase F.2).

Architecture decisions (resolved here, not invented — see docs/mamba3_math_resolution.md):

SISO variant: each Mamba-3 block uses the SISO selective scan (the same recurrence
that reference_fused_block_forward implements) rather than the full MIMO variant.
Rationale: fused_block_forward dispatches to the SISO kernel; MIMO requires mimo_x /
mimo_o mixing weights and a headdim/nheads factoring that is outside this op's
contract. The full MIMO tower is the natural v2 once the MIMO forward op is exposed
as a block-level primitive.

in_proj split: each block projects its residual input [B, L, D] via a single
nn.Linear to [B, L, 2*D + 2*N]. The four slices are:
  x_raw [B, L, D]   — fed to conv + SiLU + scan (the 'u' stream)
  delta  [B, L, D]  — SSM timescale input (data-dependent, pre-softplus; softplus
                       is inside fused_block_forward via the scan)
  B      [B, L, N]  — SSM input projection (data-dependent)
  C      [B, L, N]  — SSM output projection (data-dependent)

Residual convention: pre-norm residual — RMSNorm is applied inside fused_block_forward
on the scan output; the block adds the normed output to its input (post-norm residual
would require a separate norm layer outside the op, conflicting with the fused design).

Pooling: mean-pool over the output sequence dimension L_out → [B, D] before the
linear head. Alternative (last-token) would be sensitive to the sequence-end
boundary after left-padding; mean-pool is the standard choice for multi-label
classification on variable-length ECGs.

Parameter count formula (b1 config: d_model=4096, n_layers=32, d_state=64, K=4):
  lead_proj   = D*(12+1)                                        =   53 248
  per_block   = D*(2D+2N) + D*K + D + D*N + 3*D               ≈ 34 373 632
  n_layers*block                                                 = 1 099 956 224
  head        = D*n_classes + n_classes                         =   20 485
  total                                                          ≈ 1 100 029 957  (~1.10B)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from flash_mamba_rl.kernels.ops import fused_block_forward

__all__ = ["Mamba3Config", "Mamba3ECGClassifier"]


@dataclass(frozen=True)
class Mamba3Config:
    """Configuration for :class:`Mamba3ECGClassifier`.

    Parameters
    ----------
    d_model:
        Width of the model's residual stream (= D in the kernel ops).
    n_layers:
        Number of stacked Mamba-3 blocks.
    d_state:
        SSM state dimension N. Controls per-step memory capacity.
    conv_kernel_size:
        Depthwise conv kernel width K. Must match fused_block_forward's contract.
    chunk_size:
        Passed to fused_block_forward; L_out must be divisible by this.
    eps:
        RMSNorm epsilon inside fused_block_forward.
    n_classes:
        Output classes (5 for PTB-XL superclass, 23 for subclass).
    n_leads:
        ECG input channels (12 for standard 12-lead).
    """

    d_model: int = 512
    n_layers: int = 4
    d_state: int = 16
    conv_kernel_size: int = 4
    chunk_size: int = 64
    eps: float = 1e-5
    n_classes: int = 5
    n_leads: int = 12

    @classmethod
    def tiny(cls) -> Mamba3Config:
        """CPU-testable config; small enough to forward in seconds."""
        return cls(d_model=32, n_layers=2, d_state=8, conv_kernel_size=4, chunk_size=8)

    @classmethod
    def b1(cls) -> Mamba3Config:
        """~1.10B parameter config for PTB-XL training on the B200 cluster.

        Analytic count (see module docstring):
          lead_proj  =  D*(n_leads+1)                             =     53_248
          per_block  =  D*(2D+2N) + D*K + D + D*N + 3*D          = 34_373_632
          n_layers*block                                           = 1_099_956_224
          head       =  D*n_classes + n_classes                   =     20_485
          total                                                    ~ 1_100_029_957
        """
        return cls(d_model=4096, n_layers=32, d_state=64, conv_kernel_size=4, chunk_size=64)

    def analytic_param_count(self) -> int:
        """Exact formula matching _MambaBlock + Mamba3ECGClassifier parameter layout."""
        D, N, K = self.d_model, self.d_state, self.conv_kernel_size
        lead_proj = D * (self.n_leads + 1)  # weight + bias
        # per block: in_proj (no bias, weight only) + conv_weight + conv_bias
        #            + A (not a param; A = -exp(A_log), A_log IS a param)
        #            + D_skip + norm_weight + pre_norm (3 D-shaped params)
        #   in_proj weight:  D * (2D + 2N)
        #   conv_weight:     D * K
        #   A_log (-> A):    D * N
        #   conv_bias + D_skip + norm_weight: 3 * D
        per_block = D * (2 * D + 2 * N) + D * K + D * N + 3 * D
        head = D * self.n_classes + self.n_classes
        return lead_proj + self.n_layers * per_block + head


class _MambaBlock(nn.Module):
    """Single Mamba-3 SISO block: in_proj → conv+SiLU+scan+RMSNorm → residual add.

    Parameters produced here match the fused_block_forward signature exactly.
    The causal K-1 left-pad is applied here before the op; L_out = L - (K-1)
    is returned, so callers accumulate the pad deficit across blocks.

    Decision: all blocks share the same L_out (pad is only added once at the
    network's entry, not per block). The in_proj projection operates on L
    (padded length) but only the L_out slice of its delta/B/C outputs is
    passed to the op — matching the reference convention where delta/B/C
    have shape [B, L_out, *].
    """

    def __init__(self, cfg: Mamba3Config) -> None:
        super().__init__()
        D, N, K = cfg.d_model, cfg.d_state, cfg.conv_kernel_size
        self.cfg = cfg
        # data-dependent projections: x_raw, delta, B, C
        self.in_proj = nn.Linear(D, 2 * D + 2 * N, bias=False)
        # depthwise conv: [D, 1, K]
        self.conv_weight = nn.Parameter(torch.empty(D, 1, K))
        self.conv_bias = nn.Parameter(torch.zeros(D))
        # A = -exp(A_log) (official Mamba convention). A is then strictly negative
        # for ANY value of A_log, so a_bar = exp(delta*A) stays in (0,1) and the scan
        # state cannot integrate/explode over long L. (The prior log_A-as-A init left
        # the j=0 column at A=0 — a pure integrator that drifts positive under training
        # and overflows the L=1000 scan into NaN.) S4D-real init: A = -(j+1).
        a_log = torch.log(torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).expand(D, N))
        self.A_log = nn.Parameter(a_log.contiguous())
        self.D_skip = nn.Parameter(torch.ones(D))
        self.norm_weight = nn.Parameter(torch.ones(D))
        self._reset_conv()

    def _reset_conv(self) -> None:
        K = self.cfg.conv_kernel_size
        D = self.cfg.d_model
        # Kaiming uniform analogous to nn.Conv1d default init
        nn.init.kaiming_uniform_(self.conv_weight.view(D, K), a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, D] (L already includes K-1 left-pad). Returns [B, L_out, D]."""
        cfg = self.cfg
        K = cfg.conv_kernel_size

        # in_proj over the full (padded) sequence
        proj = self.in_proj(x)  # [B, L, 2D+2N]

        D = cfg.d_model
        N = cfg.d_state
        x_raw = proj[..., :D]  # [B, L, D]
        delta = proj[..., D : 2 * D]  # [B, L, D]  (pre-softplus timescale)
        B_proj = proj[..., 2 * D : 2 * D + N]  # [B, L, N]
        C_proj = proj[..., 2 * D + N :]  # [B, L, N]

        # Slice to L_out for the SSM projections — conv removes K-1 steps
        delta_out = delta[:, K - 1 :, :]  # [B, L_out, D]
        B_out = B_proj[:, K - 1 :, :]  # [B, L_out, N]
        C_out = C_proj[:, K - 1 :, :]  # [B, L_out, N]

        A = -torch.exp(self.A_log)  # [D, N], strictly negative (decay) for any A_log

        y = fused_block_forward(
            x_raw,
            self.conv_weight,
            self.conv_bias,
            delta_out,
            A,
            B_out,
            C_out,
            self.D_skip,
            self.norm_weight,
            conv_kernel_size=K,
            eps=cfg.eps,
            chunk_size=cfg.chunk_size,
        )  # [B, L_out, D]

        # Residual: add input at L_out positions (strip the pad from input too)
        return y + x[:, K - 1 :, :]


class Mamba3ECGClassifier(nn.Module):
    """1B Mamba-3 multi-label ECG classifier for PTB-XL.

    Input:  [B, 12, T]  float32  (loader's native layout)
    Output: [B, n_classes]  float32  logits (no activation — BCEWithLogitsLoss)

    Forward steps:
      1. Transpose to [B, T, 12] and project leads → d_model.
      2. Prepend K-1 zeros (causal left-pad) → [B, T+K-1, d_model].
      3. Pass through n_layers Mamba-3 blocks; each returns [B, L_out, D]
         where L_out = (T+K-1) - (K-1) = T (only the first block removes the
         pad — subsequent blocks receive already-trimmed sequences and must NOT
         re-pad). The design threads T through unchanged after block 0.
      4. Mean-pool over L_out → [B, D].
      5. Linear → [B, n_classes] logits.

    IMPORTANT: input T (after conv trim) must satisfy T % chunk_size == 0.
    The caller is responsible for picking T accordingly (e.g. T=1000 for 100 Hz,
    chunk_size=8 or chunk_size=40 both divide it).
    """

    def __init__(self, cfg: Mamba3Config | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = Mamba3Config()
        self.cfg = cfg
        D = cfg.d_model
        self.lead_proj = nn.Linear(cfg.n_leads, D)
        self.blocks = nn.ModuleList([_MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.head = nn.Linear(D, cfg.n_classes)

    def forward(self, ecg: Tensor) -> Tensor:
        """ecg: [B, 12, T] float32. Returns [B, n_classes] logits."""
        K = self.cfg.conv_kernel_size
        x = ecg.transpose(1, 2)  # [B, T, 12]
        x = self.lead_proj(x)  # [B, T, D]

        # Causal left-pad once — all blocks consume L = T + K - 1 on entry,
        # and each block trims back to T by the valid convolution.
        # After block 0 the sequence is T again; subsequent blocks receive T
        # and pad again, trimming back to T - (K-1) each time. To avoid
        # shrinkage after block 0, we pad before EVERY block, not just the first.
        # This is consistent with the reference convention (caller pads, op strips).
        for block in self.blocks:
            pad = x.new_zeros(x.shape[0], K - 1, x.shape[2])
            x = block(torch.cat([pad, x], dim=1))  # [B, T, D] each iteration

        pooled = x.mean(dim=1)  # [B, D]
        return cast(Tensor, self.head(pooled))  # [B, n_classes]

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
