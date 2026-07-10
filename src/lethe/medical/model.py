"""Mamba-3 ECG classifier for PTB-XL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from lethe.kernels.ops import fused_block_forward

__all__ = ["Mamba3Config", "Mamba3ECGClassifier"]


@dataclass(frozen=True)
class Mamba3Config:
    """Configuration for :class:`Mamba3ECGClassifier`."""

    d_model: int = 512
    n_layers: int = 4
    d_state: int = 16
    conv_kernel_size: int = 4
    chunk_size: int = 64
    eps: float = 1e-5
    n_classes: int = 5
    n_leads: int = 12
    # Regularization (#19): dropout on each block's output and the pooled representation before the head.
    dropout: float = 0.0

    @classmethod
    def tiny(cls) -> Mamba3Config:
        """CPU-testable config; small enough to forward in seconds."""
        return cls(d_model=32, n_layers=2, d_state=8, conv_kernel_size=4, chunk_size=8)

    @classmethod
    def b_mid(cls) -> Mamba3Config:
        """~160M config (#19): 1.1B b1 over-parameterizes ~17k labels (val loss rose at step 2500)."""
        return cls(
            d_model=2048, n_layers=12, d_state=64, conv_kernel_size=4, chunk_size=64, dropout=0.2
        )

    @classmethod
    def b1(cls) -> Mamba3Config:
        """~1.10B parameter config for PTB-XL training on the B200 cluster."""
        return cls(d_model=4096, n_layers=32, d_state=64, conv_kernel_size=4, chunk_size=64)

    def analytic_param_count(self) -> int:
        """Exact formula matching _MambaBlock + Mamba3ECGClassifier parameter layout."""
        D, N, K = self.d_model, self.d_state, self.conv_kernel_size
        lead_proj = D * (self.n_leads + 1)  # weight + bias
        # Per block: in_proj D*(2D+2N) + conv_weight D*K + A_log D*N + 3*D (conv_bias/D_skip/norm_weight).
        per_block = D * (2 * D + 2 * N) + D * K + D * N + 3 * D
        head = D * self.n_classes + self.n_classes
        return lead_proj + self.n_layers * per_block + head


class _MambaBlock(nn.Module):
    """Single Mamba-3 SISO block: in_proj → conv+SiLU+scan+RMSNorm → residual add."""

    def __init__(self, cfg: Mamba3Config) -> None:
        super().__init__()
        D, N, K = cfg.d_model, cfg.d_state, cfg.conv_kernel_size
        self.cfg = cfg
        # data-dependent projections: x_raw, delta, B, C
        self.in_proj = nn.Linear(D, 2 * D + 2 * N, bias=False)
        # depthwise conv: [D, 1, K]
        self.conv_weight = nn.Parameter(torch.empty(D, 1, K))
        self.conv_bias = nn.Parameter(torch.zeros(D))
        # A = -exp(A_log) (official Mamba convention).
        a_log = torch.log(torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(0).expand(D, N))
        self.A_log = nn.Parameter(a_log.contiguous())
        self.D_skip = nn.Parameter(torch.ones(D))
        self.norm_weight = nn.Parameter(torch.ones(D))
        self.drop = nn.Dropout(cfg.dropout)
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

        # Slice to L_out for the SSM projections: conv removes K-1 steps
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

        # Residual: add input at L_out positions (strip the pad from input too).
        return cast(Tensor, self.drop(y) + x[:, K - 1 :, :])


class Mamba3ECGClassifier(nn.Module):
    """1.1B Mamba-1 SISO multi-label ECG classifier for PTB-XL (Mamba-3 MIMO block pending)."""

    def __init__(self, cfg: Mamba3Config | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = Mamba3Config()
        self.cfg = cfg
        D = cfg.d_model
        self.lead_proj = nn.Linear(cfg.n_leads, D)
        self.blocks = nn.ModuleList([_MambaBlock(cfg) for _ in range(cfg.n_layers)])
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(D, cfg.n_classes)

    def forward(self, ecg: Tensor) -> Tensor:
        """ecg: [B, 12, T] float32. Returns [B, n_classes] logits."""
        K = self.cfg.conv_kernel_size
        x = ecg.transpose(1, 2)  # [B, T, 12]
        x = self.lead_proj(x)  # [B, T, D]

        # Causal left-pad once: each block consumes L = T + K - 1 and trims back to T via valid conv.
        for block in self.blocks:
            pad = x.new_zeros(x.shape[0], K - 1, x.shape[2])
            x = block(torch.cat([pad, x], dim=1))  # [B, T, D] each iteration

        pooled = self.dropout(x.mean(dim=1))  # [B, D]
        return cast(Tensor, self.head(pooled))  # [B, n_classes]

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
