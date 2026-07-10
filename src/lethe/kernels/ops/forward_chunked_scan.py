"""C1, hand-written SISO selective-scan forward."""

from __future__ import annotations

from importlib import util as _importlib_util
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from lethe.kernels.autotune import KernelConfig

from ._resource_meta import max_resource_meta

_TRITON_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
_HAS_TRITON: bool | None = None


def _triton_usable() -> bool:
    global _HAS_TRITON
    if _HAS_TRITON is None:
        _HAS_TRITON = _importlib_util.find_spec("triton") is not None
    return _HAS_TRITON


_CHUNK_PARALLEL_CAP = 512
_BWD_TILE_BUDGET = 2048
_MAX_BLOCK_N = 128


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n. Mirrors triton.next_power_of_2."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _chunk_parallel_bwd_scratch_bytes(
    batch: int, seq_len: int, d_model: int, n_state: int, chunk_len: int
) -> int:
    """Compute the fp32 scratch buffer size (bytes) for chunk-parallel backward."""
    block_n = _next_power_of_2(n_state)
    if block_n > _MAX_BLOCK_N:
        block_n = _MAX_BLOCK_N
    block_d = min(64, max(16, _BWD_TILE_BUDGET // block_n))
    n_chunks = seq_len // chunk_len if chunk_len > 0 else 1
    n_d_blocks = (d_model + block_d - 1) // block_d
    hbuf_elems = batch * n_chunks * n_d_blocks * chunk_len * block_d * block_n
    return hbuf_elems * 4


def _auto_chunk_len(seq_len: int, requested: int | None) -> int:
    """Largest divisor of ``seq_len`` not exceeding ``requested`` (or the cap)."""
    target = min(requested if requested is not None else _CHUNK_PARALLEL_CAP, seq_len)
    for k in range(target, 0, -1):
        if seq_len % k == 0:
            return k
    return 1


def _default_scan_mode(
    seq_len: int,
    batch: int,
    width: int,
    *,
    is_forward: bool,
    n_state: int | None = None,
    device: torch.device | None = None,
) -> str:
    """Launch-default scan mode for a shape, consulted only when scan_mode is unset."""
    if is_forward and seq_len <= 512:
        return "serial"
    if batch >= 8 and width >= 4096 and seq_len <= 4096:
        return "serial"
    if batch >= 8 and width >= 2048 and seq_len <= 512:
        return "serial"
    if not is_forward and n_state is not None:
        if n_state > _MAX_BLOCK_N:
            return "serial"
        if torch.cuda.is_available():
            chunk_len = _auto_chunk_len(seq_len, None)
            scratch = _chunk_parallel_bwd_scratch_bytes(batch, seq_len, width, n_state, chunk_len)
            cuda_device = device if device is not None and device.type == "cuda" else None
            try:
                free_bytes, _ = torch.cuda.mem_get_info(cuda_device)
            except RuntimeError:
                free_bytes = 0
            if scratch > 0.7 * free_bytes:
                return "serial"
    return "chunk_parallel"


def _resolve_scan_mode(
    config: KernelConfig | None,
    seq_len: int,
    batch: int,
    width: int,
    *,
    is_forward: bool,
    n_state: int | None = None,
    device: torch.device | None = None,
) -> str:
    """The effective scan mode: an explicit config knob, else the shape default."""
    if config is not None and config.scan_mode is not None:
        return config.scan_mode
    return _default_scan_mode(
        seq_len, batch, width, is_forward=is_forward, n_state=n_state, device=device
    )


def _scan_eager(u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor) -> Tensor:
    """Reference math, replicated op-for-op, in fp32 (fp64 stays fp64)."""
    compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    uc = u.to(compute_dtype)
    dc = delta.to(compute_dtype)
    ac = A.to(compute_dtype)
    bc = B.to(compute_dtype)
    cc = C.to(compute_dtype)
    dsc = D.to(compute_dtype)

    batch, seq_len, d_model = uc.shape
    n_state = ac.shape[1]

    delta_bar = F.softplus(dc)
    a_bar = torch.exp(delta_bar.unsqueeze(-1) * ac.unsqueeze(0).unsqueeze(0))
    b_bar = delta_bar.unsqueeze(-1) * bc.unsqueeze(2)

    h = torch.zeros(batch, d_model, n_state, dtype=compute_dtype, device=uc.device)
    ys: list[Tensor] = []
    for t in range(seq_len):
        h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * uc[:, t, :].unsqueeze(-1)
        ys.append((h * cc[:, t, :].unsqueeze(1)).sum(-1) + dsc * uc[:, t, :])
    return torch.stack(ys, dim=1).to(u.dtype)


class _ForwardScanCuda(torch.autograd.Function):
    """Triton forward; backward is the hand-written Triton kernel (C2)."""

    @staticmethod
    def forward(
        ctx: Any,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
        config: KernelConfig | None = None,
    ) -> Tensor:
        ctx.save_for_backward(u, delta, A, B, C, D)
        ctx.config = config
        batch, seq_len, width = u.shape
        if _resolve_scan_mode(config, seq_len, batch, width, is_forward=True) == "chunk_parallel":
            from lethe.kernels.ops import _triton_chunk_parallel_fwd

            k = _auto_chunk_len(seq_len, config.chunk_len if config is not None else None)
            return _triton_chunk_parallel_fwd.launch_chunk_parallel_scan(
                u, delta, A, B, C, D, chunk_len=k, config=config
            )
        from lethe.kernels.ops import _triton_fwd_scan

        return _triton_fwd_scan.launch_forward_scan(u, delta, A, B, C, D, config=config)

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Tensor | None, ...]:
        u, delta, a, b, c, d_skip = ctx.saved_tensors
        config = ctx.config
        # Both modes are correct for any tiling; chunk_parallel reassociates the backward too.
        batch, seq_len, width = u.shape
        n_state = a.shape[1]
        if (
            _resolve_scan_mode(
                config, seq_len, batch, width, is_forward=False, n_state=n_state, device=u.device
            )
            == "chunk_parallel"
        ):
            from lethe.kernels.ops import _triton_chunk_parallel_bwd

            k = _auto_chunk_len(seq_len, config.chunk_len if config is not None else None)
            grads = _triton_chunk_parallel_bwd.launch_backward_chunk_parallel_scan(
                u, delta, a, b, c, d_skip, grad_y, chunk_len=k, config=config
            )
        else:
            from lethe.kernels.ops import _triton_bwd_scan

            grads = _triton_bwd_scan.launch_backward_scan(
                u, delta, a, b, c, d_skip, grad_y, config=config
            )
        # +1 None for the non-tensor forward arg (config).
        return (*grads, None)


def forward_chunked_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    *,
    chunk_size: int = 64,
    config: KernelConfig | None = None,
) -> Tensor:
    """Selective state-space scan forward (SISO, Mamba-1 recurrence)."""
    seq_len = u.shape[1]
    if seq_len % chunk_size != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by chunk_size {chunk_size}")
    if u.is_cuda and u.dtype in _TRITON_DTYPES and _triton_usable():
        return cast(
            Tensor,
            _ForwardScanCuda.apply(u, delta, A, B, C, D, config),  # type: ignore[no-untyped-call]
        )
    return _scan_eager(u, delta, A, B, C, D)


def triton_scan_resource_meta(config: KernelConfig | None = None) -> dict[str, int] | None:
    """Resource metadata of the compiled Triton forward scan kernel, if any."""
    if not _triton_usable():
        return None
    from lethe.kernels.ops import _triton_chunk_parallel_fwd, _triton_fwd_scan

    mode = config.scan_mode if config is not None else None
    if mode == "chunk_parallel":
        return _triton_chunk_parallel_fwd.resource_meta()
    if mode == "serial":
        return _triton_fwd_scan.resource_meta()
    return max_resource_meta(
        _triton_fwd_scan.resource_meta(), _triton_chunk_parallel_fwd.resource_meta()
    )
