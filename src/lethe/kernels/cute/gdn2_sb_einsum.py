"""Stage-B intra-einsum kernel, replaces the ``decay_rel`` einsums after the K#2 fusion."""
# NB: no `from __future__ import annotations`, keep consistent with the DSL kernel files.

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute

    _HAVE = True
except ImportError:  # pragma: no cover - CPU dev box
    _HAVE = False

D_K = 128
CHUNK = 64
THREADS = 128


def is_available() -> bool:
    return _HAVE


def sbe_dims_ok(c: int, d_k: int) -> bool:
    """True iff the kernel's proven tile fits: C=64, d_k=128 exactly."""
    return c == CHUNK and d_k == D_K


def _cur_stream() -> "cuda_driver.CUstream":
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _mark_simt(t: Tensor) -> object:
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


if _HAVE:

    @cute.kernel
    def _kern_sbe(
        m_coef: cute.Tensor,
        m_k: cute.Tensor,
        m_q: cute.Tensor,
        m_g2: cute.Tensor,
        m_dqi: cute.Tensor,
        m_dki: cute.Tensor,
        c: cutlass.Constexpr,
        d_k: cutlass.Constexpr,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        _, _, z = cute.arch.block_idx()

        # Pass A: dq_intra[i,d] = sum_{s<=i} coef[i,s]*k[s,d]*exp2(g2[i,d]-g2[s,d]).
        for e in cutlass.range(tidx, c * d_k, THREADS):
            i_, d_ = e // d_k, e % d_k
            g2_i = m_g2[z, i_, d_]
            acc_a = cutlass.Float32(0.0)
            for s_ in cutlass.range(i_ + 1):
                dec = cute.math.exp2(g2_i - m_g2[z, s_, d_], fastmath=True)
                acc_a += m_coef[z, i_, s_] * m_k[z, s_, d_] * dec
            m_dqi[z, i_, d_] = acc_a

        # Pass B: dk_intra[s,d] = sum_{i>=s} coef[i,s]*q[i,d]*exp2(g2[i,d]-g2[s,d]).
        for e in cutlass.range(tidx, c * d_k, THREADS):
            s_, d_ = e // d_k, e % d_k
            g2_s = m_g2[z, s_, d_]
            acc_b = cutlass.Float32(0.0)
            for i2 in cutlass.range(s_, c, 1):
                dec = cute.math.exp2(m_g2[z, i2, d_] - g2_s, fastmath=True)
                acc_b += m_coef[z, i2, s_] * m_q[z, i2, d_] * dec
            m_dki[z, s_, d_] = acc_b

    @cute.jit
    def _sbe_host(
        m_coef: cute.Tensor,
        m_k: cute.Tensor,
        m_q: cute.Tensor,
        m_g2: cute.Tensor,
        m_dqi: cute.Tensor,
        m_dki: cute.Tensor,
        z_dim: cutlass.Constexpr,
        c: cutlass.Constexpr,
        d_k: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
    ) -> None:
        _kern_sbe(m_coef, m_k, m_q, m_g2, m_dqi, m_dki, c, d_k).launch(
            grid=(1, 1, z_dim), block=(THREADS, 1, 1), stream=stream
        )

    _sbe_cache: dict[tuple[int, ...], object] = {}


def run_sb_einsum(da_qk: Tensor, k: Tensor, q: Tensor, g2: Tensor) -> tuple[Tensor, Tensor]:
    """Both stage-B intra einsums in ONE grid-z launch, no ``decay_rel`` materialized."""
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    bsz, hh, nt, c, d_k = k.shape
    if not sbe_dims_ok(c, d_k):
        raise ValueError(f"run_sb_einsum requires C={CHUNK}, d_k={D_K}; got c={c}, d_k={d_k}")
    z = bsz * hh * nt
    f32 = torch.float32

    coef = da_qk.reshape(z, c, c).to(f32).contiguous()
    kf = k.reshape(z, c, d_k).to(f32).contiguous()
    qf = q.reshape(z, c, d_k).to(f32).contiguous()
    g2f = g2.reshape(z, c, d_k).to(f32).contiguous()
    dqi = torch.zeros(z, c, d_k, dtype=f32, device=k.device)
    dki = torch.zeros(z, c, d_k, dtype=f32, device=k.device)

    stream = _cur_stream()
    args = (
        _mark_simt(coef),
        _mark_simt(kf),
        _mark_simt(qf),
        _mark_simt(g2f),
        _mark_simt(dqi),
        _mark_simt(dki),
        z,
        c,
        d_k,
        stream,
    )
    key = (z, c, d_k)
    ex = _sbe_cache.get(key)
    if ex is None:
        ex = cute.compile(_sbe_host, *args)
        _sbe_cache[key] = ex
    # Constexpr args (z, c, d_k) baked + dropped: call tuple = 6 marked tensors + stream.
    ex(*args[:6], stream)
    maybe_sync()
    return (
        dqi.reshape(bsz, hh, nt, c, d_k),
        dki.reshape(bsz, hh, nt, c, d_k),
    )
