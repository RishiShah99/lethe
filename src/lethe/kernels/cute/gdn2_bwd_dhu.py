"""K#1, native Blackwell (sm_100) reverse inter-chunk state scan (GDN backward, B4)."""
# NB: no `from __future__ import annotations`, PEP 563 stringizes @cute.struct fields.

import contextlib
from collections.abc import Iterator

import torch
from torch import Tensor

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.pipeline as pipeline
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.nvgpu import cpasync, tcgen05
    from cutlass.cute.runtime import from_dlpack

    _HAVE = True
except ImportError:  # pragma: no cover - CPU dev box
    _HAVE = False


def _cur_stream() -> "cuda_driver.CUstream":
    """torch's current CUDA stream as a CuTe-DSL ``CUstream``."""
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


GRAPH_CAPTURE = False  # True inside a CUDA-graph capture → kernel-boundary syncs become no-ops


@contextlib.contextmanager
def graph_capture() -> Iterator[None]:
    """Mark the cw backward as inside a ``torch.cuda.graph`` capture."""
    global GRAPH_CAPTURE
    prev = GRAPH_CAPTURE
    GRAPH_CAPTURE = True
    try:
        yield
    finally:
        GRAPH_CAPTURE = prev


def maybe_sync() -> None:
    """``torch.cuda.synchronize()`` unless inside a graph capture (:func:`graph_capture`)."""
    if not GRAPH_CAPTURE:
        torch.cuda.synchronize()


CHUNK = 64
D_K = 128
D_V = 64
THREADS = 128
AB_STAGES = 2


def is_available() -> bool:
    return _HAVE


if _HAVE:
    _io = cutlass.Float16
    _acc = cutlass.Float32
    # increment A: M=d_k, N=d_v, K=2C (qg/−w stacked on do/dvl along the C contraction).
    _MNK_INST = (D_K, D_V, 16)
    _MNK_TILER = (D_K, D_V, 2 * CHUNK)
    _AB_MBAR = AB_STAGES * 2

    @cute.struct
    class _Smem:
        ab_mbar: cute.struct.MemRange[cutlass.Int64, _AB_MBAR]
        acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_buf: cutlass.Int32

    @cute.kernel
    def _gemm_kernel(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b: cute.CopyAtom,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        coord = (bidx, bidy, None)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
        tmem.allocate(512)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)

        nbytes = cute.size_in_bytes(
            _io, cute.select(a_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(_io, cute.select(b_layout, mode=[0, 1, 2]))
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc_mbar.data_ptr(),
        ).make_participants()

        gA = cute.local_tile(mA, _MNK_TILER, coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, _MNK_TILER, coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, _MNK_TILER, coord, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)

        sub = 1
        epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // sub),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
        tmem_thr = tmem_copy.get_slice(tidx)
        tDtC = tmem_thr.partition_S(tCtAcc_epi)
        tDgC = tmem_thr.partition_D(gC_epi)
        tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _acc)
        tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _io)

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc_empty = acc_prod.acquire_and_advance()
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_a,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b,
                    tBgB[(None, ab_e.count)],
                    tBsB[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    c = (None, None, kb, ab_f.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[c], tCrB[c], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            acc_empty.commit()

        tmem.relinquish_alloc_permit()
        acc_f = acc_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_f.release()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.jit
    def _gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
        op = tcgen05.MmaF16BF16Op(
            _io,
            _acc,
            _MNK_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, a.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, b.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, _MNK_TILER, tiled_mma)
        b_atom, b_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b, b1, _MNK_TILER, tiled_mma)
        grid = cute.ceil_div((*c.layout.shape, 1), _MNK_TILER[:2])
        _gemm_kernel(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout).launch(
            grid=grid, block=(THREADS, 1, 1)
        )

    # Keyed by shape/dtype so an off-tile caller recompiles instead of reusing the wrong kernel.
    _gemm_aa_cache: dict[tuple[object, ...], object] = {}

    def _gemm_aa(a: Tensor, b: Tensor, out: Tensor) -> None:
        """Run increment A's (128,64,128) GEMM: ``out[128,64] = a[128,128] @ b[64,128]^T``."""
        ca, cb, cc = _mark(a.contiguous()), _mark(b.contiguous()), _mark(out)
        key = (a.shape, b.shape, out.shape, a.dtype, b.dtype, out.dtype)
        ex = _gemm_aa_cache.get(key)
        if ex is None:
            ex = cute.compile(_gemm_host, ca, cb, cc)
            _gemm_aa_cache[key] = ex
        ex(ca, cb, cc)

    @cute.kernel
    def _gemm_kernel_b(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b: cute.CopyAtom,
        mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        coord = (bidx, bidy, None, bidz)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
        tmem.allocate(512)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)

        nbytes = cute.size_in_bytes(
            _io, cute.select(a_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(_io, cute.select(b_layout, mode=[0, 1, 2]))
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc_mbar.data_ptr(),
        ).make_participants()

        gA = cute.local_tile(mA, _MNK_TILER, coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, _MNK_TILER, coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, _MNK_TILER, coord, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)

        sub = 1
        epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // sub),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        gC_epi = cute.zipped_divide(tCgC, epi_tiler)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
        tmem_thr = tmem_copy.get_slice(tidx)
        tDtC = tmem_thr.partition_S(tCtAcc_epi)
        tDgC = tmem_thr.partition_D(gC_epi)
        tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _acc)
        tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _io)

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc_empty = acc_prod.acquire_and_advance()
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_a,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b,
                    tBgB[(None, ab_e.count)],
                    tBsB[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    c = (None, None, kb, ab_f.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[c], tCrB[c], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            acc_empty.commit()

        tmem.relinquish_alloc_permit()
        acc_f = acc_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_f.release()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.jit
    def _gemm_host_b(
        a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda_driver.CUstream
    ) -> None:
        op = tcgen05.MmaF16BF16Op(
            _io,
            _acc,
            _MNK_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, a.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, b.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, _MNK_TILER, tiled_mma)
        b_atom, b_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b, b1, _MNK_TILER, tiled_mma)
        grid = (
            cute.ceil_div(c.layout.shape[0], _MNK_TILER[0]),
            cute.ceil_div(c.layout.shape[1], _MNK_TILER[1]),
            c.layout.shape[2],
        )
        _gemm_kernel_b(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout).launch(
            grid=grid, block=(THREADS, 1, 1), stream=stream
        )

    _gemm_b_cache: dict[int, object] = {}  # cute.compile of _gemm_host_b, keyed by batch size Z

    def _gemm_batched(a: Tensor, b: Tensor, out: Tensor) -> None:
        """Batched ``out[z]=a[z]@b[z]^T`` on the (128,64,128) config, L=z as grid-z."""
        z = int(a.shape[0])
        ca, cb, cc = _mark_b(a), _mark_b(b), _mark_b(out)
        ex = _gemm_b_cache.get(z)
        if ex is None:
            ex = cute.compile(_gemm_host_b, ca, cb, cc, _cur_stream())
            _gemm_b_cache[z] = ex
        ex(ca, cb, cc, _cur_stream())

    def _mark_b(t: Tensor) -> object:
        """Mark a batch-first [Z,R,S] tensor as a cute [R,S,L] tensor (L = largest stride)."""
        v = t.contiguous().permute(1, 2, 0)
        return from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _bmm_tc(x: Tensor, y: Tensor) -> Tensor:
    """Batched ``x[Z,M,K] @ y[Z,K,N] -> [Z,M,N]`` via the batched (128,64,128) a@b^T GEMM."""
    f16, dev = torch.float16, x.device
    z, m, kk = x.shape
    n = y.shape[-1]
    if m > D_K or kk > D_K:
        raise ValueError(
            f"_bmm_tc stages M,K into a ({D_K},{D_K}) buffer; got x[Z,{m},{kk}], both must "
            f"be <= {D_K} (N tiles by D_V={D_V}; M,K below {D_K} zero-pad and stay correct)"
        )
    a = torch.zeros(z, D_K, D_K, dtype=f16, device=dev)
    a[:, :m, :kk] = x.to(f16)
    yt = y.transpose(-1, -2).contiguous()  # [Z, N, K]
    out = torch.zeros(z, m, n, dtype=torch.float32, device=dev)
    for n0 in range(0, n, D_V):
        wn = min(D_V, n - n0)
        bmat = torch.zeros(z, D_V, D_K, dtype=f16, device=dev)
        bmat[:, :wn, :kk] = yt[:, n0 : n0 + wn].to(f16)
        o = torch.zeros(z, D_K, D_V, dtype=f16, device=dev)
        _gemm_batched(a, bmat, o)
        out[:, :, n0 : n0 + wn] = o[:, :m, :wn].float()
    return out


def _mark(t: Tensor) -> object:
    return (
        from_dlpack(t.contiguous(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )


def run_k1(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run K#1; return ``(dh, dv2, dh0)`` head-major chunked. Increment A handles NT=1."""
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    if nt != 1:
        raise NotImplementedError("increment A is NT=1 only; the reverse loop is increment B")

    gamma = torch.exp2(g2)
    qg = q * gamma[..., None]  # [B,HV,1,C,d_k]
    # A = [qg | −w] along C -> [n_bh, d_k, 2C] (K-major: contraction 2C contiguous).
    a = torch.cat([qg, -w], dim=3).reshape(b * hv, c * 2, d_k).transpose(1, 2).contiguous()
    bmat = torch.cat([do, dv_local], dim=3).reshape(b * hv, c * 2, d_v).transpose(1, 2).contiguous()
    a = a.to(torch.float16)
    bmat = bmat.to(torch.float16)

    n_bh = b * hv
    dev = q.device
    dh0 = torch.zeros(n_bh, d_k, d_v, dtype=torch.float16, device=dev)
    for i in range(n_bh):
        ai, bi = a[i].contiguous(), bmat[i].contiguous()
        ci = torch.zeros(d_k, d_v, dtype=torch.float16, device=dev)
        _gemm_aa(ai, bi, ci)
        dh0[i] = ci

    maybe_sync()

    dh = torch.zeros(b, hv, nt, d_k, d_v, dtype=torch.float16, device=dev)
    dv2 = dv_local.to(torch.float16).clone()
    return dh, dv2, dh0.reshape(b, hv, d_k, d_v)


def run_k1_incB_host(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Increment B, host-orchestrated reverse inter-chunk state scan."""
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    if d_k > D_K:
        raise ValueError(f"increment-B stages d_k into a {D_K}-wide tile; got d_k={d_k} > {D_K}")
    dev = q.device
    n_bh = b * hv
    f16 = torch.float16

    gamma = torch.exp2(g2)
    qg = q * gamma[..., None]
    decay = torch.exp2(g_last[..., None] - g2)
    glast_exp = torch.exp2(g_last)

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(n_bh, *x.shape[2:])

    kf, qgf, wf, dof = _flat(k), _flat(qg), _flat(w), _flat(do)
    dvlf, decf, gef = _flat(dv_local), _flat(decay), _flat(glast_exp)
    b_dh = _flat(dht).contiguous().float()  # [n_bh,d_k,d_v] natural, resident

    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=torch.float32, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=torch.float32, device=dev)

    for i in range(n_bh):
        for it in reversed(range(nt)):
            dh[i, it] = b_dh[i]
            # G1: b_dv[C,d_v] = k[C,d_k] @ b_dh[d_k,d_v].
            a_g1 = torch.zeros(d_k, d_k, dtype=f16, device=dev)
            a_g1[:c] = kf[i, it].to(f16)
            out_g1 = torch.zeros(d_k, d_v, dtype=f16, device=dev)
            _gemm_aa(a_g1, b_dh[i].transpose(-1, -2).to(f16), out_g1)
            b_dv = out_g1[:c].float() * decf[i, it][:, None] + dvlf[i, it]  # [C,d_v]
            dv2[i, it] = b_dv
            # GA (= increment A's stacked GEMM): t[d_k,d_v] = [qg|−w]@[do|b_dv]^T
            a_ga = torch.cat([qgf[i, it], -wf[i, it]], dim=0).transpose(-1, -2).to(f16)  # [d_k,2C]
            b_ga = torch.cat([dof[i, it], b_dv], dim=0).transpose(-1, -2).to(f16)  # [d_v,2C]
            t = torch.zeros(d_k, d_v, dtype=f16, device=dev)
            _gemm_aa(a_ga, b_ga, t)
            b_dh[i] = gef[i, it] * b_dh[i] + t.float()

    maybe_sync()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


def run_k1_incB_batched(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lever B, reverse-state scan with the (b,hv) groups batched into one GEMM per step."""
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    if d_k > D_K:
        raise ValueError(f"increment-B stages d_k into a {D_K}-wide tile; got d_k={d_k} > {D_K}")
    dev = q.device
    n_bh = b * hv

    gamma = torch.exp2(g2)
    qg = q * gamma[..., None]
    decay = torch.exp2(g_last[..., None] - g2)
    glast_exp = torch.exp2(g_last)

    def _flat(x: Tensor) -> Tensor:
        return x.reshape(n_bh, *x.shape[2:])

    kf, qgf, wf, dof = _flat(k), _flat(qg), _flat(w), _flat(do)
    dvlf, decf, gef = _flat(dv_local), _flat(decay), _flat(glast_exp)
    b_dh = _flat(dht).contiguous().float()  # [n_bh, d_k, d_v] natural, resident

    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=torch.float32, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=torch.float32, device=dev)

    for it in reversed(range(nt)):
        dh[:, it] = b_dh
        b_dv = _bmm_tc(kf[:, it], b_dh) * decf[:, it][..., None] + dvlf[:, it]  # [n_bh,C,d_v]
        dv2[:, it] = b_dv
        a_ga = torch.cat([qgf[:, it], -wf[:, it]], dim=1).transpose(-1, -2)  # [n_bh,d_k,2C]
        b_ga = torch.cat([dof[:, it], b_dv], dim=1)  # [n_bh,2C,d_v]
        t = _bmm_tc(a_ga, b_ga)  # [n_bh,d_k,d_v]
        b_dh = gef[:, it][:, None, None] * b_dh + t

    maybe_sync()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


def _incb2_pack_scalar(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
) -> dict[str, Tensor]:
    """Host-precompute inc-B2's per-chunk operand buffers, flattened over L = n_bh·NT."""
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    ll = n_bh * nt
    dev = q.device

    gamma = torch.exp2(g2)
    qg = q * gamma[..., None]
    decay = torch.exp2(g_last[..., None] - g2)  # [B,HV,NT,C]
    glast = torch.exp2(g_last)  # [B,HV,NT]

    def _flatL(x: Tensor) -> Tensor:
        return x.reshape(ll, *x.shape[3:])

    kL, qgL, wL, doL = _flatL(k), _flatL(qg), _flatL(w), _flatL(do)
    decL, dvlL, geL = _flatL(decay), _flatL(dv_local), glast.reshape(ll)

    a_g1 = torch.zeros(ll, d_k, d_k, dtype=q.dtype, device=dev)
    a_g1[:, :c] = kL
    a_ga = torch.cat([qgL, -wL], dim=1).transpose(-1, -2).contiguous()  # [L, d_k, 2C]
    b_ga = torch.zeros(ll, d_v, 2 * c, dtype=q.dtype, device=dev)
    b_ga[:, :, :c] = doL.transpose(-1, -2)  # do^T; the [:, :, C:] half is b_dv^T in-kernel
    return {
        "a_g1": a_g1,
        "a_ga": a_ga,
        "b_ga": b_ga,
        "decay": decL,
        "dv_local": dvlL,
        "glast": geL,
    }


def _run_k1_incB2_modelled(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-torch model of inc-B2's exact in-kernel dataflow (the kernel's spec)."""
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    dev = q.device
    n_bh = b * hv
    buf = _incb2_pack_scalar(q, k, w, g2, g_last, do, dv_local)
    a_g1, a_ga, b_ga = buf["a_g1"], buf["a_ga"], buf["b_ga"]
    decay, dv_local_L, glast = buf["decay"], buf["dv_local"], buf["glast"]

    b_dh = dht.reshape(n_bh, d_k, d_v).clone()
    dh = torch.zeros(n_bh, nt, d_k, d_v, dtype=q.dtype, device=dev)
    dv2 = torch.zeros(n_bh, nt, c, d_v, dtype=q.dtype, device=dev)

    for i in range(n_bh):
        for it in reversed(range(nt)):
            lid = i * nt + it
            dh[i, it] = b_dh[i]
            bdv_raw = a_g1[lid] @ b_dh[i]  # [128, d_v], rows[:C] real (G1: a @ b_dh)
            b_dv = bdv_raw[:c] * decay[lid][:, None] + dv_local_L[lid]  # [C, d_v]
            dv2[i, it] = b_dv
            b_ga[lid, :, c:] = b_dv.transpose(-1, -2)  # round-trip: b_dv^T into the 2nd half
            t = a_ga[lid] @ b_ga[lid].transpose(-1, -2)  # [d_k, d_v] (GA: a @ b^T)
            b_dh[i] = glast[lid] * b_dh[i] + t

    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


if _HAVE:

    @cute.kernel
    def _incb2_kernel(
        tiled_mma: cute.TiledMma,
        tma_ag1: cute.CopyAtom,
        m_ag1: cute.Tensor,
        tma_aga: cute.CopyAtom,
        m_aga: cute.Tensor,
        tma_bdhT: cute.CopyAtom,
        m_bdhT: cute.Tensor,
        tma_bga: cute.CopyAtom,
        m_bga: cute.Tensor,
        m_bdv: cute.Tensor,
        m_t: cute.Tensor,
        m_bdh: cute.Tensor,
        m_dh: cute.Tensor,
        m_dv2: cute.Tensor,
        m_decay: cute.Tensor,
        m_dvl: cute.Tensor,
        m_glast: cute.Tensor,
        m_bdhT_s: cute.Tensor,
        m_bga_s: cute.Tensor,
        m_bdv_s: cute.Tensor,
        m_t_s: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        nt: cutlass.Constexpr,
        c: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        bisect: cutlass.Constexpr,
    ) -> None:
        # bisect: 0 full, 1 G1-only, 2 G1+GA no-rt, 3 G1 plain-bh, 4 relinquish-after-mainloop.
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, _, bh = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
        tmem.allocate(512)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_ag1)
            cpasync.prefetch_descriptor(tma_aga)
            cpasync.prefetch_descriptor(tma_bdhT)
            cpasync.prefetch_descriptor(tma_bga)

        nbytes = cute.size_in_bytes(
            _io, cute.select(a_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(_io, cute.select(b_layout, mode=[0, 1, 2]))
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc_mbar.data_ptr(),
        ).make_participants()

        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)  # layout carrier (pure, no TMEM ptr)
        tmem.wait_for_alloc()
        if cutlass.const_expr(bisect != 4):
            tmem.relinquish_alloc_permit()

        # range_constexpr UNROLLS the reverse loop (nt is Constexpr) → NO dynamic scf.for.
        for it in cutlass.range_constexpr(nt):
            tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc_frag.layout)
            i_t = nt - 1 - it
            lid = bh * nt + i_t

            # dh[lid] = current natural b_dh (read before this chunk's carry updates it)
            for e in cutlass.range(tidx, D_K * d_v, THREADS):
                r, col = e // d_v, e % d_v
                m_dh[lid, r, col] = m_bdh[bh, r, col].to(_io)
            cute.arch.barrier()

            # ===== G1: m_bdv[·,bh] = m_ag1[·,lid] @ m_bdhT[·,bh]^T (proven (128,64,128) body) =====
            if cutlass.const_expr(bisect == 3):
                gA = cute.local_tile(m_ag1, _MNK_TILER, (0, 0, None, bh), proj=(1, None, 1))
            else:
                gA = cute.local_tile(m_ag1, _MNK_TILER, (0, 0, None, lid), proj=(1, None, 1))
            gB = cute.local_tile(m_bdhT, _MNK_TILER, (0, 0, None, bh), proj=(None, 1, 1))
            gC = cute.local_tile(m_bdv, _MNK_TILER, (0, 0, None, bh), proj=(1, 1, None))
            thr_mma = tiled_mma.get_slice(0)
            tCgC = thr_mma.partition_C(gC)
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_ag1,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_bdhT,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1])),)
            tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
            gC_epi = cute.zipped_divide(tCgC, epi_tiler)
            tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)
            tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
            tmem_thr = tmem_copy.get_slice(tidx)
            tDtC = tmem_thr.partition_S(tCtAcc_epi)
            tDgC = tmem_thr.partition_D(gC_epi)
            tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _acc)
            tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _io)
            if warp_idx == 0:
                acc_empty = acc_prod.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(cute.size(gA, mode=[2]), prefetch_stages=AB_STAGES - 1):
                    ab_e = ab_prod.acquire_and_advance()
                    # gmem K coord is literal 0; ab_e.count grows across mainloops, goes OOB.
                    cute.copy(
                        tma_ag1,
                        tAgA[(None, 0)],
                        tAsA[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    cute.copy(
                        tma_bdhT,
                        tBgB[(None, 0)],
                        tBsB[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    ab_f = ab_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cc = (None, None, kb, ab_f.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[cc], tCrB[cc], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_f.release()
                acc_empty.commit()
            if cutlass.const_expr(bisect == 4):
                tmem.relinquish_alloc_permit()  # proven-ordering probe: relinquish after mainloop
            acc_f = acc_cons.wait_and_advance()
            for i in cutlass.range(cute.size(tDtC, mode=[2])):
                cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
                tCrC.store(tCrAcc.load().to(_io))
                cute.autovec_copy(tCrC, tDgC[None, None, i])
            acc_f.release()
            pipeline.sync(barrier_id=1)

            if cutlass.const_expr(bisect == 0 or bisect == 2):
                # SIMT glue: b_dv = bdv_raw[:C]·decay + dv_local; write dv2 + b_ga's b_dv^T half.
                for e in cutlass.range(tidx, c * d_v, THREADS):
                    r, col = e // d_v, e % d_v
                    val = m_bdv_s[bh, r, col].to(_acc) * m_decay[lid, r] + m_dvl[lid, r, col]
                    m_dv2[lid, r, col] = val.to(_io)
                    if cutlass.const_expr(bisect == 0):
                        m_bga_s[lid, col, c + r] = val.to(_io)  # b_dv^T → 2nd half (round-trip)
                if cutlass.const_expr(bisect == 0):
                    cute.arch.fence_proxy("async.global")  # SIMT store → TMA (async-proxy) read
                cute.arch.barrier()

                # ===== GA: m_t[·,bh] = m_aga[·,lid] @ m_bga[·,lid]^T =====
                gA = cute.local_tile(m_aga, _MNK_TILER, (0, 0, None, lid), proj=(1, None, 1))
                gB = cute.local_tile(m_bga, _MNK_TILER, (0, 0, None, lid), proj=(None, 1, 1))
                gC = cute.local_tile(m_t, _MNK_TILER, (0, 0, None, bh), proj=(1, 1, None))
                thr_mma = tiled_mma.get_slice(0)
                tCgC = thr_mma.partition_C(gC)
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    tma_aga,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sA, 0, 3),
                    cute.group_modes(thr_mma.partition_A(gA), 0, 3),
                )
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    tma_bga,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(thr_mma.partition_B(gB), 0, 3),
                )
                tCrA = tiled_mma.make_fragment_A(sA)
                tCrB = tiled_mma.make_fragment_B(sB)
                tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
                gC_epi = cute.zipped_divide(tCgC, epi_tiler)
                tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
                tmem_thr = tmem_copy.get_slice(tidx)
                tDtC = tmem_thr.partition_S(tCtAcc_epi)
                tDgC = tmem_thr.partition_D(gC_epi)
                tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _acc)
                tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, _io)
                if warp_idx == 0:
                    acc_empty = acc_prod.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for _kt in cutlass.range(
                        cute.size(gA, mode=[2]), prefetch_stages=AB_STAGES - 1
                    ):
                        ab_e = ab_prod.acquire_and_advance()
                        cute.copy(
                            tma_aga,
                            tAgA[(None, 0)],
                            tAsA[(None, ab_e.index)],
                            tma_bar_ptr=ab_e.barrier,
                        )
                        cute.copy(
                            tma_bga,
                            tBgB[(None, 0)],
                            tBsB[(None, ab_e.index)],
                            tma_bar_ptr=ab_e.barrier,
                        )
                        ab_f = ab_cons.wait_and_advance()
                        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                            cc = (None, None, kb, ab_f.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[cc], tCrB[cc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        ab_f.release()
                    acc_empty.commit()
                acc_f = acc_cons.wait_and_advance()
                for i in cutlass.range(cute.size(tDtC, mode=[2])):
                    cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
                    tCrC.store(tCrAcc.load().to(_io))
                    cute.autovec_copy(tCrC, tDgC[None, None, i])
                acc_f.release()
                pipeline.sync(barrier_id=1)

                # SIMT carry: b_dh = exp2(g_last)·b_dh + t; write natural b_dh + transposed b_dhT.
                for e in cutlass.range(tidx, D_K * d_v, THREADS):
                    r, col = e // d_v, e % d_v
                    val = m_glast[lid, r] * m_bdh[bh, r, col] + m_t_s[bh, r, col].to(_acc)
                    m_bdh[bh, r, col] = val
                    if cutlass.const_expr(bisect == 0):
                        m_bdhT_s[bh, col, r] = val.to(_io)  # transposed operand for next G1
                if cutlass.const_expr(bisect == 0):
                    cute.arch.fence_proxy("async.global")  # b_dhT round-trip → next chunk's G1 TMA
                cute.arch.barrier()

        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.jit
    def _incb2_host(
        m_ag1: cute.Tensor,
        m_aga: cute.Tensor,
        m_bdhT: cute.Tensor,
        m_bga: cute.Tensor,
        m_bdv: cute.Tensor,
        m_t: cute.Tensor,
        m_bdh: cute.Tensor,
        m_dh: cute.Tensor,
        m_dv2: cute.Tensor,
        m_decay: cute.Tensor,
        m_dvl: cute.Tensor,
        m_glast: cute.Tensor,
        m_bdhT_s: cute.Tensor,
        m_bga_s: cute.Tensor,
        m_bdv_s: cute.Tensor,
        m_t_s: cute.Tensor,
        n_bh: cutlass.Constexpr,
        nt: cutlass.Constexpr,
        c: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        bisect: cutlass.Constexpr,
    ) -> None:
        op = tcgen05.MmaF16BF16Op(
            _io,
            _acc,
            _MNK_INST,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, _MNK_TILER, m_ag1.element_type, AB_STAGES
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, _MNK_TILER, m_bga.element_type, AB_STAGES
        )
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        at_ag1, t_ag1 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_ag1, a1, _MNK_TILER, tiled_mma)
        at_aga, t_aga = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_aga, a1, _MNK_TILER, tiled_mma)
        at_bdhT, t_bdhT = cute.nvgpu.make_tiled_tma_atom_B(
            op_tma, m_bdhT, b1, _MNK_TILER, tiled_mma
        )
        at_bga, t_bga = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bga, b1, _MNK_TILER, tiled_mma)
        _incb2_kernel(
            tiled_mma,
            at_ag1,
            t_ag1,
            at_aga,
            t_aga,
            at_bdhT,
            t_bdhT,
            at_bga,
            t_bga,
            m_bdv,
            m_t,
            m_bdh,
            m_dh,
            m_dv2,
            m_decay,
            m_dvl,
            m_glast,
            m_bdhT_s,
            m_bga_s,
            m_bdv_s,
            m_t_s,
            a_layout,
            b_layout,
            nt,
            c,
            d_v,
            bisect,
        ).launch(grid=(1, 1, n_bh), block=(THREADS, 1, 1))

    _incb2_cache: dict[tuple[int, ...], object] = {}  # cute.compile keyed (n_bh,nt,c,d_v,bisect)


def _mark_simt(t: Tensor) -> object:
    """Mark a plain contiguous tensor for in-kernel SIMT element access (no TMA atom)."""
    return from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


def _incb2_launch(
    a_g1: Tensor,
    a_ga: Tensor,
    b_ga: Tensor,
    b_dh: Tensor,
    decay: Tensor,
    dvl: Tensor,
    glast: Tensor,
    n_bh: int,
    nt: int,
    c: int,
    d_k: int,
    d_v: int,
    *,
    bisect: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mark the inc-B2 operands, compile-cache :func:`_incb2_host`, launch once, return flat."""
    dev = a_g1.device
    f16, f32 = torch.float16, torch.float32
    ll = n_bh * nt

    # The 4 dual-use scratch tensors get both a TMA view and a SIMT glue view over the same storage.
    a_g1_16 = a_g1.to(f16).contiguous()  # [L,128,128] TMA operand A (G1)
    a_ga_16 = a_ga.to(f16).contiguous()  # [L,128,128] TMA operand A (GA)
    b_ga_16 = b_ga.to(f16).contiguous()  # [L,d_v,2C] TMA operand B (GA); 2nd half written in-kernel
    b_dhT = b_dh.transpose(-1, -2).contiguous().to(f16)  # [n_bh,d_v,d_k] resident G1 operand
    bdv_raw = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)  # G1 epilogue landing
    t_scr = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)  # GA epilogue landing
    dh = torch.zeros(ll, d_k, d_v, dtype=f16, device=dev)
    dv2 = torch.zeros(ll, c, d_v, dtype=f16, device=dev)

    args = (
        _mark_b(a_g1_16),
        _mark_b(a_ga_16),
        _mark_b(b_dhT),
        _mark_b(b_ga_16),
        _mark_b(bdv_raw),
        _mark_b(t_scr),
        _mark_simt(b_dh),
        _mark_simt(dh),
        _mark_simt(dv2),
        _mark_simt(decay.to(f32)),
        _mark_simt(dvl.to(f32)),
        _mark_simt(glast.to(f32)),
        _mark_simt(b_dhT),
        _mark_simt(b_ga_16),
        _mark_simt(bdv_raw),
        _mark_simt(t_scr),
        n_bh,
        nt,
        c,
        d_v,
        bisect,
    )
    key = (n_bh, nt, c, d_v, bisect)
    ex = _incb2_cache.get(key)
    if ex is None:
        ex = cute.compile(_incb2_host, *args)
        _incb2_cache[key] = ex
    # n_bh/nt/c/d_v/bisect are baked at compile, dropped from the call; pass 16 tensors + stream.
    ex(*args[:16])
    torch.cuda.synchronize()
    return (
        dh.reshape(n_bh, nt, d_k, d_v),
        dv2.reshape(n_bh, nt, c, d_v),
        b_dh.reshape(n_bh, d_k, d_v),
    )


def run_k1_incB2(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
    *,
    bisect: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Lever D, the fused persistent reverse-state scan (one CTA per (b,hv), loop in-kernel)."""
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    buf = _incb2_pack_scalar(q, k, w, g2, g_last, do, dv_local)
    b_dh = dht.reshape(n_bh, d_k, d_v).clone().float()
    glast = buf["glast"][:, None].expand(n_bh * nt, d_k).contiguous()  # [L] scalar → [L,d_k]
    dh, dv2, dh0 = _incb2_launch(
        buf["a_g1"],
        buf["a_ga"],
        buf["b_ga"],
        b_dh,
        buf["decay"],
        buf["dv_local"],
        glast,
        n_bh,
        nt,
        c,
        d_k,
        d_v,
        bisect=bisect,
    )
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        dh0.reshape(b, hv, d_k, d_v),
    )


# Lever B is the default reverse-state path (batched over the (b,hv) groups).
run_k1_incB = run_k1_incB_batched
