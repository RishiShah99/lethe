"""Level-2 de-glue channel-wise K#1 — GEMM + epilogue-fused tcgen05 kernels (SILICON-GO).

Promoted from ``scratch/k1_l2_epilogue.py`` after the burst-2 box gate
(results/k1_l2_epilogue_box.json: worst max_abs 7e-5 vs the fp64 cw bundle,
bit-deterministic). Two batched kernels — each the silicon-verified
``gdn2_bwd_dhu._gemm_kernel_b`` mainloop/epilogue with the per-step SIMT glue
appended after ``pipeline.sync``, before ``tmem.free`` — collapse the lever-B
reverse step (2 ``_bmm_tc`` launches + ~6 torch ops) to EXACTLY 2 launches and
0 torch compute ops:

  _kern_g1_epi — G1 GEMM (``a_g1[z] @ b_dhT[z]^T``) then SIMT: dh snapshot,
                 ``b_dv = raw[:C] + dv_local`` -> dv2 + b_ga second half
                 (``b_dv^T``), fence_proxy -> barrier (store->fence->barrier->TMA).
  _kern_ga_epi — GA GEMM (``a_ga[z] @ b_ga[z]^T``) then SIMT carry:
                 ``b_dh = glast (.) b_dh + t`` -> fp32 b_dh + fp16 b_dhT
                 (next G1's TMA operand), fence_proxy -> barrier.

Tile contract: ``_MNK_TILER`` bakes M=128, N=64, K=128 — C=64, d_k=128, d_v=64
exactly (:func:`l2_dims_ok`); d_v=128 stays on the lever-B batched path until the
N-tiling increment. relinquish_alloc_permit sits AFTER acc_empty.commit() in both
kernels (byte-fidelity to _gemm_kernel_b). Launches ride torch's current stream ->
CUDA-graph-capturable; ``maybe_sync`` at the end only.

Off-box: imports cleanly (guarded try/except -> _HAVE); ``_modelled_l2`` (the
kernel spec with fp16 round-trips modelled) runs anywhere and is CPU-pinned by
``tests/test_gdn2_l2.py``. Box gate harness: ``scratch/k1_l2_epilogue.py``.
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes @cute.struct fields.

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

D_K = 128
D_V = 64
CHUNK = 64
THREADS = 128
AB_STAGES = 2


def is_available() -> bool:
    return _HAVE


def l2_dims_ok(c: int, d_k: int, d_v: int) -> bool:
    """True iff the Level-2 kernels' baked tile fits: C=64, d_k=128, d_v=64 exactly."""
    return c == CHUNK and d_k == D_K and d_v == D_V


def _cur_stream() -> "cuda_driver.CUstream":
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _mark_b(t: Tensor) -> object:
    """[Z,R,S] -> cute [R,S,Z] L-major (TMA wants L as outermost stride)."""
    v = t.contiguous().permute(1, 2, 0)
    return from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _mark_simt(t: Tensor) -> object:
    """Plain contiguous [Z,...] tensor for SIMT element access."""
    return from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


if _HAVE:
    _io = cutlass.Float16
    _acc = cutlass.Float32
    _MNK_INST = (D_K, D_V, 16)
    _MNK_TILER = (D_K, D_V, 2 * CHUNK)
    _AB_MBAR = AB_STAGES * 2

    @cute.struct
    class _Smem:
        ab_mbar: cute.struct.MemRange[cutlass.Int64, _AB_MBAR]
        acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_buf: cutlass.Int32

    # ------------------------------------------------------------------
    # _kern_g1_epi: G1 GEMM + dh snapshot + dv2/b_ga glue
    #
    # a_g1[Z,128,128] @ b_dhT[Z,64,128]^T → raw[Z,128,64]
    # SIMT: dh snapshot over 128*d_v elements (full d_k rows),
    #       then glue over c*d_v elements writing dv2 + b_ga second half.
    # fence_proxy (each writing thread) then barrier, after both loops.
    # relinquish_alloc_permit: AFTER acc_empty.commit(), matching _gemm_kernel_b.
    # ------------------------------------------------------------------
    @cute.kernel
    def _kern_g1_epi(
        tiled_mma: cute.TiledMma,
        tma_ag1: cute.CopyAtom,
        m_ag1: cute.Tensor,
        tma_bdhT: cute.CopyAtom,
        m_bdhT: cute.Tensor,
        m_bdv: cute.Tensor,
        m_bdh_pre: cute.Tensor,
        m_bga_s: cute.Tensor,
        m_dv2: cute.Tensor,
        m_dh: cute.Tensor,
        m_dvl: cute.Tensor,
        m_bdv_s: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        c: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
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
            cpasync.prefetch_descriptor(tma_ag1)
            cpasync.prefetch_descriptor(tma_bdhT)

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

        gA = cute.local_tile(m_ag1, _MNK_TILER, coord, proj=(1, None, 1))
        gB = cute.local_tile(m_bdhT, _MNK_TILER, coord, proj=(None, 1, 1))
        gC = cute.local_tile(m_bdv, _MNK_TILER, coord, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)
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

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc_frag.layout)

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

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc_empty = acc_prod.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_ag1,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bdhT,
                    tBgB[(None, ab_e.count)],
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

        # relinquish AFTER acc_empty.commit() — byte-fidelity to _gemm_kernel_b:407-409.
        tmem.relinquish_alloc_permit()
        acc_f = acc_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_f.release()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem.retrieve_ptr(_acc))

        # dh snapshot: full D_K=128 rows of b_dh BEFORE this chunk's carry update.
        for e in cutlass.range(tidx, D_K * d_v, THREADS):
            r, col = e // d_v, e % d_v
            m_dh[bidz, r, col] = m_bdh_pre[bidz, r, col].to(_io)

        # b_dv glue: only rows [:c] of the G1 raw output are real (k_dec zero-padded to 128).
        for e in cutlass.range(tidx, c * d_v, THREADS):
            r, col = e // d_v, e % d_v
            val = m_bdv_s[bidz, r, col].to(_acc) + m_dvl[bidz, r, col]
            m_dv2[bidz, r, col] = val.to(_io)
            # b_dv^T into GA's 2nd half; fp16 round-trip matches TMA read precision.
            m_bga_s[bidz, col, c + r] = val.to(_io)

        cute.arch.fence_proxy("async.global")
        cute.arch.barrier()

    # ------------------------------------------------------------------
    # _kern_ga_epi: GA GEMM + b_dh carry update.
    #
    # a_ga[Z,128,128] @ b_ga[Z,64,128]^T → t[Z,128,64]
    # SIMT carry over d_k*d_v elements; fence_proxy then barrier.
    # relinquish_alloc_permit: AFTER acc_empty.commit().
    # ------------------------------------------------------------------
    @cute.kernel
    def _kern_ga_epi(
        tiled_mma: cute.TiledMma,
        tma_aga: cute.CopyAtom,
        m_aga: cute.Tensor,
        tma_bga: cute.CopyAtom,
        m_bga: cute.Tensor,
        m_t: cute.Tensor,
        m_bdh: cute.Tensor,
        m_bdhT_s: cute.Tensor,
        m_glast: cute.Tensor,
        m_t_s: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        d_k: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
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
            cpasync.prefetch_descriptor(tma_aga)
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

        gA = cute.local_tile(m_aga, _MNK_TILER, coord, proj=(1, None, 1))
        gB = cute.local_tile(m_bga, _MNK_TILER, coord, proj=(None, 1, 1))
        gC = cute.local_tile(m_t, _MNK_TILER, coord, proj=(1, 1, None))
        thr_mma = tiled_mma.get_slice(0)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)
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

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc_frag.layout)

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

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc_empty = acc_prod.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_aga,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bga,
                    tBgB[(None, ab_e.count)],
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

        # relinquish AFTER acc_empty.commit() — byte-fidelity to _gemm_kernel_b:407-409.
        tmem.relinquish_alloc_permit()
        acc_f = acc_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC, mode=[2])):
            cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
            tCrC.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC, tDgC[None, None, i])
        acc_f.release()
        pipeline.sync(barrier_id=1)
        tmem.free(tmem.retrieve_ptr(_acc))

        for e in cutlass.range(tidx, d_k * d_v, THREADS):
            r, col = e // d_v, e % d_v
            val = m_glast[bidz, r] * m_bdh[bidz, r, col] + m_t_s[bidz, r, col].to(_acc)
            m_bdh[bidz, r, col] = val
            # fp16 round-trip: b_dhT is the next G1's TMA B operand.
            m_bdhT_s[bidz, col, r] = val.to(_io)
        cute.arch.fence_proxy("async.global")
        cute.arch.barrier()

    @cute.jit
    def _host_g1_epi(
        a_g1: cute.Tensor,
        b_dhT: cute.Tensor,
        bdv_raw: cute.Tensor,
        b_dh_pre: cute.Tensor,
        b_ga_s: cute.Tensor,
        dv2: cute.Tensor,
        dh: cute.Tensor,
        dvl: cute.Tensor,
        bdv_raw_s: cute.Tensor,
        c: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
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
            tiled_mma, _MNK_TILER, a_g1.element_type, AB_STAGES
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, _MNK_TILER, b_dhT.element_type, AB_STAGES
        )
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a_g1, a1, _MNK_TILER, tiled_mma)
        b_atom, b_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b_dhT, b1, _MNK_TILER, tiled_mma)
        grid = (
            cute.ceil_div(bdv_raw.layout.shape[0], _MNK_TILER[0]),
            cute.ceil_div(bdv_raw.layout.shape[1], _MNK_TILER[1]),
            bdv_raw.layout.shape[2],
        )
        _kern_g1_epi(
            tiled_mma,
            a_atom,
            a_t,
            b_atom,
            b_t,
            bdv_raw,
            b_dh_pre,
            b_ga_s,
            dv2,
            dh,
            dvl,
            bdv_raw_s,
            a_layout,
            b_layout,
            c,
            d_v,
        ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)

    @cute.jit
    def _host_ga_epi(
        a_ga: cute.Tensor,
        b_ga: cute.Tensor,
        t_scr: cute.Tensor,
        b_dh: cute.Tensor,
        b_dhT_s: cute.Tensor,
        glast: cute.Tensor,
        t_scr_s: cute.Tensor,
        d_k: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
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
            tiled_mma, _MNK_TILER, a_ga.element_type, AB_STAGES
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, _MNK_TILER, b_ga.element_type, AB_STAGES
        )
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a_ga, a1, _MNK_TILER, tiled_mma)
        b_atom, b_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b_ga, b1, _MNK_TILER, tiled_mma)
        grid = (
            cute.ceil_div(t_scr.layout.shape[0], _MNK_TILER[0]),
            cute.ceil_div(t_scr.layout.shape[1], _MNK_TILER[1]),
            t_scr.layout.shape[2],
        )
        _kern_ga_epi(
            tiled_mma,
            a_atom,
            a_t,
            b_atom,
            b_t,
            t_scr,
            b_dh,
            b_dhT_s,
            glast,
            t_scr_s,
            a_layout,
            b_layout,
            d_k,
            d_v,
        ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)

    # compile-cache keyed (n_bh, c, d_v) — shapes are static across steps.
    _g1_cache: dict[tuple[int, int, int], object] = {}
    _ga_cache: dict[tuple[int, int, int], object] = {}


def _modelled_l2(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure-torch model of _kern_g1_epi + _kern_ga_epi per step (the kernel spec).

    Includes fp16 cast round-trips for all kernel I/O: operand casts, GEMM outputs
    (bdv_raw, t_scr land fp16 in GMEM before SIMT re-reads), b_ga/b_dhT stores.
    fp32 GEMM stand-ins (no tcgen05). In fp64 input tracks the cw bundle within the fp16
    quantisation floor (~3e-4 relative).
    """
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    dev = q.device
    f16, f32 = torch.float16, torch.float32

    from lethe.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw as _pack

    buf = _pack(q, k, wy, g2, g_last, do, dv_local)
    a_g1_L = buf["a_g1"]
    a_ga_L = buf["a_ga"]
    b_ga_L = buf["b_ga"]
    dvl_L = buf["dv_local"]
    glast_L = buf["glast"]

    b_dh = dht.reshape(n_bh, d_k, d_v).clone().to(f32)
    # fp16 round-trip: b_dhT is the TMA B operand for G1.
    b_dhT = b_dh.to(f16).transpose(-1, -2).contiguous()

    dh_out = torch.zeros(nt, n_bh, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(nt, n_bh, c, d_v, dtype=f16, device=dev)

    for it in reversed(range(nt)):
        # Operand casts: f16 then back to f32 mirrors kernel fp16 GEMM precision.
        a_g1_step = a_g1_L.reshape(n_bh, nt, d_k, d_k)[:, it].to(f16).to(f32)
        a_ga_step = a_ga_L.reshape(n_bh, nt, d_k, 2 * c)[:, it].to(f16).to(f32)
        b_ga_step = b_ga_L.reshape(n_bh, nt, d_v, 2 * c)[:, it].to(f16).clone()
        dvl_step = dvl_L.reshape(n_bh, nt, c, d_v)[:, it].to(f32)
        gl_step = glast_L.reshape(n_bh, nt, d_k)[:, it].to(f32)

        dh_out[it] = b_dh.to(f16)

        raw = a_g1_step @ b_dhT.to(f32).transpose(-1, -2)
        raw = raw.to(f16).to(f32)
        b_dv = raw[:, :c] + dvl_step
        dv2_out[it] = b_dv.to(f16)
        # b_dv^T round-trip into GA's 2nd half; fp16 matches in-kernel SIMT store dtype.
        b_ga_step[:, :, c:] = b_dv.to(f16).transpose(-1, -2)

        t = a_ga_step @ b_ga_step.to(f32).transpose(-1, -2)
        t = t.to(f16).to(f32)
        b_dh = gl_step[:, :, None] * b_dh + t
        # fp16 round-trip: b_dhT is the next G1's TMA operand; precision floor matches kernel.
        b_dhT = b_dh.to(f16).transpose(-1, -2).contiguous()

    return (
        dh_out.permute(1, 0, 2, 3).reshape(b, hv, nt, d_k, d_v),
        dv2_out.permute(1, 0, 2, 3).reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


# ------------------------------------------------------------------
# Host launcher: run_k1_incB_l2
# ------------------------------------------------------------------


def run_k1_incB_l2(
    q: Tensor,
    k: Tensor,
    wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Level-2 channel-wise reverse scan: 2 kernel launches per step, 0 torch compute ops.

    Pre-packs once with :func:`_incb2_pack_cw`, restages to step-major layout so per-step
    slices are zero-copy contiguous views, then per reverse chunk ``it`` fires exactly:
    - :func:`_kern_g1_epi`: G1 GEMM + dh snapshot + dv2 + b_ga glue.
    - :func:`_kern_ga_epi`: GA GEMM + b_dh/b_dhT carry update.
    compile-cache keyed (n_bh, c, D_V) — static across all steps. Returns ``(dh, dv2, dh0)``
    head-major chunked.
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    dev = q.device
    f16, f32 = torch.float16, torch.float32

    # _MNK_TILER bakes M=D_K=128, N=D_V=64, K=2*CHUNK=128.
    if c != CHUNK or d_k != D_K:
        raise ValueError(f"c={c} d_k={d_k}: require c={CHUNK}, d_k={D_K} (baked into _MNK_TILER)")
    if d_v != D_V:
        raise ValueError(f"d_v={d_v}: only d_v={D_V} supported; d_v=128 needs two-tile N-loop")

    from lethe.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw as _pack

    buf = _pack(q, k, wy, g2, g_last, do, dv_local)

    # Restage ONCE to step-major [nt, n_bh, ...] so buf_sm[it] is a contiguous zero-copy view.
    def _sm(x: Tensor, *trailing: int) -> Tensor:
        return x.reshape(n_bh, nt, *trailing).transpose(0, 1).contiguous()

    ag1_sm = _sm(buf["a_g1"].to(f16), D_K, D_K)  # [nt, n_bh, 128, 128]
    aga_sm = _sm(buf["a_ga"].to(f16), D_K, 2 * c)  # [nt, n_bh, 128, 2C]
    bga_sm = _sm(buf["b_ga"].to(f16), d_v, 2 * c)  # [nt, n_bh, 64, 2C]
    dvl_sm = _sm(buf["dv_local"].to(f32), c, d_v)  # [nt, n_bh, C, 64]
    gl_sm = _sm(buf["glast"].to(f32), d_k)  # [nt, n_bh, 128]

    b_dh = (
        dht.reshape(n_bh, d_k, d_v).clone().to(f32)
    )  # clone: .contiguous().to() no-ops on contiguous fp32 -> would alias + mutate caller dht
    b_dhT = b_dh.to(f16).transpose(-1, -2).contiguous()  # [n_bh, d_v, d_k]

    # Step-major outputs: dh_out[it] and dv2_out[it] are contiguous zero-copy views.
    dh_out = torch.zeros(nt, n_bh, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(nt, n_bh, c, d_v, dtype=f16, device=dev)

    # Hoisted scratch: each launch fully overwrites these before SIMT reads them.
    bdv_raw = torch.zeros(n_bh, D_K, D_V, dtype=f16, device=dev)
    t_scr = torch.zeros(n_bh, D_K, D_V, dtype=f16, device=dev)

    # Constexpr args (c, D_V, d_k) are baked at cute.compile time and DROPPED from the
    # runtime arg list by the executor — do NOT re-pass them at call time.
    # Only marked cute.Tensors and the CUstream are runtime args (same as _gemm_batched).
    key = (n_bh, c, D_V)
    stream = _cur_stream()

    for it in reversed(range(nt)):
        ag1_step = ag1_sm[it]
        aga_step = aga_sm[it]
        bga_step = bga_sm[it]
        dvl_step = dvl_sm[it]
        gl_step = gl_sm[it]

        # compile-args include Constexpr ints (baked); call-args omit them.
        g1_compile = (
            _mark_b(ag1_step),
            _mark_b(b_dhT),
            _mark_b(bdv_raw),
            _mark_simt(b_dh),
            _mark_simt(bga_step),
            _mark_simt(dv2_out[it]),
            _mark_simt(dh_out[it]),
            _mark_simt(dvl_step),
            _mark_simt(bdv_raw),
            c,
            D_V,
            stream,
        )
        ex_g1 = _g1_cache.get(key)
        if ex_g1 is None:
            ex_g1 = cute.compile(_host_g1_epi, *g1_compile)
            _g1_cache[key] = ex_g1
        g1_call = (
            _mark_b(ag1_step),
            _mark_b(b_dhT),
            _mark_b(bdv_raw),
            _mark_simt(b_dh),
            _mark_simt(bga_step),
            _mark_simt(dv2_out[it]),
            _mark_simt(dh_out[it]),
            _mark_simt(dvl_step),
            _mark_simt(bdv_raw),
            stream,
        )
        ex_g1(*g1_call)

        ga_compile = (
            _mark_b(aga_step),
            _mark_b(bga_step),
            _mark_b(t_scr),
            _mark_simt(b_dh),
            _mark_simt(b_dhT),
            _mark_simt(gl_step),
            _mark_simt(t_scr),
            d_k,
            D_V,
            stream,
        )
        ex_ga = _ga_cache.get(key)
        if ex_ga is None:
            ex_ga = cute.compile(_host_ga_epi, *ga_compile)
            _ga_cache[key] = ex_ga
        ga_call = (
            _mark_b(aga_step),
            _mark_b(bga_step),
            _mark_b(t_scr),
            _mark_simt(b_dh),
            _mark_simt(b_dhT),
            _mark_simt(gl_step),
            _mark_simt(t_scr),
            stream,
        )
        ex_ga(*ga_call)

    from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync

    maybe_sync()
    return (
        dh_out.permute(1, 0, 2, 3).reshape(b, hv, nt, d_k, d_v),
        dv2_out.permute(1, 0, 2, 3).reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )
