"""Two-GEMM TMEM-offset probe — the inc-B2 unblock gate (mamba2_ssd lifecycle).

Every faulting cell in the inc-B2 bisect matrix shared ONE lifecycle: a full 512-column
``tmem.allocate`` + ``relinquish`` per GEMM — illegal PTX (alloc-after-relinquish) or a
blocking second alloc. NVIDIA's own ``mamba2_ssd.py`` in the SAME pinned 4.5.2 wheel runs
4 tcgen05 MMAs per kernel via ONE alloc + static per-accumulator column offsets
(``tcgen05.find_tmem_tensor_col_offset``) + ONE relinquish at kernel end
(``scratch/refs/mamba2_ssd_v452.py``). This probe transplants that lifecycle onto our
proven single-GEMM scaffolding (``loop_gemm_repro.py``, silicon-GO) in three modes:

  independent  TWO GEMMs (C0 = A@B0^T, C1 = A@B1^T) back-to-back, one alloc, acc0 at
               column offset 0 / acc1 at offset ncols(acc0), two tiled_mma objects,
               distinct acc mbarriers, relinquish+free ONCE  → the lifecycle verdict
  looped       the same two accumulators inside ``cutlass.range(NT)`` (real scf.for):
               acc0 recomputed per iteration, acc1 a LOOP-CARRIED TMEM accumulator
               (C1 = NT·A@B1^T) with handles re-derived in-loop (the loopfix idiom)
               → the reverse-recurrence structure inc-B2 needs
  dependent    GEMM0 → acc0 → tcgen05.ld → f16 → tcgen05.st into a TMEM A-operand
               region → GEMM1 (A from TMEM, mamba2_ssd's intra1→intra2 staging chain):
               C1 = f16(A@B0^T) @ B1^T  → the fused-chain data flow

Interpretation: independent+looped GO ⇒ the DSL wall is falsified and run_k1_incB2
ports onto the offset-partition pattern; dependent GO additionally proves the staging
chain for full fusion. dependent is best-effort (fragment-correspondence between the
Ld32x32b load and St16x128b store partitionings is assumed elementwise, as mamba2_ssd's
segsum does) — a dependent-only failure indicts the probe's staging, not the lifecycle.

Run on the box (one process per mode):
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/tmem_offset_probe.py --mode independent --out ...
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/tmem_offset_probe.py --mode looped --out ...
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/tmem_offset_probe.py --mode dependent --out ...
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes @cute.struct fields.

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

try:
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

M = 128
N = 64
K = 128
K2 = 64  # dependent-mode GEMM1 contraction dim (= N of GEMM0)
NT = 4
THREADS = 128
AB_STAGES = 2

if _HAVE:
    _io = cutlass.Float16
    _acc = cutlass.Float32
    _MNK_INST = (M, N, 16)
    _TILER0 = (M, N, K)
    _TILER1 = (M, N, K2)
    _AB_MBAR = AB_STAGES * 2

    @cute.struct
    class _Smem:
        ab_mbar: cute.struct.MemRange[cutlass.Int64, _AB_MBAR]
        acc0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        acc1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_buf: cutlass.Int32

    @cute.kernel
    def _kern_independent(
        mma0: cute.TiledMma,
        mma1: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b0: cute.CopyAtom,
        mB0: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        mC0: cute.Tensor,
        mC1: cute.Tensor,
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
        sB0 = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
        sB1 = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)

        acc_shape0 = mma0.partition_shape_C(_TILER0[:2])
        tCtAcc0_frag = mma0.make_fragment_C(acc_shape0)
        acc_shape1 = mma1.partition_shape_C(_TILER0[:2])
        tCtAcc1_frag = mma1.make_fragment_C(acc_shape1)
        ncols0 = 64  # (128,64) f32 acc = 64 cols (tile_N * stages)
        tmem.allocate(128)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b0)
            cpasync.prefetch_descriptor(tma_b1)

        nbytes = cute.size_in_bytes(
            _io, cute.select(a_layout, mode=[0, 1, 2])
        ) + 2 * cute.size_in_bytes(_io, cute.select(b_layout, mode=[0, 1, 2]))
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc0_prod, acc0_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc0_mbar.data_ptr(),
        ).make_participants()
        acc1_prod, acc1_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc1_mbar.data_ptr(),
        ).make_participants()

        gA = cute.local_tile(mA, _TILER0, coord, proj=(1, None, 1))
        gB0 = cute.local_tile(mB0, _TILER0, coord, proj=(None, 1, 1))
        gB1 = cute.local_tile(mB1, _TILER0, coord, proj=(None, 1, 1))
        gC0 = cute.local_tile(mC0, _TILER0, coord, proj=(1, 1, None))
        gC1 = cute.local_tile(mC1, _TILER0, coord, proj=(1, 1, None))
        thr_mma0 = mma0.get_slice(0)
        thr_mma1 = mma1.get_slice(0)
        tCgC0 = thr_mma0.partition_C(gC0)
        tCgC1 = thr_mma1.partition_C(gC1)
        tCrA = mma0.make_fragment_A(sA)
        tCrB0 = mma0.make_fragment_B(sB0)
        tCrB1 = mma1.make_fragment_B(sB1)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma0.partition_A(gA), 0, 3),
        )
        tB0s, tB0g = cute.nvgpu.cpasync.tma_partition(
            tma_b0,
            0,
            cute.make_layout(1),
            cute.group_modes(sB0, 0, 3),
            cute.group_modes(thr_mma0.partition_B(gB0), 0, 3),
        )
        tB1s, tB1g = cute.nvgpu.cpasync.tma_partition(
            tma_b1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(thr_mma1.partition_B(gB1), 0, 3),
        )

        tmem.wait_for_alloc()
        base = tmem.retrieve_ptr(_acc)
        tCtAcc0 = cute.make_tensor(base, tCtAcc0_frag.layout)
        tCtAcc1 = cute.make_tensor(base + ncols0, tCtAcc1_frag.layout)

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc0_empty = acc0_prod.acquire_and_advance()
            acc1_empty = acc1_prod.acquire_and_advance()
            mma0.set(tcgen05.Field.ACCUMULATE, False)
            mma1.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_a,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b0,
                    tB0g[(None, ab_e.count)],
                    tB0s[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b1,
                    tB1g[(None, ab_e.count)],
                    tB1s[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    c = (None, None, kb, ab_f.index)
                    cute.gemm(mma0, tCtAcc0, tCrA[c], tCrB0[c], tCtAcc0)
                    mma0.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(mma1, tCtAcc1, tCrA[c], tCrB1[c], tCtAcc1)
                    mma1.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            acc0_empty.commit()
            acc1_empty.commit()

        tmem.relinquish_alloc_permit()

        epi_tiler = ((cute.size(tCtAcc0, mode=[0, 0]), cute.size(tCtAcc0, mode=[0, 1])),)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)

        tCtAcc0_epi = cute.zipped_divide(tCtAcc0, epi_tiler)
        gC0_epi = cute.zipped_divide(tCgC0, epi_tiler)
        tmem_copy0 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc0_epi[None, 0])
        thr0 = tmem_copy0.get_slice(tidx)
        tDtC0 = thr0.partition_S(tCtAcc0_epi)
        tDgC0 = thr0.partition_D(gC0_epi)
        tCrAcc0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _acc)
        tCrC0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _io)
        acc0_f = acc0_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC0, mode=[2])):
            cute.copy(tmem_copy0, tDtC0[None, None, i], tCrAcc0)
            tCrC0.store(tCrAcc0.load().to(_io))
            cute.autovec_copy(tCrC0, tDgC0[None, None, i])
        acc0_f.release()

        tCtAcc1_epi = cute.zipped_divide(tCtAcc1, epi_tiler)
        gC1_epi = cute.zipped_divide(tCgC1, epi_tiler)
        tmem_copy1 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc1_epi[None, 0])
        thr1 = tmem_copy1.get_slice(tidx)
        tDtC1 = thr1.partition_S(tCtAcc1_epi)
        tDgC1 = thr1.partition_D(gC1_epi)
        tCrAcc1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _acc)
        tCrC1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _io)
        acc1_f = acc1_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC1, mode=[2])):
            cute.copy(tmem_copy1, tDtC1[None, None, i], tCrAcc1)
            tCrC1.store(tCrAcc1.load().to(_io))
            cute.autovec_copy(tCrC1, tDgC1[None, None, i])
        acc1_f.release()

        pipeline.sync(barrier_id=1)
        tmem.free(base)

    @cute.kernel
    def _kern_looped(
        mma0: cute.TiledMma,
        mma1: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b0: cute.CopyAtom,
        mB0: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        mC0: cute.Tensor,
        mC1: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
    ) -> None:
        # The inc-B2 structure cell: a REAL scf.for whose body issues both GEMMs, acc1
        # loop-CARRIED in TMEM across iterations (C1 = NT*A@B1^T), handles re-derived
        # in-loop (loopfix idiom), operands loaded once and held for the whole loop.
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        coord = (bidx, bidy, None)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB0 = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
        sB1 = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)

        acc_shape0 = mma0.partition_shape_C(_TILER0[:2])
        tCtAcc0_frag = mma0.make_fragment_C(acc_shape0)
        acc_shape1 = mma1.partition_shape_C(_TILER0[:2])
        tCtAcc1_frag = mma1.make_fragment_C(acc_shape1)
        ncols0 = 64  # (128,64) f32 acc = 64 cols (tile_N * stages)
        tmem.allocate(128)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b0)
            cpasync.prefetch_descriptor(tma_b1)

        nbytes = cute.size_in_bytes(
            _io, cute.select(a_layout, mode=[0, 1, 2])
        ) + 2 * cute.size_in_bytes(_io, cute.select(b_layout, mode=[0, 1, 2]))
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc0_prod, acc0_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc0_mbar.data_ptr(),
        ).make_participants()
        acc1_prod, acc1_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc1_mbar.data_ptr(),
        ).make_participants()

        gA = cute.local_tile(mA, _TILER0, coord, proj=(1, None, 1))
        gB0 = cute.local_tile(mB0, _TILER0, coord, proj=(None, 1, 1))
        gB1 = cute.local_tile(mB1, _TILER0, coord, proj=(None, 1, 1))
        gC0 = cute.local_tile(mC0, _TILER0, coord, proj=(1, 1, None))
        gC1 = cute.local_tile(mC1, _TILER0, coord, proj=(1, 1, None))
        thr_mma0 = mma0.get_slice(0)
        thr_mma1 = mma1.get_slice(0)
        tCgC0 = thr_mma0.partition_C(gC0)
        tCgC1 = thr_mma1.partition_C(gC1)
        tCrA = mma0.make_fragment_A(sA)
        tCrB0 = mma0.make_fragment_B(sB0)
        tCrB1 = mma1.make_fragment_B(sB1)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma0.partition_A(gA), 0, 3),
        )
        tB0s, tB0g = cute.nvgpu.cpasync.tma_partition(
            tma_b0,
            0,
            cute.make_layout(1),
            cute.group_modes(sB0, 0, 3),
            cute.group_modes(thr_mma0.partition_B(gB0), 0, 3),
        )
        tB1s, tB1g = cute.nvgpu.cpasync.tma_partition(
            tma_b1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(thr_mma1.partition_B(gB1), 0, 3),
        )

        tmem.wait_for_alloc()

        if warp_idx == 0:
            acc0_empty = acc0_prod.acquire_and_advance()
            acc1_empty = acc1_prod.acquire_and_advance()
            ab_e = ab_prod.acquire_and_advance()
            cute.copy(
                tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
            )
            cute.copy(
                tma_b0, tB0g[(None, ab_e.count)], tB0s[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
            )
            cute.copy(
                tma_b1, tB1g[(None, ab_e.count)], tB1s[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
            )
            ab_f = ab_cons.wait_and_advance()
            mma1.set(tcgen05.Field.ACCUMULATE, False)
            for _it in cutlass.range(NT):
                base_it = tmem.retrieve_ptr(_acc)
                tCtAcc0 = cute.make_tensor(base_it, tCtAcc0_frag.layout)
                tCtAcc1 = cute.make_tensor(base_it + ncols0, tCtAcc1_frag.layout)
                mma0.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    c = (None, None, kb, ab_f.index)
                    cute.gemm(mma0, tCtAcc0, tCrA[c], tCrB0[c], tCtAcc0)
                    mma0.set(tcgen05.Field.ACCUMULATE, True)
                    cute.gemm(mma1, tCtAcc1, tCrA[c], tCrB1[c], tCtAcc1)
                    mma1.set(tcgen05.Field.ACCUMULATE, True)
            ab_f.release()
            acc0_empty.commit()
            acc1_empty.commit()

        tmem.relinquish_alloc_permit()

        base = tmem.retrieve_ptr(_acc)
        tCtAcc0_out = cute.make_tensor(base, tCtAcc0_frag.layout)
        tCtAcc1_out = cute.make_tensor(base + ncols0, tCtAcc1_frag.layout)
        epi_tiler = ((cute.size(tCtAcc0_out, mode=[0, 0]), cute.size(tCtAcc0_out, mode=[0, 1])),)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)

        tCtAcc0_epi = cute.zipped_divide(tCtAcc0_out, epi_tiler)
        gC0_epi = cute.zipped_divide(tCgC0, epi_tiler)
        tmem_copy0 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc0_epi[None, 0])
        thr0 = tmem_copy0.get_slice(tidx)
        tDtC0 = thr0.partition_S(tCtAcc0_epi)
        tDgC0 = thr0.partition_D(gC0_epi)
        tCrAcc0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _acc)
        tCrC0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _io)
        acc0_f = acc0_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC0, mode=[2])):
            cute.copy(tmem_copy0, tDtC0[None, None, i], tCrAcc0)
            tCrC0.store(tCrAcc0.load().to(_io))
            cute.autovec_copy(tCrC0, tDgC0[None, None, i])
        acc0_f.release()

        tCtAcc1_epi = cute.zipped_divide(tCtAcc1_out, epi_tiler)
        gC1_epi = cute.zipped_divide(tCgC1, epi_tiler)
        tmem_copy1 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc1_epi[None, 0])
        thr1 = tmem_copy1.get_slice(tidx)
        tDtC1 = thr1.partition_S(tCtAcc1_epi)
        tDgC1 = thr1.partition_D(gC1_epi)
        tCrAcc1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _acc)
        tCrC1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _io)
        acc1_f = acc1_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC1, mode=[2])):
            cute.copy(tmem_copy1, tDtC1[None, None, i], tCrAcc1)
            tCrC1.store(tCrAcc1.load().to(_io))
            cute.autovec_copy(tCrC1, tDgC1[None, None, i])
        acc1_f.release()

        pipeline.sync(barrier_id=1)
        tmem.free(base)

    @cute.kernel
    def _kern_dependent(
        mma0: cute.TiledMma,
        mma1: cute.TiledMma,
        tma_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_b0: cute.CopyAtom,
        mB0: cute.Tensor,
        tma_b1: cute.CopyAtom,
        mB1: cute.Tensor,
        mC0: cute.Tensor,
        mC1: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b0_layout: cute.ComposedLayout,
        b1_layout: cute.ComposedLayout,
        a2_tmem_layout: cute.ComposedLayout,
    ) -> None:
        # GEMM0 (A@B0^T, K=128) -> acc0; stage f16(acc0) into a TMEM A-operand region
        # (mamba2_ssd intra1->intra2 idiom: Ld32x32b -> rmem -> St16x128b -> fence);
        # GEMM1 (A from TMEM, B1 from smem, K=64) -> acc1. One alloc, one relinquish.
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        coord = (bidx, bidy, None)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB0 = smem.allocate_tensor(
            _io, b0_layout.outer, byte_alignment=128, swizzle=b0_layout.inner
        )
        sB1 = smem.allocate_tensor(
            _io, b1_layout.outer, byte_alignment=128, swizzle=b1_layout.inner
        )

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)

        acc_shape0 = mma0.partition_shape_C(_TILER0[:2])
        tCtAcc0_frag = mma0.make_fragment_C(acc_shape0)
        acc_shape1 = mma1.partition_shape_C(_TILER1[:2])
        tCtAcc1_frag = mma1.make_fragment_C(acc_shape1)
        tCrA2_fake = mma1.make_fragment_A(a2_tmem_layout.outer.shape)
        ncols0 = 64  # (128,64) f32 acc
        ncols_a2 = 32  # f16 [128,64] operand region = 64*16/32 cols
        tmem.allocate(256)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b0)
            cpasync.prefetch_descriptor(tma_b1)

        nbytes = (
            cute.size_in_bytes(_io, cute.select(a_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(_io, cute.select(b0_layout, mode=[0, 1, 2]))
            + cute.size_in_bytes(_io, cute.select(b1_layout, mode=[0, 1, 2]))
        )
        ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
            num_stages=AB_STAGES,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            tx_count=nbytes,
            barrier_storage=storage.ab_mbar.data_ptr(),
        ).make_participants()
        acc0_prod, acc0_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc0_mbar.data_ptr(),
        ).make_participants()
        acc1_prod, acc1_cons = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
            barrier_storage=storage.acc1_mbar.data_ptr(),
        ).make_participants()

        gA = cute.local_tile(mA, _TILER0, coord, proj=(1, None, 1))
        gB0 = cute.local_tile(mB0, _TILER0, coord, proj=(None, 1, 1))
        gB1 = cute.local_tile(mB1, _TILER1, coord, proj=(None, 1, 1))
        gC0 = cute.local_tile(mC0, _TILER0, coord, proj=(1, 1, None))
        gC1 = cute.local_tile(mC1, _TILER1, coord, proj=(1, 1, None))
        thr_mma0 = mma0.get_slice(0)
        thr_mma1 = mma1.get_slice(0)
        tCgC0 = thr_mma0.partition_C(gC0)
        tCgC1 = thr_mma1.partition_C(gC1)
        tCrA = mma0.make_fragment_A(sA)
        tCrB0 = mma0.make_fragment_B(sB0)
        tCrB1 = mma1.make_fragment_B(sB1)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma0.partition_A(gA), 0, 3),
        )
        tB0s, tB0g = cute.nvgpu.cpasync.tma_partition(
            tma_b0,
            0,
            cute.make_layout(1),
            cute.group_modes(sB0, 0, 3),
            cute.group_modes(thr_mma0.partition_B(gB0), 0, 3),
        )
        tB1s, tB1g = cute.nvgpu.cpasync.tma_partition(
            tma_b1,
            0,
            cute.make_layout(1),
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(thr_mma1.partition_B(gB1), 0, 3),
        )

        tmem.wait_for_alloc()
        base = tmem.retrieve_ptr(_acc)
        tCtAcc0 = cute.make_tensor(base, tCtAcc0_frag.layout)
        tCrA2 = cute.make_tensor(cute.recast_ptr(base + ncols0, dtype=_io), tCrA2_fake.layout)
        tCtAcc1 = cute.make_tensor(base + ncols0 + ncols_a2, tCtAcc1_frag.layout)

        nk = cute.size(gA, mode=[2])
        if warp_idx == 0:
            acc0_empty = acc0_prod.acquire_and_advance()
            mma0.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_a,
                    tAgA[(None, ab_e.count)],
                    tAsA[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b0,
                    tB0g[(None, ab_e.count)],
                    tB0s[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_b1,
                    tB1g[(None, ab_e.count)],
                    tB1s[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    c = (None, None, kb, ab_f.index)
                    cute.gemm(mma0, tCtAcc0, tCrA[c], tCrB0[c], tCtAcc0)
                    mma0.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            acc0_empty.commit()

        # Stage acc0 -> f16 -> TMEM A2 region (all 128 threads), then C0 epilogue.
        epi_tiler0 = ((cute.size(tCtAcc0, mode=[0, 0]), cute.size(tCtAcc0, mode=[0, 1])),)
        ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)
        tCtAcc0_epi = cute.zipped_divide(tCtAcc0, epi_tiler0)
        gC0_epi = cute.zipped_divide(tCgC0, epi_tiler0)
        tmem_copy0 = tcgen05.make_tmem_copy(ld_atom, tCtAcc0_epi[None, 0])
        thr0 = tmem_copy0.get_slice(tidx)
        tDtC0 = thr0.partition_S(tCtAcc0_epi)
        tDgC0 = thr0.partition_D(gC0_epi)
        tCrAcc0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _acc)
        tCrC0 = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _io)

        st_atom = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE), _io
        )
        tiled_r2t = tcgen05.make_tmem_copy(st_atom, tCrA2)
        thr_r2t = tiled_r2t.get_slice(tidx)
        tRT_t = thr_r2t.partition_D(tCrA2)
        tRT_r = cute.make_rmem_tensor(thr_r2t.partition_S(tCrA2).shape, _io)

        acc0_f = acc0_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC0, mode=[2])):
            cute.copy(tmem_copy0, tDtC0[None, None, i], tCrAcc0)
            tCrC0.store(tCrAcc0.load().to(_io))
            cute.autovec_copy(tCrC0, tDgC0[None, None, i])
        # Reload the full accumulator per-thread and store as the f16 TMEM operand.
        # Fragment correspondence (Ld32x32b ownership == St16x128b ownership over the
        # same [M, K2] region) is asserted host-side via shape equality before launch.
        cute.copy(tmem_copy0, tDtC0[None, None, 0], tCrAcc0)
        tRT_r.store(tCrAcc0.load().to(_io))
        cute.arch.fence_view_async_tmem_load()
        cute.copy(tiled_r2t, tRT_r, tRT_t)
        cute.arch.fence_view_async_tmem_store()
        acc0_f.release()
        pipeline.sync(barrier_id=1)

        if warp_idx == 0:
            acc1_empty = acc1_prod.acquire_and_advance()
            mma1.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(cute.size(tCrA2, mode=[2])):
                c1 = (None, None, kb, 0)
                cute.gemm(mma1, tCtAcc1, tCrA2[c1], tCrB1[c1], tCtAcc1)
                mma1.set(tcgen05.Field.ACCUMULATE, True)
            acc1_empty.commit()

        tmem.relinquish_alloc_permit()

        epi_tiler1 = ((cute.size(tCtAcc1, mode=[0, 0]), cute.size(tCtAcc1, mode=[0, 1])),)
        tCtAcc1_epi = cute.zipped_divide(tCtAcc1, epi_tiler1)
        gC1_epi = cute.zipped_divide(tCgC1, epi_tiler1)
        tmem_copy1 = tcgen05.make_tmem_copy(ld_atom, tCtAcc1_epi[None, 0])
        thr1 = tmem_copy1.get_slice(tidx)
        tDtC1 = thr1.partition_S(tCtAcc1_epi)
        tDgC1 = thr1.partition_D(gC1_epi)
        tCrAcc1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _acc)
        tCrC1 = cute.make_rmem_tensor(tDgC1[None, None, 0].shape, _io)
        acc1_f = acc1_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tDtC1, mode=[2])):
            cute.copy(tmem_copy1, tDtC1[None, None, i], tCrAcc1)
            tCrC1.store(tCrAcc1.load().to(_io))
            cute.autovec_copy(tCrC1, tDgC1[None, None, i])
        acc1_f.release()

        pipeline.sync(barrier_id=1)
        tmem.free(base)

    def _mk_mma(a_src: object) -> object:
        return cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                _io,
                _acc,
                _MNK_INST,
                tcgen05.CtaGroup.ONE,
                a_src,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )

    def _make_host_two(kern: object) -> object:
        @cute.jit
        def _host(
            a: cute.Tensor,
            b0: cute.Tensor,
            b1: cute.Tensor,
            c0: cute.Tensor,
            c1: cute.Tensor,
        ) -> None:
            mma0 = _mk_mma(tcgen05.OperandSource.SMEM)
            mma1 = _mk_mma(tcgen05.OperandSource.SMEM)
            a_layout = sm100_utils.make_smem_layout_a(mma0, _TILER0, a.element_type, AB_STAGES)
            b_layout = sm100_utils.make_smem_layout_b(mma0, _TILER0, b0.element_type, AB_STAGES)
            a1 = cute.select(a_layout, mode=[0, 1, 2])
            bl1 = cute.select(b_layout, mode=[0, 1, 2])
            op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, _TILER0, mma0)
            b0_atom, b0_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b0, bl1, _TILER0, mma0)
            b1_atom, b1_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b1, bl1, _TILER0, mma1)
            grid = cute.ceil_div((*c0.layout.shape, 1), _TILER0[:2])
            kern(
                mma0,
                mma1,
                a_atom,
                a_t,
                b0_atom,
                b0_t,
                b1_atom,
                b1_t,
                c0,
                c1,
                a_layout,
                b_layout,
            ).launch(grid=grid, block=(THREADS, 1, 1))

        return _host

    @cute.jit
    def _host_dependent(
        a: cute.Tensor,
        b0: cute.Tensor,
        b1: cute.Tensor,
        c0: cute.Tensor,
        c1: cute.Tensor,
    ) -> None:
        mma0 = _mk_mma(tcgen05.OperandSource.SMEM)
        mma1 = _mk_mma(tcgen05.OperandSource.TMEM)
        a_layout = sm100_utils.make_smem_layout_a(mma0, _TILER0, a.element_type, AB_STAGES)
        b0_layout = sm100_utils.make_smem_layout_b(mma0, _TILER0, b0.element_type, AB_STAGES)
        b1_layout = sm100_utils.make_smem_layout_b(mma1, _TILER1, b1.element_type, AB_STAGES)
        a2_tmem_layout = sm100_utils.make_smem_layout_a(mma1, _TILER1, b1.element_type, 1)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b0l = cute.select(b0_layout, mode=[0, 1, 2])
        b1l = cute.select(b1_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, _TILER0, mma0)
        b0_atom, b0_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b0, b0l, _TILER0, mma0)
        b1_atom, b1_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b1, b1l, _TILER1, mma1)
        grid = cute.ceil_div((*c0.layout.shape, 1), _TILER0[:2])
        _kern_dependent(
            mma0,
            mma1,
            a_atom,
            a_t,
            b0_atom,
            b0_t,
            b1_atom,
            b1_t,
            c0,
            c1,
            a_layout,
            b0_layout,
            b1_layout,
            a2_tmem_layout,
        ).launch(grid=grid, block=(THREADS, 1, 1))

    def _mark(t: Tensor) -> object:
        return (
            from_dlpack(t.contiguous(), assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16)
        )

    def run(mode: str) -> dict[str, Any]:
        dev = "cuda"
        torch.manual_seed(0)
        a = torch.randn(M, K, dtype=torch.float16, device=dev)
        b0 = torch.randn(N, K, dtype=torch.float16, device=dev)
        b1 = torch.randn(N, K, dtype=torch.float16, device=dev)
        c0 = torch.zeros(M, N, dtype=torch.float16, device=dev)
        c1 = torch.zeros(M, N, dtype=torch.float16, device=dev)

        if mode == "dependent":
            b1d = torch.randn(N, K2, dtype=torch.float16, device=dev)
            ca, cb0, cb1, cc0, cc1 = _mark(a), _mark(b0), _mark(b1d), _mark(c0), _mark(c1)
            ex = cute.compile(_host_dependent, ca, cb0, cb1, cc0, cc1)
            ex(ca, cb0, cb1, cc0, cc1)
            torch.cuda.synchronize()
            ref0 = (a.float() @ b0.float().t()).half()
            ref1 = ref0.float() @ b1d.float().t()
            return {"c0": c0, "ref0": ref0.float(), "c1": c1, "ref1": ref1}

        kern = _kern_independent if mode == "independent" else _kern_looped
        ca, cb0, cb1, cc0, cc1 = _mark(a), _mark(b0), _mark(b1), _mark(c0), _mark(c1)
        ex = cute.compile(_make_host_two(kern), ca, cb0, cb1, cc0, cc1)
        ex(ca, cb0, cb1, cc0, cc1)
        torch.cuda.synchronize()
        ref0 = a.float() @ b0.float().t()
        ref1 = a.float() @ b1.float().t()
        if mode == "looped":
            ref1 = ref1 * NT
        return {"c0": c0, "ref0": ref0, "c1": c1, "ref1": ref1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent", "looped", "dependent"], required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    res: dict[str, Any] = {"mode": args.mode, "nt": NT}
    try:
        if not _HAVE:
            raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
        t = run(args.mode)
        go = True
        for tag in ("0", "1"):
            got = t[f"c{tag}"].float()
            ref = t[f"ref{tag}"].float()
            diff = (got - ref).abs()
            denom = ref.abs().clamp_min(1e-3)
            res[f"max_abs_{tag}"] = diff.max().item()
            res[f"max_rel_{tag}"] = (diff / denom).max().item()
            res[f"finite_{tag}"] = bool(torch.isfinite(got).all().item())
            go = go and res[f"finite_{tag}"] and res[f"max_rel_{tag}"] < 5e-2
        res["GO"] = bool(go)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()
        res["GO"] = False

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"\nGO={res.get('GO')} mode={args.mode}")


if __name__ == "__main__":
    main()
