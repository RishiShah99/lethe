"""Minimal decoupled repro of the inc-B2 launch-SIGSEGV: a proven tcgen05 GEMM wrapped in scf.for.

The inc-B2 bisection (HANDOFF 2026-06-30) isolated the launch-SIGSEGV to ONE tcgen05 GEMM
(TMEM accumulator + async pipelines, allocated OUTSIDE the loop, used INSIDE) wrapped in
``for it in cutlass.range(nt)`` — it crashes even at NT=1, while the byte-identical body run
straight-line works. This file decouples that from every inc-B2 detail (no round-trip, no
glue, no dynamic-L coord): it takes the proven straight-line GEMM (``out=a@b^T`` on the
silicon-verified (128,64,128) config) and runs it three ways, each in a SEPARATE process so a
segfault in one cannot poison the next's CUDA context:

  straight   the proven body verbatim (CONTROL — must pass; guards against transcription bugs)
  loop       mainloop+epilogue wrapped in ``cutlass.range(1)``; the TMEM accumulator handle
             (``tCtAcc = make_tensor(tmem.retrieve_ptr(_acc), layout)``) bound OUTSIDE the loop
             and captured across the scf.for region boundary  → reproduces the suspected fault
  loopfix    same loop, but the handle (+ the epilogue setup that reads it) re-derived INSIDE
             the loop so the TMEM address is region-local, not captured  → the candidate fix

Interpretation: straight pass + loop crash + loopfix pass  ⇒  the in-loop TMEM-handle idiom is
the fix, and inc-B2 ``--bisect {5,6}`` (the same idiom in the real kernel) should clear too.

Run on the box (one process per mode):
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/loop_gemm_repro.py --mode straight
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/loop_gemm_repro.py --mode loop
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/loop_gemm_repro.py --mode loopfix
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

CHUNK = 64
D_K = 128
D_V = 64
THREADS = 128
AB_STAGES = 2

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
    # The shared front matter (thread/smem/tmem/pipelines/tiles/fragments/TMA partition) is
    # IDENTICAL across the three kernels — only the tail (handle binding + mainloop + epilogue,
    # and whether it is wrapped in scf.for) differs. It cannot be factored into a device helper:
    # the if→scf.if AST preprocess only reaches the @cute.kernel body, so `if warp_idx == 0`
    # would raise "dynamic Boolean" in a helper. Hence three explicit, faithful copies.
    # ------------------------------------------------------------------

    @cute.kernel
    def _kern_straight(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
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

        nbytes = cute.size_in_bytes(_io, cute.select(a_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
            _io, cute.select(b_layout, mode=[0, 1, 2])
        )
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
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)

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
                cute.copy(tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                cute.copy(tma_b, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
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

    @cute.kernel
    def _kern_loop(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
    ) -> None:
        # M0 — proven body with ONLY the mainloop+epilogue wrapped in cutlass.range(1). The TMEM
        # accumulator handle + epilogue setup stay OUTSIDE → tCtAcc/tmem_copy/... are SSA values
        # captured across the scf.for region boundary (the suspected garbage-TMEM-address fault).
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

        nbytes = cute.size_in_bytes(_io, cute.select(a_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
            _io, cute.select(b_layout, mode=[0, 1, 2])
        )
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
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )

        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)

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
        for _ in cutlass.range(1):
            if warp_idx == 0:
                acc_empty = acc_prod.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                    ab_e = ab_prod.acquire_and_advance()
                    cute.copy(tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    cute.copy(tma_b, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
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

    @cute.kernel
    def _kern_loopfix(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
    ) -> None:
        # M1 (the candidate fix) — same range(1) loop, but the TMEM accumulator handle and the
        # epilogue setup that reads it are re-derived INSIDE the loop, so the TMEM address is
        # region-local (mirrors inc-B2 --bisect {5,6}). relinquish stays OUTSIDE, once.
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

        nbytes = cute.size_in_bytes(_io, cute.select(a_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
            _io, cute.select(b_layout, mode=[0, 1, 2])
        )
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
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)  # layout carrier (pure, no TMEM ptr)

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )

        tmem.wait_for_alloc()
        tmem.relinquish_alloc_permit()

        nk = cute.size(gA, mode=[2])
        for _ in cutlass.range(1):
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

            if warp_idx == 0:
                acc_empty = acc_prod.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(nk, prefetch_stages=AB_STAGES - 1):
                    ab_e = ab_prod.acquire_and_advance()
                    cute.copy(tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    cute.copy(tma_b, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    ab_f = ab_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        c = (None, None, kb, ab_f.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[c], tCrB[c], tCtAcc)
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
        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.kernel
    def _kern_straight2(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC0: cute.Tensor, mC1: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
    ) -> None:
        # The decisive multi-GEMM test: TWO proven GEMMs back-to-back in ONE kernel, NO loop, each
        # with a FULL per-GEMM TMEM + pipeline lifecycle (allocate→…→free), into two outputs. One
        # GEMM/kernel is GO (straight/loop/loopfix); ≥2 (loop_tile L=4, scf.for AND unrolled) fault.
        # straight2 GO ⇒ the fix for inc-B2 is a full per-GEMM lifecycle; 139 ⇒ ≥2 tcgen05 epilogues
        # per kernel fault regardless (fusion blocked in this DSL).
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        coord = (bidx, bidy, None)

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
        nbytes = cute.size_in_bytes(_io, cute.select(a_layout, mode=[0, 1, 2])) + cute.size_in_bytes(
            _io, cute.select(b_layout, mode=[0, 1, 2])
        )

        for g in cutlass.range_constexpr(2):
            mC = mC0 if cutlass.const_expr(g == 0) else mC1
            tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
            tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
            tmem.allocate(512)
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
                tma_a, 0, cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_b, 0, cute.make_layout(1),
                cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            tmem.wait_for_alloc()
            tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)
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
                    cute.copy(tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    cute.copy(tma_b, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    ab_f = ab_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        cc = (None, None, kb, ab_f.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[cc], tCrB[cc], tCtAcc)
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
            cute.arch.barrier()

    @cute.jit
    def _host_straight2(a: cute.Tensor, b: cute.Tensor, c0: cute.Tensor, c1: cute.Tensor) -> None:
        op = tcgen05.MmaF16BF16Op(
            _io, _acc, _MNK_INST, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, a.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, b.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        a_atom, a_t = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, _MNK_TILER, tiled_mma)
        b_atom, b_t = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b, b1, _MNK_TILER, tiled_mma)
        grid = cute.ceil_div((*c0.layout.shape, 1), _MNK_TILER[:2])
        _kern_straight2(tiled_mma, a_atom, a_t, b_atom, b_t, c0, c1, a_layout, b_layout).launch(
            grid=grid, block=(THREADS, 1, 1)
        )

    def _make_host(kern: object) -> object:
        @cute.jit
        def _host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
            op = tcgen05.MmaF16BF16Op(
                _io, _acc, _MNK_INST, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
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
            kern(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout).launch(
                grid=grid, block=(THREADS, 1, 1)
            )

        return _host

    _KERNS = {"straight": _kern_straight, "loop": _kern_loop, "loopfix": _kern_loopfix}

    def _mark(t: Tensor) -> object:
        return (
            from_dlpack(t.contiguous(), assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, divisibility=16)
        )

    def run(mode: str) -> Tensor:
        dev = "cuda"
        torch.manual_seed(0)
        a = torch.randn(D_K, D_K, dtype=torch.float16, device=dev)
        b = torch.randn(D_V, D_K, dtype=torch.float16, device=dev)
        out = torch.zeros(D_K, D_V, dtype=torch.float16, device=dev)
        if mode == "straight2":
            out1 = torch.zeros(D_K, D_V, dtype=torch.float16, device=dev)
            ca, cb, cc0, cc1 = _mark(a), _mark(b), _mark(out), _mark(out1)
            ex = cute.compile(_host_straight2, ca, cb, cc0, cc1)
            ex(ca, cb, cc0, cc1)
            torch.cuda.synchronize()
            return out  # both GEMMs compute a@b^T; checking c0 is sufficient
        ca, cb, cc = _mark(a), _mark(b), _mark(out)
        ex = cute.compile(_make_host(_KERNS[mode]), ca, cb, cc)
        ex(ca, cb, cc)
        torch.cuda.synchronize()
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["straight", "loop", "loopfix", "straight2"], required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    res: dict[str, Any] = {"mode": args.mode}
    try:
        if not _HAVE:
            raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
        out = run(args.mode)
        torch.manual_seed(0)  # same draw as run(): a, b (out is zeros, consumes no RNG)
        a = torch.randn(D_K, D_K, dtype=torch.float16, device="cuda")
        b = torch.randn(D_V, D_K, dtype=torch.float16, device="cuda")
        ref = a.float() @ b.float().t()  # [128, 64]
        got = out.float()
        diff = (got - ref).abs()
        denom = ref.abs().clamp_min(1e-3)
        res["max_abs"] = diff.max().item()
        res["max_rel"] = (diff / denom).max().item()
        res["finite"] = bool(torch.isfinite(got).all().item())
        res["GO"] = bool(res["finite"] and res["max_rel"] < 5e-2)
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
