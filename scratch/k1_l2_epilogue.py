"""Level-2 de-glue: fold elementwise glue into tcgen05 kernel epilogues (channel-wise K#1).

Level 1 (run_k1_incB_batched in gdn2_bwd_dhu_cw.py) batches the (b,hv) groups but per
reverse chunk ``it`` still fires: 1 _bmm_tc G1 launch · 1 torch add · 1 copy · 2 torch.cat
· 1 _bmm_tc GA launch · 1 scale-add — ~6 python ops + 2 kernel launches + torch overhead.

Level 2 collapses that to EXACTLY 2 kernel launches per step, 0 torch compute ops.

Two new batched kernels, each = the silicon-verified _gemm_kernel_b epilogue (proven
(128,64,128) config; _MNK_TILER bakes M=128, N=64, K=128, so c=64 and d_k=128 are required)
with SIMT glue appended after pipeline.sync, before tmem.free:

  _kern_g1_epi  — G1 GEMM (a_g1[z]@b_dhT[z]^T → raw [128,64]) then SIMT:
                  dh snapshot loop: for e in range(128*d_v): m_dh[z,r,col] = m_bdh_pre[z,r,col]
                  glue loop: for e in range(c*d_v):
                    val = raw[z,r,col] + dv_local[z,r,col]
                    write dv2[z,r,col] + b_ga[z,col,C+r] (b_dv^T → GA's 2nd half)
                  single barrier() + fence_proxy("async.global")

  _kern_ga_epi  — GA GEMM (a_ga[z]@b_ga[z]^T → t [128,64]) then SIMT:
                  for e in range(d_k*d_v): val = glast[z,r]*b_dh[z,r,col] + t[z,r,col]
                  write b_dh[z,r,col] (fp32 carry) + b_dhT[z,col,r] (fp16, next G1 operand)
                  barrier() + fence_proxy

relinquish_alloc_permit is placed AFTER acc_empty.commit() in BOTH kernels, matching the
byte-fidelity target _gemm_kernel_b (gdn2_bwd_dhu.py:407-409), NOT before the mainloop.

Host layout: ONCE before the loop, restage all flat-L operand arrays to step-major
([nt, n_bh, ...] order) so buf[it] is a true contiguous view with zero copy per step.
Output buffers dh_out/dv2_out are [nt, n_bh, ...] step-major so kernel output slices
dh_out[it]/dv2_out[it] are also zero-copy contiguous views.
bdv_raw/t_scr hoisted outside the loop; each launch fully overwrites them.

Off-box: imports cleanly (guarded try/except → _HAVE). cute.compile only on sm_100.
IMPORTANT: no `from __future__ import annotations` (breaks @cute.struct/@cute.kernel).
"""

import argparse
import json
import traceback
from pathlib import Path

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
except ImportError:
    _HAVE = False

# ------------------------------------------------------------------
# Shared constants — must match gdn2_bwd_dhu.py exactly (proven config).
# _MNK_TILER bakes M=128, N=64, K=128; c=CHUNK=64 and d_k=D_K=128 are required.
# ------------------------------------------------------------------
D_K = 128
D_V = 64
CHUNK = 64
THREADS = 128
AB_STAGES = 2


def _cur_stream() -> "cuda_driver.CUstream":
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _mark_b(t: Tensor) -> object:
    """[Z,R,S] → cute [R,S,Z] L-major (TMA wants L as outermost stride)."""
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
    # Single barrier + fence_proxy after both loops.
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
            tma_ag1, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_bdhT, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
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
                cute.copy(tma_ag1, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)],
                          tma_bar_ptr=ab_e.barrier)
                cute.copy(tma_bdhT, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)],
                          tma_bar_ptr=ab_e.barrier)
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

        cute.arch.barrier()
        cute.arch.fence_proxy("async.global")

    # ------------------------------------------------------------------
    # _kern_ga_epi: GA GEMM + b_dh carry update.
    #
    # a_ga[Z,128,128] @ b_ga[Z,64,128]^T → t[Z,128,64]
    # SIMT carry over d_k*d_v elements, single barrier + fence_proxy.
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
            tma_aga, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_bga, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
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
                cute.copy(tma_aga, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)],
                          tma_bar_ptr=ab_e.barrier)
                cute.copy(tma_bga, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)],
                          tma_bar_ptr=ab_e.barrier)
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
        cute.arch.barrier()
        cute.arch.fence_proxy("async.global")

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
            _io, _acc, _MNK_INST, tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, a_g1.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, b_dhT.element_type, AB_STAGES)
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
            tiled_mma, a_atom, a_t, b_atom, b_t,
            bdv_raw, b_dh_pre, b_ga_s, dv2, dh, dvl, bdv_raw_s,
            a_layout, b_layout, c, d_v,
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
            _io, _acc, _MNK_INST, tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, a_ga.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, b_ga.element_type, AB_STAGES)
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
            tiled_mma, a_atom, a_t, b_atom, b_t,
            t_scr, b_dh, b_dhT_s, glast, t_scr_s,
            a_layout, b_layout, d_k, d_v,
        ).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)

    # compile-cache keyed (n_bh, c, d_v) — shapes are static across steps.
    _g1_cache: dict[tuple[int, int, int], object] = {}
    _ga_cache: dict[tuple[int, int, int], object] = {}


# ------------------------------------------------------------------
# Pure-torch model of the Level-2 per-step dataflow.
#
# Mirrors the kernel's fp16 cast round-trips:
#   - GEMM operands (a_g1, a_ga, b_ga first half) cast f16 before matmul → back to f32.
#   - b_dv^T stored f16 into b_ga's second half, read back f32 for GA.
#   - b_dhT stored f16, used f16 for next G1's matmul.
#   - glast and dv_local stay f32 (SIMT reads them as _acc=Float32).
# fp32 GEMM stand-ins; in fp64 input it tracks the cw bundle to roundoff.
# ------------------------------------------------------------------


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

    Includes fp16 cast round-trips for all kernel I/O (operand casts, b_ga/b_dhT stores).
    fp32 GEMM stand-ins (no tcgen05). In fp64 input tracks the cw bundle within the fp16
    quantisation floor (~3e-4 relative).
    """
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    dev = q.device
    f16, f32 = torch.float16, torch.float32

    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw as _pack

    buf = _pack(q, k, wy, g2, g_last, do, dv_local)
    a_g1_L = buf["a_g1"]
    a_ga_L = buf["a_ga"]
    b_ga_L = buf["b_ga"]
    dvl_L  = buf["dv_local"]
    glast_L = buf["glast"]

    b_dh = dht.reshape(n_bh, d_k, d_v).clone().to(f32)
    # fp16 round-trip: b_dhT is the TMA B operand for G1.
    b_dhT = b_dh.to(f16).transpose(-1, -2).contiguous()

    dh_out  = torch.zeros(nt, n_bh, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(nt, n_bh, c,   d_v, dtype=f16, device=dev)

    for it in reversed(range(nt)):
        # Operand casts: f16 then back to f32 mirrors kernel fp16 GEMM precision.
        a_g1_step = a_g1_L.reshape(n_bh, nt, d_k, d_k)[:, it].to(f16).to(f32)
        a_ga_step = a_ga_L.reshape(n_bh, nt, d_k, 2 * c)[:, it].to(f16).to(f32)
        b_ga_step = b_ga_L.reshape(n_bh, nt, d_v, 2 * c)[:, it].to(f16).clone()
        dvl_step  = dvl_L.reshape(n_bh, nt, c, d_v)[:, it].to(f32)
        gl_step   = glast_L.reshape(n_bh, nt, d_k)[:, it].to(f32)

        dh_out[it] = b_dh.to(f16)

        raw = a_g1_step @ b_dhT.to(f32).transpose(-1, -2)
        b_dv = raw[:, :c] + dvl_step
        dv2_out[it] = b_dv.to(f16)
        # b_dv^T round-trip into GA's 2nd half; fp16 matches in-kernel SIMT store dtype.
        b_ga_step[:, :, c:] = b_dv.to(f16).transpose(-1, -2)

        t = a_ga_step @ b_ga_step.to(f32).transpose(-1, -2)
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

    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw as _pack

    buf = _pack(q, k, wy, g2, g_last, do, dv_local)

    # Restage ONCE to step-major [nt, n_bh, ...] so buf_sm[it] is a contiguous zero-copy view.
    def _sm(x: Tensor, *trailing: int) -> Tensor:
        return x.reshape(n_bh, nt, *trailing).transpose(0, 1).contiguous()

    ag1_sm  = _sm(buf["a_g1"].to(f16),    D_K, D_K)    # [nt, n_bh, 128, 128]
    aga_sm  = _sm(buf["a_ga"].to(f16),    D_K, 2 * c)  # [nt, n_bh, 128, 2C]
    bga_sm  = _sm(buf["b_ga"].to(f16),    d_v, 2 * c)  # [nt, n_bh, 64, 2C]
    dvl_sm  = _sm(buf["dv_local"].to(f32), c,   d_v)   # [nt, n_bh, C, 64]
    gl_sm   = _sm(buf["glast"].to(f32),   d_k)          # [nt, n_bh, 128]

    b_dh  = dht.reshape(n_bh, d_k, d_v).contiguous().to(f32)
    b_dhT = b_dh.to(f16).transpose(-1, -2).contiguous()  # [n_bh, d_v, d_k]

    # Step-major outputs: dh_out[it] and dv2_out[it] are contiguous zero-copy views.
    dh_out  = torch.zeros(nt, n_bh, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(nt, n_bh, c,   d_v, dtype=f16, device=dev)

    # Hoisted scratch: each launch fully overwrites these before SIMT reads them.
    bdv_raw = torch.zeros(n_bh, D_K, D_V, dtype=f16, device=dev)
    t_scr   = torch.zeros(n_bh, D_K, D_V, dtype=f16, device=dev)

    key = (n_bh, c, D_V)
    stream = _cur_stream()

    for it in reversed(range(nt)):
        ag1_step = ag1_sm[it]   # [n_bh, 128, 128] — contiguous
        aga_step = aga_sm[it]   # [n_bh, 128, 2C]
        bga_step = bga_sm[it]   # [n_bh, 64, 2C]  — written in-kernel (b_dv^T 2nd half)
        dvl_step = dvl_sm[it]   # [n_bh, C, 64]
        gl_step  = gl_sm[it]    # [n_bh, 128]

        g1_args = (
            _mark_b(ag1_step),
            _mark_b(b_dhT),
            _mark_b(bdv_raw),
            _mark_simt(b_dh),
            _mark_simt(bga_step),
            _mark_simt(dv2_out[it]),
            _mark_simt(dh_out[it]),
            _mark_simt(dvl_step),
            _mark_simt(bdv_raw),
            c, D_V, stream,
        )
        ex_g1 = _g1_cache.get(key)
        if ex_g1 is None:
            ex_g1 = cute.compile(_host_g1_epi, *g1_args)
            _g1_cache[key] = ex_g1
        ex_g1(*g1_args)

        ga_args = (
            _mark_b(aga_step),
            _mark_b(bga_step),
            _mark_b(t_scr),
            _mark_simt(b_dh),
            _mark_simt(b_dhT),
            _mark_simt(gl_step),
            _mark_simt(t_scr),
            d_k, D_V, stream,
        )
        ex_ga = _ga_cache.get(key)
        if ex_ga is None:
            ex_ga = cute.compile(_host_ga_epi, *ga_args)
            _ga_cache[key] = ex_ga
        ex_ga(*ga_args)

    from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import maybe_sync

    maybe_sync()
    return (
        dh_out.permute(1, 0, 2, 3).reshape(b, hv, nt, d_k, d_v),
        dv2_out.permute(1, 0, 2, 3).reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


# ------------------------------------------------------------------
# Desk gate: _modelled_l2 vs fp64 build_microgate_bundles_cw expected.
# ------------------------------------------------------------------


def _rel(got: Tensor, ref: Tensor) -> float:
    return (
        (got.float() - ref.float()).abs().max()
        / ref.float().abs().max().clamp_min(1e-12)
    ).item()


def _l2(x: Tensor) -> Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def desk_check() -> bool:
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    # All d_v=64 (the kernel tile width). nt ∈ {1,2,3} — nt=1 masked blocker 1 before fix.
    shapes = [
        (1, 1, 1, 64, 128, 64),
        (2, 2, 2, 64, 128, 64),
        (1, 1, 3, 64, 128, 64),
    ]
    worst = 0.0
    for shape in shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 17 + c)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        got = _modelled_l2(
            i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"],
            i1["do"], i1["dv_local"], i1["dht"],
        )
        for got_t, name in zip(got, ("dh", "dv2", "dh0"), strict=True):
            rel = _rel(got_t, e1[name])
            worst = max(worst, rel)
        print(f"  shape={shape}  worst_scale_rel_so_far={worst:.2e}")

    tol = 5e-3
    ok = worst < tol
    print(f"\n_modelled_l2 vs fp64 ref: worst_scale_rel={worst:.2e}  tol={tol:.0e}  GO={ok}")
    return ok


def _build_cw_bundle(b: int, h: int, nt: int, c: int, d_k: int, d_v: int) -> dict:
    from flash_mamba_rl.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    t = nt * c
    gen = torch.Generator().manual_seed(42)
    dt = torch.float64
    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
    bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
    wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    bun = build_microgate_bundles_cw(q, k, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    k1 = bun["k1"]
    return {
        "inputs": {n: t_.float() for n, t_ in k1.inputs.items()},
        "expected": {n: t_.float() for n, t_ in k1.expected.items()},
        "meta": {**k1.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }


# ------------------------------------------------------------------
# Box CLI: --mode cw grades run_k1_incB_l2 vs cw bundle.
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desk-check", action="store_true", help="off-box pure-torch model gate")
    ap.add_argument("--mode", type=str, default="cw", choices=["cw"])
    ap.add_argument("--bundle", type=str, default="k1_bundle_cw_nt4.pt",
                    help="cw K#1 bundle .pt (inputs: q,k,wy,g2,g_last,do,dv_local,dht); "
                         "built in-process at (b=2,h=2,nt=4,c=64,d_k=128,d_v=64) if absent")
    ap.add_argument("--atol", type=float, default=5e-3)
    ap.add_argument("--rtol", type=float, default=5e-3)
    ap.add_argument("--out", type=str, default="results/k1_l2_epilogue.json")
    args = ap.parse_args()

    if args.desk_check:
        ok = desk_check()
        raise SystemExit(0 if ok else 1)

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(f"{bundle_path} not found — building in-process (b=2,h=2,nt=4,c=64,d_k=128,d_v=64)")
        payload = _build_cw_bundle(b=2, h=2, nt=4, c=64, d_k=128, d_v=64)
        torch.save(payload, bundle_path)
        print(f"saved {bundle_path}")
    else:
        payload = torch.load(bundle_path, weights_only=False)

    out: dict = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": args.mode,
        "bundle": str(bundle_path),
    }
    try:
        inp = {kk: vv.cuda() for kk, vv in payload["inputs"].items()}
        exp = payload["expected"]
        out["meta"] = payload.get("meta", {})

        dh1, dv2_1, dh0_1 = run_k1_incB_l2(
            inp["q"], inp["k"], inp["wy"], inp["g2"], inp["g_last"],
            inp["do"], inp["dv_local"], inp["dht"],
        )
        dh2, dv2_2, dh0_2 = run_k1_incB_l2(
            inp["q"], inp["k"], inp["wy"], inp["g2"], inp["g_last"],
            inp["do"], inp["dv_local"], inp["dht"],
        )

        def _cmp(name: str, got: Tensor, ref: Tensor) -> dict:
            g = got.float().cpu()
            r = ref.float().cpu() if isinstance(ref, Tensor) else torch.tensor(ref)
            diff = (g - r).abs()
            denom = r.abs().clamp_min(1e-6)
            ok_ = bool((diff <= args.atol + args.rtol * r.abs()).all().item())
            return {
                "name": name,
                "max_abs": diff.max().item(),
                "max_rel": (diff / denom).max().item(),
                "finite": bool(torch.isfinite(g).all().item()),
                "passed": ok_ and bool(torch.isfinite(g).all().item()),
            }

        checks = [
            _cmp("dh",  dh1,  exp["dh"]),
            _cmp("dv2", dv2_1, exp["dv2"]),
            _cmp("dh0", dh0_1, exp["dh0"]),
        ]
        det_ok = (
            torch.equal(dh1.cpu(), dh2.cpu())
            and torch.equal(dv2_1.cpu(), dv2_2.cpu())
            and torch.equal(dh0_1.cpu(), dh0_2.cpu())
        )
        out["checks"] = checks
        out["deterministic"] = det_ok
        out["GO"] = all(c["passed"] for c in checks) and det_ok
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()
        out["GO"] = False

    dest = Path(args.out)
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nGO={out.get('GO')}  ->  {dest}")


if __name__ == "__main__":
    main()
