"""Level-3 de-glue K#1 — fused reverse-scan kernel: offset-partition lifecycle + UNROLLED chunks.

Wired from ``scratch/k1_incb2_v3_unroll.py`` after the burst-3 silicon gates
(results/k1_incb2_v3_{scalar_nt1,scalar_nt4,cw_nt4,cw_nt8}.json: GO + deterministic,
worst scale_rel ~6.5e-4 vs the fp64 bundles). ONE launch for the whole reverse chunk
scan — the entire ``it`` loop lives in-kernel with ``b_dh`` TMEM-resident per chunk,
vs Level-2's 2 launches/chunk.

The burst-2 v2 attempt kept a real ``scf.for`` over chunks and hoisted the epilogue
``make_tmem_copy`` atoms pre-loop — box-adjudicated ICE ("failed to legalize unresolved
materialization ... remained live after conversion"): the mamba2_ssd pre-loop-atom idiom
covers its STORE atoms, not our LOAD ``tmem_copy`` in this config. v3 is the documented
fallback: ``cutlass.range_constexpr(nt)`` UNROLLS the chunk loop so every chunk is the
silicon-proven straight-line body (atoms consumed outside any scf region); the K-tile
mainloops stay real dynamic loops. Everything else is v2 verbatim: ONE tmem.allocate(128),
acc0/acc1 at column offsets 0/64 (tmem_offset_probe.py GO), one relinquish + one free at
the end, hoisted L-free TMA partitions sliced per chunk, two mma objects + two acc
mbarrier sets, epilogue sync on barrier_id=2, store->fence->barrier ordering on both GMEM
round-trips (b_ga second half; b_dh/b_dhT carry).

Cost model: one executable per (n_bh, nt, c, d_v, cw) — nt is baked; IR size grows with
nt (silicon compile walls 9/12/15 s at nt=1/4/8; nt=16/32 is the burst-4 probe). Per-chunk
TMEM readback is structural (the carry feeds the next chunk's GEMM operands through
SMEM/GMEM), so unrolling — not a dynamic loop — is the only legal shape for it on this
wheel.

TMEM column budget (total = 128 <= 512):
  acc0  (128,64) fp32 G1 accumulator  — cols  0 .. 63   (_NCOLS0 = 64)
  acc1  (128,64) fp32 GA accumulator  — cols 64 .. 127  (_NCOLS0 + _NCOLS1 = 128)

Tile contract: single-N-tile only — C=64, d_k=128, d_v=64 exactly (:func:`l3_dims_ok`);
``gB``/``gC`` are tiled at fixed coords with ``_MNK_TILER`` N=64, so for d_v=128 only the
first 64 value-columns would ever be written. d_v=128 stays on the Level-2/lever-B paths
until the N-tiling increment (acc2/acc3 at TMEM cols 128/192, budget 256 <= 512).

Launcher deltas vs the gated scratch file (host boundary only, kernel body identical):
launches ride torch's current stream (:func:`_cur_stream`) -> CUDA-graph-capturable, and
the trailing ``torch.cuda.synchronize()`` is now ``maybe_sync()`` (capture-aware) — the
Level-2 promote discipline. Executor contract: the compile tuple keeps the full signature;
the call tuple passes ONLY the 16 marked cute.Tensors + the CUstream (Constexprs are baked
and DROPPED from the runtime arg list — re-passing them shifts pointer slots -> SIGSEGV).

Off-box: imports cleanly (guarded try/except -> _HAVE). The kernel spec is
``gdn2_bwd_dhu_cw._run_k1_incB2_modelled`` (fp64-pinned by the orchestration checks).
Box gate harness: ``scratch/k1_incb2_v3_unroll.py``.
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes @cute.struct fields.

import torch
from torch import Tensor

from lethe.kernels.cute.gdn2_bwd_dhu import maybe_sync

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
_NCOLS0 = 64  # acc0 at col offset 0
_NCOLS1 = 64  # acc1 at col offset 64; total = 128
_TMEM_TOTAL = 128


def is_available() -> bool:
    return _HAVE


def l3_dims_ok(c: int, d_k: int, d_v: int) -> bool:
    """True iff the fused kernel's baked tile fits: C=64, d_k=128, d_v=64 exactly."""
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
    _MNK_INST = (128, 64, 16)
    _MNK_TILER = (128, 64, 128)
    _AB_MBAR = AB_STAGES * 2

    @cute.struct
    class _Smem:
        ab_mbar: cute.struct.MemRange[cutlass.Int64, _AB_MBAR]
        acc0_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        acc1_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_buf: cutlass.Int32

    # ─── kernel ───────────────────────────────────────────────────────────────

    @cute.kernel
    def _kern_incb2_v3(
        mma0: cute.TiledMma,
        mma1: cute.TiledMma,
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
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, _, bh = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        # barrier_id=1 reserved for TmemAllocator; epilogue pipeline.sync uses barrier_id=2.
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)

        acc_shape0 = mma0.partition_shape_C(_MNK_TILER[:2])
        tCtAcc0_frag = mma0.make_fragment_C(acc_shape0)
        acc_shape1 = mma1.partition_shape_C(_MNK_TILER[:2])
        tCtAcc1_frag = mma1.make_fragment_C(acc_shape1)

        # ONE alloc: acc0 at col 0, acc1 at col 64.
        tmem.allocate(_TMEM_TOTAL)

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

        tmem.wait_for_alloc()

        # ── Pre-loop hoisting ───────────────────────────────────────────────
        # Epilogue copy atoms + partitions built ONCE; with the chunk loop UNROLLED
        # they are consumed in straight-line context (no scf region -> the v2 ICE
        # class is structurally gone).
        base_pre = tmem.retrieve_ptr(_acc)
        tCtAcc0_pre = cute.make_tensor(base_pre, tCtAcc0_frag.layout)
        tCtAcc1_pre = cute.make_tensor(base_pre + _NCOLS0, tCtAcc1_frag.layout)

        epi_tiler = ((cute.size(tCtAcc0_pre, mode=[0, 0]), cute.size(tCtAcc0_pre, mode=[0, 1])),)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)

        tCtAcc0_epi_pre = cute.zipped_divide(tCtAcc0_pre, epi_tiler)
        tCtAcc1_epi_pre = cute.zipped_divide(tCtAcc1_pre, epi_tiler)
        tmem_copy0 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc0_epi_pre[None, 0])
        tmem_copy1 = tcgen05.make_tmem_copy(tmem_atom, tCtAcc1_epi_pre[None, 0])
        thr0 = tmem_copy0.get_slice(tidx)
        thr1 = tmem_copy1.get_slice(tidx)

        # bh-fixed C-side chains (bh is constant per CTA — loop-invariant).
        gC0_pre = cute.local_tile(m_bdv, _MNK_TILER, (0, 0, None, bh), proj=(1, 1, None))
        gC1_pre = cute.local_tile(m_t, _MNK_TILER, (0, 0, None, bh), proj=(1, 1, None))
        thr_mma0_pre = mma0.get_slice(0)
        thr_mma1_pre = mma1.get_slice(0)
        tCgC0_pre = thr_mma0_pre.partition_C(gC0_pre)
        tCgC1_pre = thr_mma1_pre.partition_C(gC1_pre)
        gC0_epi_pre = cute.zipped_divide(tCgC0_pre, epi_tiler)
        gC1_epi_pre = cute.zipped_divide(tCgC1_pre, epi_tiler)
        tDgC0_pre = thr0.partition_D(gC0_epi_pre)
        tDgC1_pre = thr1.partition_D(gC1_epi_pre)

        # bh-fixed B-operand for G1 (bdhT never changes index across chunks).
        gB_g1_pre = cute.local_tile(m_bdhT, _MNK_TILER, (0, 0, None, bh), proj=(None, 1, 1))
        thr_mma0_b = mma0.get_slice(0)
        tBsB_g1_pre, tBgB_g1_pre = cute.nvgpu.cpasync.tma_partition(
            tma_bdhT,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma0_b.partition_B(gB_g1_pre), 0, 3),
        )

        # L-free A/B partitions for lid-varying operands (sliced by lid inside copy calls).
        # Mirrors the ab_e.count slicing idiom used in the K-mainloop (mamba2_ssd ~901-909).
        gA_g1_all = cute.local_tile(m_ag1, _MNK_TILER, (0, 0, None, None), proj=(1, None, 1))
        gA_ga_all = cute.local_tile(m_aga, _MNK_TILER, (0, 0, None, None), proj=(1, None, 1))
        gB_ga_all = cute.local_tile(m_bga, _MNK_TILER, (0, 0, None, None), proj=(None, 1, 1))
        thr_mma0_a = mma0.get_slice(0)
        thr_mma1_a = mma1.get_slice(0)
        thr_mma1_b = mma1.get_slice(0)
        tAsA_g1_pre, tAgA_g1_all = cute.nvgpu.cpasync.tma_partition(
            tma_ag1,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma0_a.partition_A(gA_g1_all), 0, 3),
        )
        tAsA_ga_pre, tAgA_ga_all = cute.nvgpu.cpasync.tma_partition(
            tma_aga,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma1_a.partition_A(gA_ga_all), 0, 3),
        )
        tBsB_ga_pre, tBgB_ga_all = cute.nvgpu.cpasync.tma_partition(
            tma_bga,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma1_b.partition_B(gB_ga_all), 0, 3),
        )

        # Fragment shapes are fixed (layout determined by mma + tiler, not by lid/bh).
        tCrA_g1 = mma0.make_fragment_A(sA)
        tCrB_g1 = mma0.make_fragment_B(sB)
        tCrA_ga = mma1.make_fragment_A(sA)
        tCrB_ga = mma1.make_fragment_B(sB)

        # rmem work buffers: shape from the pre-loop partition_D (loop-invariant).
        tCrAcc0 = cute.make_rmem_tensor(tDgC0_pre[None, None, 0].shape, _acc)
        tCrC0 = cute.make_rmem_tensor(tDgC0_pre[None, None, 0].shape, _io)
        tCrAcc1 = cute.make_rmem_tensor(tDgC1_pre[None, None, 0].shape, _acc)
        tCrC1 = cute.make_rmem_tensor(tDgC1_pre[None, None, 0].shape, _io)

        # ── REVERSE CHUNK LOOP — range_constexpr UNROLL (the burst-2 fallback) ──
        # The dynamic scf.for made the pre-loop tmem_copy atoms illegal ("failed to
        # legalize unresolved materialization ... remained live after conversion",
        # box-adjudicated). Unrolling makes every chunk the proven straight-line body;
        # the only dynamic loops left are the K-tile mainloops + SIMT element loops.
        # nt is Constexpr + cache-keyed, so one executable per nt is the contract.
        for it in cutlass.range_constexpr(nt):
            tCtAcc0 = cute.make_tensor(base_pre, tCtAcc0_frag.layout)
            tCtAcc1 = cute.make_tensor(base_pre + _NCOLS0, tCtAcc1_frag.layout)

            i_t = nt - 1 - it
            lid = bh * nt + i_t

            # dh[lid] = current b_dh (read before this chunk mutates it).
            for e in cutlass.range(tidx, 128 * d_v, THREADS):
                r, col = e // d_v, e % d_v
                m_dh[lid, r, col] = m_bdh[bh, r, col].to(_io)
            cute.arch.barrier()

            # ── G1: m_bdv[·,bh] = m_ag1[·,lid] @ m_bdhT[·,bh]^T ──────────
            tCtAcc0_epi = cute.zipped_divide(tCtAcc0, epi_tiler)
            tDtC0 = thr0.partition_S(tCtAcc0_epi)

            if warp_idx == 0:
                acc0_empty = acc0_prod.acquire_and_advance()
                mma0.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(
                    cute.size(gA_g1_all, mode=[2]), prefetch_stages=AB_STAGES - 1
                ):
                    ab_e = ab_prod.acquire_and_advance()
                    # gmem K coordinate is literal 0: the K-rest extent is statically 1
                    # (tiler K=128 over 128-deep operands) and ab_e.count grows
                    # MONOTONICALLY across the 2*nt unrolled mainloops (mamba2_ssd
                    # resets its producer state per work tile for exactly this reason)
                    # — count as the coordinate is OOB from chunk-0's GA onward.
                    # ab_e.index (SMEM stage) and ab_e.barrier stay.
                    cute.copy(
                        tma_ag1,
                        tAgA_g1_all[(None, 0, lid)],
                        tAsA_g1_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    cute.copy(
                        tma_bdhT,
                        tBgB_g1_pre[(None, 0)],
                        tBsB_g1_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    ab_f = ab_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA_g1, mode=[2])):
                        cc = (None, None, kb, ab_f.index)
                        cute.gemm(mma0, tCtAcc0, tCrA_g1[cc], tCrB_g1[cc], tCtAcc0)
                        mma0.set(tcgen05.Field.ACCUMULATE, True)
                    ab_f.release()
                acc0_empty.commit()

            acc0_f = acc0_cons.wait_and_advance()
            for i in cutlass.range(cute.size(tDtC0, mode=[2])):
                cute.copy(tmem_copy0, tDtC0[None, None, i], tCrAcc0)
                tCrC0.store(tCrAcc0.load().to(_io))
                cute.autovec_copy(tCrC0, tDgC0_pre[None, None, i])
            acc0_f.release()
            pipeline.sync(barrier_id=2)

            # SIMT glue: b_dv = bdv_raw[:C]·decay + dv_local; write dv2 + b_ga 2nd half.
            # PTX order: stores → fence_proxy (each writing thread) → barrier → TMA re-read.
            for e in cutlass.range(tidx, c * d_v, THREADS):
                r, col = e // d_v, e % d_v
                val = m_bdv_s[bh, r, col].to(_acc) * m_decay[lid, r] + m_dvl[lid, r, col]
                m_dv2[lid, r, col] = val.to(_io)
                m_bga_s[lid, col, c + r] = val.to(_io)
            cute.arch.fence_proxy("async.global")
            cute.arch.barrier()

            # ── GA: m_t[·,bh] = m_aga[·,lid] @ m_bga[·,lid]^T ─────────────
            tCtAcc1_epi = cute.zipped_divide(tCtAcc1, epi_tiler)
            tDtC1 = thr1.partition_S(tCtAcc1_epi)

            if warp_idx == 0:
                acc1_empty = acc1_prod.acquire_and_advance()
                mma1.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(
                    cute.size(gA_ga_all, mode=[2]), prefetch_stages=AB_STAGES - 1
                ):
                    ab_e = ab_prod.acquire_and_advance()
                    cute.copy(
                        tma_aga,
                        tAgA_ga_all[(None, 0, lid)],
                        tAsA_ga_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    cute.copy(
                        tma_bga,
                        tBgB_ga_all[(None, 0, lid)],
                        tBsB_ga_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    ab_f = ab_cons.wait_and_advance()
                    for kb in cutlass.range_constexpr(cute.size(tCrA_ga, mode=[2])):
                        cc = (None, None, kb, ab_f.index)
                        cute.gemm(mma1, tCtAcc1, tCrA_ga[cc], tCrB_ga[cc], tCtAcc1)
                        mma1.set(tcgen05.Field.ACCUMULATE, True)
                    ab_f.release()
                acc1_empty.commit()

            acc1_f = acc1_cons.wait_and_advance()
            for i in cutlass.range(cute.size(tDtC1, mode=[2])):
                cute.copy(tmem_copy1, tDtC1[None, None, i], tCrAcc1)
                tCrC1.store(tCrAcc1.load().to(_io))
                cute.autovec_copy(tCrC1, tDgC1_pre[None, None, i])
            acc1_f.release()
            pipeline.sync(barrier_id=2)

            # SIMT carry: b_dh = exp2(g_last)·b_dh + t; write b_dh + b_dhT.
            # PTX order: stores → fence_proxy → barrier → next-chunk G1 TMA re-read.
            for e in cutlass.range(tidx, 128 * d_v, THREADS):
                r, col = e // d_v, e % d_v
                val = m_glast[lid, r] * m_bdh[bh, r, col] + m_t_s[bh, r, col].to(_acc)
                m_bdh[bh, r, col] = val
                m_bdhT_s[bh, col, r] = val.to(_io)
            cute.arch.fence_proxy("async.global")
            cute.arch.barrier()

        # ONE relinquish after all MMA issue; ONE free.
        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=2)
        tmem.free(base_pre)

    # ─── host jit ─────────────────────────────────────────────────────────────

    @cute.jit
    def _incb2_v3_host(
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
        mma0 = cute.make_tiled_mma(op)
        mma1 = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(mma0, _MNK_TILER, m_ag1.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma0, _MNK_TILER, m_bga.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        at_ag1, t_ag1 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_ag1, a1, _MNK_TILER, mma0)
        at_aga, t_aga = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_aga, a1, _MNK_TILER, mma1)
        at_bdhT, t_bdhT = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bdhT, b1, _MNK_TILER, mma0)
        at_bga, t_bga = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bga, b1, _MNK_TILER, mma1)
        _kern_incb2_v3(
            mma0,
            mma1,
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
        ).launch(grid=(1, 1, n_bh), block=(THREADS, 1, 1), stream=stream)

    _v3_cache: dict[tuple[int, ...], object] = {}


# ─── public launcher ──────────────────────────────────────────────────────────


def run_k1_incB2_v3(
    q: Tensor,
    k: Tensor,
    w_or_wy: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
    dht: Tensor,
    *,
    cw: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fused K#1 reverse scan on the proven TMEM offset-partition lifecycle — ONE launch.

    Packs via ``_incb2_pack_scalar`` (scalar) or ``_incb2_pack_cw`` (cw), launches
    ``_kern_incb2_v3`` once for all n_bh groups on torch's current stream. Returns
    ``(dh, dv2, dh0)`` head-major. Single-N-tile only: d_v=64, d_k=128, c=64
    (:func:`l3_dims_ok`).
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")

    from lethe.kernels.cute.gdn2_bwd_dhu import _incb2_pack_scalar
    from lethe.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    if d_v != D_V:
        raise ValueError(
            f"run_k1_incB2_v3 is single-N-tile (N=64); d_v={d_v} unsupported. "
            "N-tiling for d_v=128 is the next increment."
        )
    if d_k != D_K:
        raise ValueError(f"kernel hard-codes M=128 rows; d_k={d_k} unsupported.")
    if c != CHUNK:
        raise ValueError(f"kernel hard-codes K=2c=128; c={c} unsupported (need c=64).")

    n_bh = b * hv
    dev = q.device
    f16, f32 = torch.float16, torch.float32
    ll = n_bh * nt

    if cw:
        buf = _incb2_pack_cw(q, k, w_or_wy, g2, g_last, do, dv_local)
        glast = buf["glast"]  # [L, d_k]
        decay = torch.ones(ll, c, dtype=f32, device=dev)
    else:
        buf = _incb2_pack_scalar(q, k, w_or_wy, g2, g_last, do, dv_local)
        glast = buf["glast"][:, None].expand(ll, d_k).contiguous()
        decay = buf["decay"].to(f32)

    a_g1_16 = buf["a_g1"].to(f16).contiguous()
    a_ga_16 = buf["a_ga"].to(f16).contiguous()
    b_ga_16 = buf["b_ga"].to(f16).contiguous()
    b_dh = (
        dht.reshape(n_bh, d_k, d_v).clone().float()
    )  # clone: no-op chain on contiguous fp32 would alias + mutate caller dht
    b_dhT = b_dh.transpose(-1, -2).contiguous().to(f16)
    bdv_raw = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)
    t_scr = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)
    dh_out = torch.zeros(ll, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(ll, c, d_v, dtype=f16, device=dev)

    stream = _cur_stream()
    args = (
        _mark_b(a_g1_16),
        _mark_b(a_ga_16),
        _mark_b(b_dhT),
        _mark_b(b_ga_16),
        _mark_b(bdv_raw),
        _mark_b(t_scr),
        _mark_simt(b_dh),
        _mark_simt(dh_out),
        _mark_simt(dv2_out),
        _mark_simt(decay),
        _mark_simt(buf["dv_local"].to(f32)),
        _mark_simt(glast.to(f32)),
        _mark_simt(b_dhT),
        _mark_simt(b_ga_16),
        _mark_simt(bdv_raw),
        _mark_simt(t_scr),
        n_bh,
        nt,
        c,
        d_v,
        stream,
    )
    key = (n_bh, nt, c, d_v, d_k, int(cw))
    ex = _v3_cache.get(key)
    if ex is None:
        ex = cute.compile(_incb2_v3_host, *args)
        _v3_cache[key] = ex
    # Constexpr args (n_bh, nt, c, d_v) are baked at compile and DROPPED from the
    # runtime signature — the call tuple is the 16 marked tensors + the CUstream.
    ex(*args[:16], stream)
    maybe_sync()
    return (
        dh_out.reshape(b, hv, nt, d_k, d_v),
        dv2_out.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )
