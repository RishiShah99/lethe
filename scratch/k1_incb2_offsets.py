"""K#1 inc-B2 v2 — fused reverse-scan kernel on the proven TMEM offset-partition lifecycle.

DESIGN (the central question for the box gate):

  The original ``_incb2_kernel`` used ``cutlass.range_constexpr(nt)`` (static unroll) because
  ``make_tmem_copy`` inside a dynamic ``scf.for`` is illegal (documented crash: launch-SIGSEGV
  when created inside, "live after conversion" ICE when hoisted then used inside).  The
  ``tmem_offset_probe.py`` ``looped`` mode proved a real ``cutlass.range(NT)`` works when the
  accumulator handle is re-derived IN-LOOP (``base_it = tmem.retrieve_ptr(...)`` inside the
  loop body) — the ``loopfix`` idiom.  The epilogue copy atoms are built BEFORE the main loop
  (using the pre-loop ``retrieve_ptr`` call) and reused across iterations; only the per-iteration
  TMEM tensor (``cute.make_tensor(base_it, ...)``) is constructed fresh each iteration.

  mamba2_ssd_v452.py precedent (lines ~2977-2994, ``pre_intra_tmem_store_and_partition_q``):
  ``tiled_r2t_q = tcgen05.make_tmem_copy(...)`` + ``thr_r2t_q.partition_D(tCrQ)`` are built
  ONCE before the chunk loop; inside the loop they are used as ``cute.copy(tiled_r2t_q,
  tRT_rQ, tRT_tQ[q_coord])`` with only the pipeline-stage coord varying per iteration.
  Our epilogue does the same: ``tmem_copy0 / tmem_copy1`` + their ``partition_D`` slices are
  computed once outside the in-kernel loop; inside the loop ``cute.copy(tmem_copy{0,1}, ...)``
  is called with a per-iteration TMEM source re-derived via ``base_it = tmem.retrieve_ptr(_acc)``.

  All partitioning constructs that are bh-fixed (TMA atoms for bdhT, C-side gC tiles, fragment
  shapes) are hoisted pre-loop following the mamba2_ssd pattern: helpers at mamba2_ssd lines
  ~842/854/862-870/878/1006/1018, dynamic loops at ~896/1036, in-loop only slices with dynamic
  coords (e.g. tXgX[(None, ab_e.count)] at ~901-909, tXgX[None, state.count] at ~946).
  For the lid-varying A-operands (m_ag1, m_aga, m_bga) the L-free tma_partition is hoisted and
  sliced per iteration by the dynamic lid coordinate inside the copy call — the silicon-proven
  ab_e.count slicing idiom transposed to the outer (chunk) dimension.

TMEM column budget (total = 128 ≤ 512):
  acc0  (128,64) fp32 G1 accumulator  — cols  0 .. 63   (_NCOLS0 = 64)
  acc1  (128,64) fp32 GA accumulator  — cols 64 .. 127  (_NCOLS0 + _NCOLS1 = 128)
  allocate(128): one alloc, acc0 at base+0, acc1 at base+64, relinquish once, free once.

Two distinct ``tiled_mma`` objects (mma0 for G1, mma1 for GA), two distinct acc mbarriers
(acc0_mbar / acc1_mbar in ``_Smem``).  Epilogue pipeline.sync uses ``barrier_id=2`` to decouple
from the TmemAllocator's ``barrier_for_retrieve`` on id 1.

Constexpr ``nt``: one compiled executable per (n_bh, nt, c, d_v, cw) — real scf.for over a
per-compile-baked nt.  ``d_k`` is fixed at 128 (tile M), ``c`` at 64 (half-K), both enforced
in ``run_k1_incB2_v2``.

SCOPE CONSTRAINT — single-N-tile only (d_v = 64):
  ``gB`` and ``gC`` are tiled at fixed coords with ``_MNK_TILER N=64``.  For ``d_v=128`` (the
  cw crown shapes) this kernel produces only the first 64 value-columns; the remaining 64 are
  never written.  ``run_k1_incB2_v2`` raises ``ValueError`` when ``d_v != 64``.

NEXT INCREMENT — N-tiling for d_v=128:
  A second N-pass per GEMM (``bidy`` coord 0/1, or an outer loop over N tiles) with acc2/acc3
  at TMEM cols 128/192 raises the budget to 256 ≤ 512 and unlocks the cw crown (d_k=d_v=128).
  That is a separate increment after this one silicon-gates.

Desk gate (CPU, fp64): ``_desk_gate()`` checks ``_run_k1_incB2_modelled`` vs the microgate
bundle references at d_v=64 shapes only (the single-N-tile envelope).  The gate exercises the
pure-torch model; the kernel's fp16 landing points — m_bdv (TMEM→GMEM at the G1 epilogue),
dv2 / b_ga[:,:,C:] (SIMT glue), m_t (GA epilogue), b_dhT (carry round-trip) — are only
covered by the box silicon gate.

Box gate: ``run_k1_incB2_v2`` → ``_kern_incb2_v2`` on B200; grade dh/dv2/dh0
scale_rel = diff.max()/exp.abs().max().clamp_min(1e-12) < 5e-3 (fp16 path), 2-run determinism,
write GO json (--out).  When ``--bundle`` is absent, the harness builds a bundle in-process.

OPEN RISKS (box adjudicates):
  (a) Per-iteration acc-pipeline acquire/commit/wait/release inside the real scf.for
      (``cutlass.range(nt)``) is BEYOND what ``tmem_offset_probe.py``'s ``looped`` mode proved
      (the probe committed once after the outer loop, not per-iteration).  mamba2_ssd lines
      939-985 / 1223-1302 / 1944-2002 cycle pipeline state explicitly per chunk inside a
      ``while`` tile loop using ``make_pipeline_state`` + ``advance()``; our version uses the
      higher-level ``make_participants()`` / ``acquire_and_advance()`` wrapper.  Specific risks:
      (i) ``make_participants()`` result lowering as a loop-carried value inside scf.for may
      ICE; (ii) the ab-participant count/index/phase must survive as loop-carried values across
      the outer scf.for + nested K-mainloop boundaries.  Fallback: switch to
      ``make_pipeline_state`` + explicit ``advance()`` matching mamba2_ssd's style.
  (b) Pre-loop ``make_tmem_copy`` atoms (``tmem_copy0``/``tmem_copy1``) used in-loop via
      ``cute.copy(tmem_copy0, tDtC0, ...)`` — this is the mamba2_ssd idiom but a different
      configuration from the ``loop_tile_repro.py`` ICE (there the atom was created inside the
      loop, which is distinct from being created outside and re-used inside).  If it ICEs,
      fallback = epilogue via ``cutlass.range_constexpr`` unroll (keeping the real scf.for +
      new lifecycle for the MMA path).
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

# ─── constants ────────────────────────────────────────────────────────────────
THREADS = 128
AB_STAGES = 2
_NCOLS0 = 64   # acc0 at col offset 0
_NCOLS1 = 64   # acc1 at col offset 64; total = 128
_TMEM_TOTAL = 128

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
    def _kern_incb2_v2(
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

        # ── Pre-loop hoisting (mamba2_ssd style) ────────────────────────────
        # Epilogue copy atoms built ONCE (mamba2_ssd pre_intra_tmem_store_and_partition_q
        # lines ~2977-2994): tiled_r2t / get_slice built outside, used inside.
        # The TMEM tensor is re-derived per iteration via loopfix (base_it in-loop).
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

        # ── REVERSE CHUNK LOOP ──────────────────────────────────────────────
        for it in cutlass.range(nt):
            # loopfix: TMEM handles re-derived inside the loop (scf.for-local).
            base_it = tmem.retrieve_ptr(_acc)
            tCtAcc0 = cute.make_tensor(base_it, tCtAcc0_frag.layout)
            tCtAcc1 = cute.make_tensor(base_it + _NCOLS0, tCtAcc1_frag.layout)

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
                for _kt in cutlass.range(cute.size(gA_g1_all, mode=[2]), prefetch_stages=AB_STAGES - 1):
                    ab_e = ab_prod.acquire_and_advance()
                    # Slice lid into the L-free pre-hoisted partition (ab_e.count is the K stage).
                    cute.copy(
                        tma_ag1,
                        tAgA_g1_all[(None, ab_e.count, lid)],
                        tAsA_g1_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    cute.copy(
                        tma_bdhT,
                        tBgB_g1_pre[(None, ab_e.count)],
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
                for _kt in cutlass.range(cute.size(gA_ga_all, mode=[2]), prefetch_stages=AB_STAGES - 1):
                    ab_e = ab_prod.acquire_and_advance()
                    cute.copy(
                        tma_aga,
                        tAgA_ga_all[(None, ab_e.count, lid)],
                        tAsA_ga_pre[(None, ab_e.index)],
                        tma_bar_ptr=ab_e.barrier,
                    )
                    cute.copy(
                        tma_bga,
                        tBgB_ga_all[(None, ab_e.count, lid)],
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
    def _incb2_v2_host(
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
        _kern_incb2_v2(
            mma0, mma1,
            at_ag1, t_ag1,
            at_aga, t_aga,
            at_bdhT, t_bdhT,
            at_bga, t_bga,
            m_bdv, m_t, m_bdh, m_dh, m_dv2,
            m_decay, m_dvl, m_glast,
            m_bdhT_s, m_bga_s, m_bdv_s, m_t_s,
            a_layout, b_layout,
            nt, c, d_v,
        ).launch(grid=(1, 1, n_bh), block=(THREADS, 1, 1))

    _v2_cache: dict[tuple[int, ...], object] = {}


# ─── marking helpers (mirrors gdn2_bwd_dhu._mark_b / _mark_simt) ─────────────

def _mark_b(t: Tensor) -> object:
    """[Z,R,S] → cute [R,S,Z] L-major TMA layout."""
    v = t.contiguous().permute(1, 2, 0)
    return from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _mark_simt(t: Tensor) -> object:
    return from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


# ─── public launcher ──────────────────────────────────────────────────────────

def run_k1_incB2_v2(
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
    """Fused K#1 reverse scan on the proven TMEM offset-partition lifecycle.

    Packs via ``_incb2_pack_scalar`` (scalar) or ``_incb2_pack_cw`` (cw), launches
    ``_kern_incb2_v2`` once for all n_bh groups. Returns ``(dh, dv2, dh0)`` head-major.
    Single-N-tile only: d_v must be 64, d_k must be 128, c must be 64.
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")

    from lethe.kernels.cute.gdn2_bwd_dhu import _incb2_pack_scalar
    from lethe.kernels.cute.gdn2_bwd_dhu_cw import _incb2_pack_cw

    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    if d_v != 64:
        raise ValueError(
            f"run_k1_incB2_v2 is single-N-tile (N=64); d_v={d_v} unsupported. "
            "N-tiling for d_v=128 is the next increment."
        )
    if d_k != 128:
        raise ValueError(f"kernel hard-codes M=128 rows; d_k={d_k} unsupported.")
    if c != 64:
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
    b_dh = dht.reshape(n_bh, d_k, d_v).clone().float()  # clone: no-op chain on contiguous fp32 would alias + mutate caller dht
    b_dhT = b_dh.transpose(-1, -2).contiguous().to(f16)
    bdv_raw = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)
    t_scr = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)
    dh_out = torch.zeros(ll, d_k, d_v, dtype=f16, device=dev)
    dv2_out = torch.zeros(ll, c, d_v, dtype=f16, device=dev)

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
    )
    key = (n_bh, nt, c, d_v, d_k, int(cw))
    ex = _v2_cache.get(key)
    if ex is None:
        ex = cute.compile(_incb2_v2_host, *args)
        _v2_cache[key] = ex
    ex(*args)
    torch.cuda.synchronize()
    return (
        dh_out.reshape(b, hv, nt, d_k, d_v),
        dv2_out.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


# ─── desk gate (CPU, fp64) ────────────────────────────────────────────────────

def _desk_gate() -> bool:
    """CPU fp64 model check: _run_k1_incB2_modelled vs bundle references (scalar + cw).

    Mirrors scratch/k1_incB2_orchestration_check.py; target ≤ 1e-12 relative error.
    Both the existing scalar and cw modelled specs are re-used verbatim — the v2 kernel
    transcribes the same dataflow (only the lifecycle changes), so these specs remain the
    ground truth. cw shapes are d_v=64 only (single-N-tile envelope). The kernel's fp16
    landing points — m_bdv (G1 TMEM→GMEM epilogue), dv2/b_ga[:,:,C:] (SIMT glue), m_t
    (GA epilogue), b_dhT (carry round-trip) — are only covered by the silicon gate.
    """
    import lethe.kernels.cute.gdn2_bwd_dhu as k1mod
    import lethe.kernels.cute.gdn2_bwd_dhu_cw as k1cw
    from lethe.kernels.references.gdn2_chunkwise import build_microgate_bundles
    from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    TOL = 1e-12

    def _l2(x: Tensor) -> Tensor:
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    def _rel(got: Tensor, ref: Tensor) -> float:
        return (
            (got.double() - ref.double()).abs().max()
            / ref.double().abs().max().clamp_min(1e-12)
        ).item()

    scalar_shapes = [(1, 1, 1, 64, 128, 64), (2, 2, 4, 64, 128, 64), (1, 1, 8, 64, 128, 64)]
    # d_v=64 only: single-N-tile envelope; d_v=128 belongs to the N-tiling increment.
    cw_shapes = [(1, 1, 1, 64, 128, 64), (2, 2, 2, 64, 128, 64), (1, 1, 3, 64, 128, 64)]

    ok = True

    for shape in scalar_shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 13 + 1)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.1 + 0.01)
        beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles(q, k_t, v, g, beta, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        mdl = k1mod._run_k1_incB2_modelled(
            i1["q"], i1["k"], i1["w"], i1["g2"], i1["g_last"],
            i1["do"], i1["dv_local"], i1["dht"],
        )
        worst = 0.0
        for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
            r = _rel(got, ref)
            worst = max(worst, r)
            if r > TOL:
                ok = False
        print(f"  scalar shape={shape}: worst_rel={worst:.2e}  (tol {TOL:.0e})")

    for shape in cw_shapes:
        b, h, nt, c, d_k, d_v = shape
        t = nt * c
        gen = torch.Generator().manual_seed(nt * 17 + 5)
        dt = torch.float64
        q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
        v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
        bun = build_microgate_bundles_cw(q, k_t, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
        i1, e1 = bun["k1"].inputs, bun["k1"].expected
        mdl = k1cw._run_k1_incB2_modelled(
            i1["q"], i1["k"], i1["wy"], i1["g2"], i1["g_last"],
            i1["do"], i1["dv_local"], i1["dht"],
        )
        worst = 0.0
        for got, ref in zip(mdl, (e1["dh"], e1["dv2"], e1["dh0"]), strict=True):
            r = _rel(got, ref)
            worst = max(worst, r)
            if r > TOL:
                ok = False
        print(f"  cw     shape={shape}: worst_rel={worst:.2e}  (tol {TOL:.0e})")

    print(f"\nDesk gate: GO={ok}  (tol {TOL:.0e})")
    return ok


# ─── box harness ──────────────────────────────────────────────────────────────

def _scale_rel(got: Tensor, exp: Tensor) -> float:
    """scale_rel = diff.max() / exp.abs().max().clamp_min(1e-12) — Phase-2/3 GO ledger metric."""
    diff = (got.float().cpu() - exp.float().cpu()).abs()
    denom = exp.float().cpu().abs().max().clamp(min=1e-12)
    return (diff.max() / denom).item()


def _compare(name: str, got: Tensor, exp: Tensor, atol: float, rtol: float) -> dict[str, Any]:
    got_f = got.float().cpu()
    exp_f = exp.float().cpu()
    diff = (got_f - exp_f).abs()
    sr = _scale_rel(got, exp)
    max_abs = diff.max().item()
    max_rel = (diff / exp_f.abs().clamp_min(1e-6)).max().item()
    finite = bool(torch.isfinite(got_f).all().item())
    # GO criterion: scale_rel < 5e-3 (Phase-2/3 ledger; 3.29e-3/3.31e-3 prior marks).
    passed = finite and sr < 5e-3
    return {
        "name": name,
        "shape": list(got_f.shape),
        "scale_rel": sr,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "finite": finite,
        "passed": passed,
    }


def _build_bundle(mode: str) -> dict[str, Any]:
    """Build an in-process K#1 bundle at the default box shape (b=2,h=2,nt=4,c=64,d_k=128,d_v=64)."""
    from lethe.kernels.references.gdn2_chunkwise import build_microgate_bundles
    from lethe.kernels.references.gdn2_chunkwise_cw import build_microgate_bundles_cw

    b, h, nt, c, d_k, d_v = 2, 2, 4, 64, 128, 64
    t = nt * c
    gen = torch.Generator().manual_seed(42)
    dt = torch.float64

    def _l2(x: Tensor) -> Tensor:
        return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    k_t = _l2(torch.randn(b, t, h, d_k, generator=gen, dtype=dt))
    v = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)
    do = torch.randn(b, t, h, d_v, generator=gen, dtype=dt)

    if mode == "cw":
        g = -(torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.1 + 0.01)
        bg = torch.rand(b, t, h, d_k, generator=gen, dtype=dt) * 0.8 + 0.1
        wg = torch.rand(b, t, h, d_v, generator=gen, dtype=dt) * 0.8 + 0.1
        bun = build_microgate_bundles_cw(q, k_t, v, g, bg, wg, do, chunk_len=c, scale=d_k**-0.5)
    else:
        g = -(torch.rand(b, t, h, generator=gen, dtype=dt) * 0.1 + 0.01)
        beta = torch.rand(b, t, h, generator=gen, dtype=dt) * 0.8 + 0.1
        bun = build_microgate_bundles(q, k_t, v, g, beta, do, chunk_len=c, scale=d_k**-0.5)

    k1 = bun["k1"]
    return {
        "inputs": {n: tv.float() for n, tv in k1.inputs.items()},
        "expected": {n: tv.float() for n, tv in k1.expected.items()},
        "meta": {**k1.meta, "B": b, "H": h, "NT": nt, "C": c, "d_k": d_k, "d_v": d_v},
    }


def _run_box(bundle_path: str, mode: str, atol: float, rtol: float) -> dict[str, Any]:
    bp = Path(bundle_path) if bundle_path else None
    # NB: Path("") normalizes to "." (exists!) — never Path() an empty string here.
    if bp is None or not bp.exists():
        print(f"{'--bundle absent' if not bundle_path else bp} — building in-process "
              f"(b=2,h=2,nt=4,c=64,d_k=128,d_v=64, mode={mode})")
        payload = _build_bundle(mode)
        if bundle_path:
            torch.save(payload, bp)
            print(f"saved {bp}")
    else:
        payload = torch.load(bundle_path, weights_only=False)

    inp = {k: v.cuda() for k, v in payload["inputs"].items()}
    exp = payload["expected"]
    cw = mode == "cw"
    key_w = "wy" if cw else "w"

    dh, dv2, dh0 = run_k1_incB2_v2(
        inp["q"], inp["k"], inp[key_w], inp["g2"], inp["g_last"],
        inp["do"], inp["dv_local"], inp["dht"], cw=cw,
    )
    checks = [
        _compare("dh", dh, exp["dh"], atol, rtol),
        _compare("dv2", dv2, exp["dv2"], atol, rtol),
        _compare("dh0", dh0, exp["dh0"], atol, rtol),
    ]

    # Second run for determinism (bit-exact).
    dh2, dv2_2, dh0_2 = run_k1_incB2_v2(
        inp["q"], inp["k"], inp[key_w], inp["g2"], inp["g_last"],
        inp["do"], inp["dv_local"], inp["dht"], cw=cw,
    )
    det = (
        torch.equal(dh.cpu(), dh2.cpu())
        and torch.equal(dv2.cpu(), dv2_2.cpu())
        and torch.equal(dh0.cpu(), dh0_2.cpu())
    )

    return {
        "device": torch.cuda.get_device_name(0),
        "bundle": bundle_path,
        "mode": mode,
        "meta": payload.get("meta", {}),
        "checks": checks,
        "deterministic": det,
        "GO": all(c_["passed"] for c_ in checks) and det,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Parse before cutlass import-time argparse fires (tmem_offset_probe.py pattern).
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--desk", action="store_true")
    ap.add_argument("--mode", choices=["scalar", "cw"], default="scalar")
    ap.add_argument("--bundle", type=str, default="")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-h", "--help", action="help")
    args = ap.parse_args()

    res: dict[str, Any] = {"mode": args.mode}
    try:
        if args.desk:
            res["GO"] = _desk_gate()
        else:
            if not _HAVE:
                raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
            res.update(_run_box(args.bundle, args.mode, args.atol, args.rtol))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["trace"] = traceback.format_exc()
        res["GO"] = False

    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    print(f"\nGO={res.get('GO')}  mode={args.mode}")


if __name__ == "__main__":
    main()
