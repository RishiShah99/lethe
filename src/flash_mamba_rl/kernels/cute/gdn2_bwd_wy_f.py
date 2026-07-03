"""Fused K#2 channel-wise — ONE grid-z-batched tcgen05 kernel for the whole WY-VJP.

The campaign's big rock: today's ``run_k2_batched`` costs ~17 ms @B2/L2048/H8, almost all
of it glue — the 1.07 GB fp32 ``masked_decay_rel`` materialization plus six einsum reads
over it. This kernel deletes that: the six GEMM sites land in TMEM and the decay pushes
are computed SIMT in-kernel with ON-THE-FLY ``exp2(g2_i - g2_s)`` on the strict-lower
triangle ONLY (the loop range IS the mask — arguments are structurally <= 0, so the
factored-overflow NaN class (c707201) cannot occur; never rewrite as 2^Gi * 2^-Gs).

K#2 is chunk-local (no inter-chunk carry), so the grid is one CTA per chunk over
Z = B*H*NT — the Level-2 batching idiom, NOT L3's unroll: no compile wall, nt not baked,
one executable per shape class, and Z=512 CTAs @L2048 ~ 3.5 waves on 148 SMs is genuinely
throughput-parallel (unlike K#1's n_bh-bound 16 CTAs).

Seven sequential mainloop+epilogue pairs (the silicon-proven L3-at-nt=3 shape), all on the
proven a@b^T (128,64,128) tiler, single K-tile each (operands K-pad to 128):

  G1  dp     = T^T @ du            A=a_tt   B=b_du     -> fp32 land s_dp32
  G2a dq_wy[:, :64]  = T^T @ dwy   A=a_tt   B=b_dwy N0 -> fp32 land s_dq32 N0
  G2b dq_wy[:, 64:]                A=a_tt   B=b_dwy N1 -> fp32 land s_dq32 N1
  G3  dT    += du @ pwv^T          A=a_du   B=b_pwv    }  ONE shared accumulator,
  G4  dT    += dwy @ q_kbg^T       A=a_dwy  B=b_qkbg   }  ACCUMULATE carried G3->G4
                                   -> fp16 land into s_dt (A-shaped, K-pad zeros persist)
  G5  X      = dT @ T^T            A=s_dt   B=b_t      -> fp16 land s_xn, SIMT transpose
                                                          into s_x (B-shaped X^T)
  G6  dM_raw = T^T @ X             A=a_tt   B=s_x      -> fp32 land s_dm32

The two chained landings (dT feeding G5, X^T feeding G6) use the L3-proven round trip:
SIMT/autovec stores -> fence_proxy("async.global") -> barrier -> TMA re-read. The G6
strict-lower mask + negate folds into the SIMT tail's reads (``-s_dm32[i,s]`` for s<i,
dead entries never read). The tail computes the two decay einsums (two passes, serial
per-thread fp32 accumulation — deterministic), dg2 via the rowsum/colsum identities
(E never built), the finals, and the reverse-cumsum over C (one key-channel per thread).

TMEM column budget (fp32 accs, M-padded to 128; make_tmem_copy trips on M=64):
  dp 0..63 | dqwy_n0 64..127 | dqwy_n1 128..191 | dT 192..255 | X 256..319 | dM 320..383
  total 384 -> ONE tmem.allocate(512) (power-of-2), static offsets, ONE relinquish + free
  at the end. ZERO headroom rule: no TMEM-staged operands; d_v=64 single-N-tile only
  (:func:`k2f_dims_ok` — d_v=128 stays on the batched path until the N-tiling increment).

Discipline carried from the banked silicon lessons, x7 mainloops:
  - gmem K coordinate literal 0 (ab_e.count grows monotonically across the 7 mainloops);
  - executor call tuple = the 32 marked cute.Tensors + CUstream ONLY (Constexprs baked+
    dropped; re-passing shifts pointer slots -> SIGSEGV);
  - all copy atoms hoisted once, consumed in straight-line context (no scf region);
  - fresh output/scratch tensors every call (no caller aliasing).

Kernel spec = ``gdn2_bwd_wy_cw._run_k2_fused_modelled`` (fp64-pinned vs k2_wy_vjp_cw_ref
by tests/test_gdn2_k2_fused.py). Box gate harness: ``scratch/k2f_microgate.py``.
Fallback ladder if silicon misbehaves: (1) ``FMR_K2F_PRED_TAIL=1`` swaps the einsum
passes' triangular loops (data-dependent inner bounds — the one reviewed no-precedent
shape) for the wheel-proven mamba2_ssd SegSum idiom (constexpr unroll + runtime -inf
exponent mask pre-exp2); (2) the 2-kernel split (A: G1/G2/G3+G4; B: G5->G6 + tail),
each grid-z batched — kernel B is then isomorphic to one L3 chunk body.
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes @cute.struct fields.

import torch
from torch import Tensor

from flash_mamba_rl.kernels.cute.gdn2_bwd_dhu import maybe_sync
from flash_mamba_rl.kernels.cute.gdn2_bwd_wy_cw import _k2f_pack, k2f_dims_ok

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
LN2_F = 0.6931471805599453
RCP_LN2_F = 1.4426950408889634
_OFF_DP = 0
_OFF_Q0 = 64
_OFF_Q1 = 128
_OFF_DT = 192
_OFF_X = 256
_OFF_DM = 320
_TMEM_ALLOC = 512  # 384 used; power-of-2 rounding per the mamba2_ssd lifecycle


def is_available() -> bool:
    return _HAVE


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
        dp_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        qwy_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        dt_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        x_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        dm_mbar: cute.struct.MemRange[cutlass.Int64, 2]
        tmem_buf: cutlass.Int32

    # ─── kernel ───────────────────────────────────────────────────────────────

    @cute.kernel
    def _kern_k2f(
        mma_dp: cute.TiledMma,
        mma_qwy: cute.TiledMma,
        mma_dt: cute.TiledMma,
        mma_x: cute.TiledMma,
        mma_dm: cute.TiledMma,
        tma_att1: cute.CopyAtom,
        m_att1: cute.Tensor,
        tma_att2: cute.CopyAtom,
        m_att2: cute.Tensor,
        tma_att6: cute.CopyAtom,
        m_att6: cute.Tensor,
        tma_adu: cute.CopyAtom,
        m_adu: cute.Tensor,
        tma_adwy: cute.CopyAtom,
        m_adwy: cute.Tensor,
        tma_sdt: cute.CopyAtom,
        m_sdt: cute.Tensor,
        tma_bdu: cute.CopyAtom,
        m_bdu: cute.Tensor,
        tma_bdwy: cute.CopyAtom,
        m_bdwy: cute.Tensor,
        tma_bpwv: cute.CopyAtom,
        m_bpwv: cute.Tensor,
        tma_bqkbg: cute.CopyAtom,
        m_bqkbg: cute.Tensor,
        tma_bt: cute.CopyAtom,
        m_bt: cute.Tensor,
        tma_bsx: cute.CopyAtom,
        m_bsx: cute.Tensor,
        m_dp_c: cute.Tensor,
        m_dq_c: cute.Tensor,
        m_dt_c: cute.Tensor,
        m_xn_c: cute.Tensor,
        m_dm_c: cute.Tensor,
        m_dp_s: cute.Tensor,
        m_dq_s: cute.Tensor,
        m_dm_s: cute.Tensor,
        m_xn_s: cute.Tensor,
        m_sx_s: cute.Tensor,
        m_k: cute.Tensor,
        m_bg: cute.Tensor,
        m_v: cute.Tensor,
        m_w: cute.Tensor,
        m_g2: cute.Tensor,
        m_dbkm: cute.Tensor,
        m_dkm: cute.Tensor,
        m_dk: cute.Tensor,
        m_dv: cute.Tensor,
        m_db: cute.Tensor,
        m_dw: cute.Tensor,
        m_dg: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        c: cutlass.Constexpr,
        d_k: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        pred_tail: cutlass.Constexpr,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        _, _, z = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(_Smem)
        sA = smem.allocate_tensor(_io, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
        sB = smem.allocate_tensor(_io, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

        # barrier_id=1 reserved for TmemAllocator; epilogue pipeline.sync uses barrier_id=2.
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
        tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
        tmem.allocate(_TMEM_ALLOC)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_att1)
            cpasync.prefetch_descriptor(tma_att2)
            cpasync.prefetch_descriptor(tma_att6)
            cpasync.prefetch_descriptor(tma_adu)
            cpasync.prefetch_descriptor(tma_adwy)
            cpasync.prefetch_descriptor(tma_sdt)
            cpasync.prefetch_descriptor(tma_bdu)
            cpasync.prefetch_descriptor(tma_bdwy)
            cpasync.prefetch_descriptor(tma_bpwv)
            cpasync.prefetch_descriptor(tma_bqkbg)
            cpasync.prefetch_descriptor(tma_bt)
            cpasync.prefetch_descriptor(tma_bsx)

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

        def _acc_pipe(ptr):
            return pipeline.PipelineUmmaAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
                barrier_storage=ptr,
            ).make_participants()

        dp_prod, dp_cons = _acc_pipe(storage.dp_mbar.data_ptr())
        qwy_prod, qwy_cons = _acc_pipe(storage.qwy_mbar.data_ptr())
        dt_prod, dt_cons = _acc_pipe(storage.dt_mbar.data_ptr())
        x_prod, x_cons = _acc_pipe(storage.x_mbar.data_ptr())
        dm_prod, dm_cons = _acc_pipe(storage.dm_mbar.data_ptr())

        tmem.wait_for_alloc()
        base = tmem.retrieve_ptr(_acc)

        # ── Pre-body hoisting: accumulators, copy atoms, partitions (straight-line) ──
        acc_shape = mma_dp.partition_shape_C(_MNK_TILER[:2])
        acc_frag = mma_dp.make_fragment_C(acc_shape)
        t_dp = cute.make_tensor(base + _OFF_DP, acc_frag.layout)
        t_q0 = cute.make_tensor(base + _OFF_Q0, acc_frag.layout)
        t_q1 = cute.make_tensor(base + _OFF_Q1, acc_frag.layout)
        t_dt = cute.make_tensor(base + _OFF_DT, acc_frag.layout)
        t_x = cute.make_tensor(base + _OFF_X, acc_frag.layout)
        t_dm = cute.make_tensor(base + _OFF_DM, acc_frag.layout)

        epi_tiler = ((cute.size(t_dp, mode=[0, 0]), cute.size(t_dp, mode=[0, 1])),)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)

        t_dp_epi = cute.zipped_divide(t_dp, epi_tiler)
        t_q0_epi = cute.zipped_divide(t_q0, epi_tiler)
        t_q1_epi = cute.zipped_divide(t_q1, epi_tiler)
        t_dt_epi = cute.zipped_divide(t_dt, epi_tiler)
        t_x_epi = cute.zipped_divide(t_x, epi_tiler)
        t_dm_epi = cute.zipped_divide(t_dm, epi_tiler)
        cp_dp = tcgen05.make_tmem_copy(tmem_atom, t_dp_epi[None, 0])
        cp_q0 = tcgen05.make_tmem_copy(tmem_atom, t_q0_epi[None, 0])
        cp_q1 = tcgen05.make_tmem_copy(tmem_atom, t_q1_epi[None, 0])
        cp_dt = tcgen05.make_tmem_copy(tmem_atom, t_dt_epi[None, 0])
        cp_x = tcgen05.make_tmem_copy(tmem_atom, t_x_epi[None, 0])
        cp_dm = tcgen05.make_tmem_copy(tmem_atom, t_dm_epi[None, 0])
        thr_dp = cp_dp.get_slice(tidx)
        thr_q0 = cp_q0.get_slice(tidx)
        thr_q1 = cp_q1.get_slice(tidx)
        thr_dt = cp_dt.get_slice(tidx)
        thr_x = cp_x.get_slice(tidx)
        thr_dm = cp_dm.get_slice(tidx)
        tS_dp = thr_dp.partition_S(t_dp_epi)
        tS_q0 = thr_q0.partition_S(t_q0_epi)
        tS_q1 = thr_q1.partition_S(t_q1_epi)
        tS_dt = thr_dt.partition_S(t_dt_epi)
        tS_x = thr_x.partition_S(t_x_epi)
        tS_dm = thr_dm.partition_S(t_dm_epi)

        # C-side landing chains (z-fixed per CTA). dq lands in two N-tiles of s_dq32;
        # dT lands in the N=64 tile 0 of the A-shaped s_dt (K-pad columns stay host zeros).
        def _land(m_c, mma, thr, ncoord):
            g = cute.local_tile(m_c, _MNK_TILER, (0, ncoord, None, z), proj=(1, 1, None))
            ge = cute.zipped_divide(mma.get_slice(0).partition_C(g), epi_tiler)
            return thr.partition_D(ge)

        tD_dp = _land(m_dp_c, mma_dp, thr_dp, 0)
        tD_q0 = _land(m_dq_c, mma_qwy, thr_q0, 0)
        tD_q1 = _land(m_dq_c, mma_qwy, thr_q1, 1)
        tD_dt = _land(m_dt_c, mma_dt, thr_dt, 0)
        tD_xn = _land(m_xn_c, mma_x, thr_x, 0)
        tD_dm = _land(m_dm_c, mma_dm, thr_dm, 0)

        # A/B TMA partitions per mainloop (all z-fixed; K-rest extent is statically 1).
        def _part_a(tma, m_t, mma):
            g = cute.local_tile(m_t, _MNK_TILER, (0, 0, None, z), proj=(1, None, 1))
            return cute.nvgpu.cpasync.tma_partition(
                tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(mma.get_slice(0).partition_A(g), 0, 3),
            )

        def _part_b(tma, m_t, mma, ncoord):
            g = cute.local_tile(m_t, _MNK_TILER, (0, ncoord, None, z), proj=(None, 1, 1))
            return cute.nvgpu.cpasync.tma_partition(
                tma,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(mma.get_slice(0).partition_B(g), 0, 3),
            )

        tAs_g1, tAg_g1 = _part_a(tma_att1, m_att1, mma_dp)
        tAs_g2, tAg_g2 = _part_a(tma_att2, m_att2, mma_qwy)
        tAs_g3, tAg_g3 = _part_a(tma_adu, m_adu, mma_dt)
        tAs_g4, tAg_g4 = _part_a(tma_adwy, m_adwy, mma_dt)
        tAs_g5, tAg_g5 = _part_a(tma_sdt, m_sdt, mma_x)
        tAs_g6, tAg_g6 = _part_a(tma_att6, m_att6, mma_dm)
        tBs_g1, tBg_g1 = _part_b(tma_bdu, m_bdu, mma_dp, 0)
        tBs_g2a, tBg_g2a = _part_b(tma_bdwy, m_bdwy, mma_qwy, 0)
        tBs_g2b, tBg_g2b = _part_b(tma_bdwy, m_bdwy, mma_qwy, 1)
        tBs_g3, tBg_g3 = _part_b(tma_bpwv, m_bpwv, mma_dt, 0)
        tBs_g4, tBg_g4 = _part_b(tma_bqkbg, m_bqkbg, mma_dt, 0)
        tBs_g5, tBg_g5 = _part_b(tma_bt, m_bt, mma_x, 0)
        tBs_g6, tBg_g6 = _part_b(tma_bsx, m_bsx, mma_dm, 0)

        tCrA_dp = mma_dp.make_fragment_A(sA)
        tCrB_dp = mma_dp.make_fragment_B(sB)
        tCrA_qwy = mma_qwy.make_fragment_A(sA)
        tCrB_qwy = mma_qwy.make_fragment_B(sB)
        tCrA_dt = mma_dt.make_fragment_A(sA)
        tCrB_dt = mma_dt.make_fragment_B(sB)
        tCrA_x = mma_x.make_fragment_A(sA)
        tCrB_x = mma_x.make_fragment_B(sB)
        tCrA_dm = mma_dm.make_fragment_A(sA)
        tCrB_dm = mma_dm.make_fragment_B(sB)

        # rmem epilogue work buffers (all accs share the (128,64) fragment shape).
        tCrAcc = cute.make_rmem_tensor(tD_dp[None, None, 0].shape, _acc)
        tCrC16 = cute.make_rmem_tensor(tD_dp[None, None, 0].shape, _io)

        # ── 7 sequential mainloops + epilogues (straight-line; L3-at-nt=3 shape) ──
        # gmem K coordinate is literal 0 in every copy: the K-rest extent is statically 1
        # (tiler K=128 over 128-deep operands) and ab_e.count grows MONOTONICALLY across
        # the 7 mainloops sharing this pipeline — count as coordinate is OOB from G2 on.

        # G1: dp = a_tt @ b_du^T
        if warp_idx == 0:
            dp_empty = dp_prod.acquire_and_advance()
            mma_dp.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g1, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_att1,
                    tAg_g1[(None, 0)],
                    tAs_g1[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bdu, tBg_g1[(None, 0)], tBs_g1[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_dp, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_dp, t_dp, tCrA_dp[cc], tCrB_dp[cc], t_dp)
                    mma_dp.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            dp_empty.commit()

        dp_f = dp_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_dp, mode=[2])):
            cute.copy(cp_dp, tS_dp[None, None, i], tCrAcc)
            cute.autovec_copy(tCrAcc, tD_dp[None, None, i])
        dp_f.release()
        pipeline.sync(barrier_id=2)

        # G2a/G2b: dq_wy N-tiles 0/1 = a_tt @ b_dwy^T
        if warp_idx == 0:
            q0_empty = qwy_prod.acquire_and_advance()
            mma_qwy.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g2, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_att2,
                    tAg_g2[(None, 0)],
                    tAs_g2[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bdwy,
                    tBg_g2a[(None, 0)],
                    tBs_g2a[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_qwy, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_qwy, t_q0, tCrA_qwy[cc], tCrB_qwy[cc], t_q0)
                    mma_qwy.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            q0_empty.commit()

        q0_f = qwy_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_q0, mode=[2])):
            cute.copy(cp_q0, tS_q0[None, None, i], tCrAcc)
            cute.autovec_copy(tCrAcc, tD_q0[None, None, i])
        q0_f.release()
        pipeline.sync(barrier_id=2)

        if warp_idx == 0:
            q1_empty = qwy_prod.acquire_and_advance()
            mma_qwy.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g2, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_att2,
                    tAg_g2[(None, 0)],
                    tAs_g2[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bdwy,
                    tBg_g2b[(None, 0)],
                    tBs_g2b[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_qwy, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_qwy, t_q1, tCrA_qwy[cc], tCrB_qwy[cc], t_q1)
                    mma_qwy.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            q1_empty.commit()

        q1_f = qwy_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_q1, mode=[2])):
            cute.copy(cp_q1, tS_q1[None, None, i], tCrAcc)
            cute.autovec_copy(tCrAcc, tD_q1[None, None, i])
        q1_f.release()
        pipeline.sync(barrier_id=2)

        # G3 + G4: dT shared accumulator — ONE acquire, ACCUMULATE carried across, ONE commit.
        if warp_idx == 0:
            dt_empty = dt_prod.acquire_and_advance()
            mma_dt.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g3, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_adu, tAg_g3[(None, 0)], tAs_g3[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
                )
                cute.copy(
                    tma_bpwv,
                    tBg_g3[(None, 0)],
                    tBs_g3[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_dt, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_dt, t_dt, tCrA_dt[cc], tCrB_dt[cc], t_dt)
                    mma_dt.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            for _kt in cutlass.range(cute.size(tAg_g4, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_adwy,
                    tAg_g4[(None, 0)],
                    tAs_g4[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bqkbg,
                    tBg_g4[(None, 0)],
                    tBs_g4[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_dt, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_dt, t_dt, tCrA_dt[cc], tCrB_dt[cc], t_dt)
                ab_f.release()
            dt_empty.commit()

        # dT lands fp16 into s_dt (G5's TMA A operand): stores -> fence -> barrier -> TMA.
        dt_f = dt_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_dt, mode=[2])):
            cute.copy(cp_dt, tS_dt[None, None, i], tCrAcc)
            tCrC16.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC16, tD_dt[None, None, i])
        dt_f.release()
        pipeline.sync(barrier_id=2)
        cute.arch.fence_proxy("async.global")
        cute.arch.barrier()

        # G5: X = s_dt @ b_t^T
        if warp_idx == 0:
            x_empty = x_prod.acquire_and_advance()
            mma_x.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g5, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_sdt, tAg_g5[(None, 0)], tAs_g5[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
                )
                cute.copy(
                    tma_bt, tBg_g5[(None, 0)], tBs_g5[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_x, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_x, t_x, tCrA_x[cc], tCrB_x[cc], t_x)
                    mma_x.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            x_empty.commit()

        # X lands fp16 into s_xn, then SIMT-transposes into s_x (G6's TMA B operand: X^T).
        x_f = x_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_x, mode=[2])):
            cute.copy(cp_x, tS_x[None, None, i], tCrAcc)
            tCrC16.store(tCrAcc.load().to(_io))
            cute.autovec_copy(tCrC16, tD_xn[None, None, i])
        x_f.release()
        pipeline.sync(barrier_id=2)

        for e in cutlass.range(tidx, c * c, THREADS):
            r_, s_ = e // c, e % c
            m_sx_s[z, s_, r_] = m_xn_s[z, r_, s_]
        cute.arch.fence_proxy("async.global")
        cute.arch.barrier()

        # G6: dM_raw = a_tt @ s_x^T
        if warp_idx == 0:
            dm_empty = dm_prod.acquire_and_advance()
            mma_dm.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(cute.size(tAg_g6, mode=[1]), prefetch_stages=AB_STAGES - 1):
                ab_e = ab_prod.acquire_and_advance()
                cute.copy(
                    tma_att6,
                    tAg_g6[(None, 0)],
                    tAs_g6[(None, ab_e.index)],
                    tma_bar_ptr=ab_e.barrier,
                )
                cute.copy(
                    tma_bsx, tBg_g6[(None, 0)], tBs_g6[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier
                )
                ab_f = ab_cons.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA_dm, mode=[2])):
                    cc = (None, None, kb, ab_f.index)
                    cute.gemm(mma_dm, t_dm, tCrA_dm[cc], tCrB_dm[cc], t_dm)
                    mma_dm.set(tcgen05.Field.ACCUMULATE, True)
                ab_f.release()
            dm_empty.commit()

        dm_f = dm_cons.wait_and_advance()
        for i in cutlass.range(cute.size(tS_dm, mode=[2])):
            cute.copy(cp_dm, tS_dm[None, None, i], tCrAcc)
            cute.autovec_copy(tCrAcc, tD_dm[None, None, i])
        dm_f.release()
        pipeline.sync(barrier_id=2)

        # ── SIMT tail (all GEMM scratches landed; gmem/rmem only from here) ──
        # d_m[i,s] = -s_dm32[i,s] on the strict-lower triangle, 0 elsewhere: the loop
        # ranges ARE the mask, so exp2 arguments g2_i - g2_s (i>s) are structurally <= 0
        # (bounded, the masked-unfactored discipline) and dead entries are never read.

        # dv/dw from dp (rows >= c of the landed dp are exact zeros via a_tt's M-pad).
        for e in cutlass.range(tidx, c * d_v, THREADS):
            i_, j_ = e // d_v, e % d_v
            dp_v = m_dp_s[z, i_, j_]
            m_dv[z, i_, j_] = dp_v * m_w[z, i_, j_]
            m_dw[z, i_, j_] = dp_v * m_v[z, i_, j_]

        # Pass A: dbk_m[i,d] = sum_{s<i} (-dM[i,s]) * k[s,d] * exp2(g2[i,d]-g2[s,d]).
        # Pass B: dk_m[s,d] = sum_{i>s} (-dM[i,s]) * (b*k)[i,d] * exp2(g2[i,d]-g2[s,d]).
        # Two trace-time shapes (skeptic finding: the triangular form's data-dependent
        # inner bounds have no wheel precedent):
        #   default    — triangular loops; the range IS the mask, exponents <= 0.
        #   pred_tail  — the mamba2_ssd SegSum idiom (v452:3021-3033): constexpr
        #                full-triangle unroll + runtime mask of the exponent to -inf
        #                BEFORE exp2 (exp2(-inf) = 0 exactly; dead dM entries are
        #                finite, so 0 * garbage = 0) — the masked_decay_rel structure.
        if cutlass.const_expr(pred_tail):
            neg_inf = cutlass.Float32(-float("inf"))
            for e in cutlass.range(tidx, c * d_k, THREADS):
                i_, d_ = e // d_k, e % d_k
                g2_i = m_g2[z, i_, d_]
                acc_a = cutlass.Float32(0.0)
                for s_ in cutlass.range_constexpr(c):
                    arg = g2_i - m_g2[z, s_, d_]
                    if s_ >= i_:
                        arg = neg_inf
                    dec = cute.math.exp2(arg, fastmath=True)
                    acc_a += (-m_dm_s[z, i_, s_]) * m_k[z, s_, d_] * dec
                m_dbkm[z, i_, d_] = acc_a
            for e in cutlass.range(tidx, c * d_k, THREADS):
                s_, d_ = e // d_k, e % d_k
                g2_s = m_g2[z, s_, d_]
                acc_b = cutlass.Float32(0.0)
                for i2 in cutlass.range_constexpr(c):
                    arg = m_g2[z, i2, d_] - g2_s
                    if i2 <= s_:
                        arg = neg_inf
                    dec = cute.math.exp2(arg, fastmath=True)
                    acc_b += (-m_dm_s[z, i2, s_]) * (m_bg[z, i2, d_] * m_k[z, i2, d_]) * dec
                m_dkm[z, s_, d_] = acc_b
        else:
            for e in cutlass.range(tidx, c * d_k, THREADS):
                i_, d_ = e // d_k, e % d_k
                g2_i = m_g2[z, i_, d_]
                acc_a = cutlass.Float32(0.0)
                for s_ in cutlass.range(i_):
                    dec = cute.math.exp2(g2_i - m_g2[z, s_, d_], fastmath=True)
                    acc_a += (-m_dm_s[z, i_, s_]) * m_k[z, s_, d_] * dec
                m_dbkm[z, i_, d_] = acc_a
            for e in cutlass.range(tidx, c * d_k, THREADS):
                s_, d_ = e // d_k, e % d_k
                g2_s = m_g2[z, s_, d_]
                acc_b = cutlass.Float32(0.0)
                for i2 in cutlass.range(s_ + 1, c, 1):
                    dec = cute.math.exp2(m_g2[z, i2, d_] - g2_s, fastmath=True)
                    acc_b += (-m_dm_s[z, i2, s_]) * (m_bg[z, i2, d_] * m_k[z, i2, d_]) * dec
                m_dkm[z, s_, d_] = acc_b

        cute.arch.barrier()

        # Finals + reverse-cumsum over C: one key channel d per thread, serial in i
        # (fixed order -> deterministic). dg2 = dq_wy*q_kbg*ln2 + ln2*(bk*dbk_m - k*dk_m).
        for d_ in cutlass.range(tidx, d_k, THREADS):
            run = cutlass.Float32(0.0)
            for ii in cutlass.range(c):
                i_ = c - 1 - ii
                kv = m_k[z, i_, d_]
                bv = m_bg[z, i_, d_]
                gam = cute.math.exp2(m_g2[z, i_, d_], fastmath=True)
                dq_v = m_dq_s[z, i_, d_]
                dbkm_v = m_dbkm[z, i_, d_]
                dkm_v = m_dkm[z, i_, d_]
                dbk = dq_v * gam + dbkm_v
                m_db[z, i_, d_] = dbk * kv
                m_dk[z, i_, d_] = dbk * bv + dkm_v
                dg2_v = dq_v * (bv * kv * gam) * LN2_F + LN2_F * (bv * kv * dbkm_v - kv * dkm_v)
                run += dg2_v
                m_dg[z, i_, d_] = RCP_LN2_F * run

        # ONE relinquish after all MMA issue; ONE free.
        tmem.relinquish_alloc_permit()
        pipeline.sync(barrier_id=2)
        tmem.free(base)

    # ─── host jit ─────────────────────────────────────────────────────────────

    @cute.jit
    def _k2f_host(
        m_att: cute.Tensor,
        m_adu: cute.Tensor,
        m_adwy: cute.Tensor,
        m_sdt: cute.Tensor,
        m_bdu: cute.Tensor,
        m_bdwy: cute.Tensor,
        m_bpwv: cute.Tensor,
        m_bqkbg: cute.Tensor,
        m_bt: cute.Tensor,
        m_bsx: cute.Tensor,
        m_dp_c: cute.Tensor,
        m_dq_c: cute.Tensor,
        m_dt_c: cute.Tensor,
        m_xn_c: cute.Tensor,
        m_dm_c: cute.Tensor,
        m_dp_s: cute.Tensor,
        m_dq_s: cute.Tensor,
        m_dm_s: cute.Tensor,
        m_xn_s: cute.Tensor,
        m_sx_s: cute.Tensor,
        m_k: cute.Tensor,
        m_bg: cute.Tensor,
        m_v: cute.Tensor,
        m_w: cute.Tensor,
        m_g2: cute.Tensor,
        m_dbkm: cute.Tensor,
        m_dkm: cute.Tensor,
        m_dk: cute.Tensor,
        m_dv: cute.Tensor,
        m_db: cute.Tensor,
        m_dw: cute.Tensor,
        m_dg: cute.Tensor,
        c: cutlass.Constexpr,
        d_k: cutlass.Constexpr,
        d_v: cutlass.Constexpr,
        pred_tail: cutlass.Constexpr,
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
        mma_dp = cute.make_tiled_mma(op)
        mma_qwy = cute.make_tiled_mma(op)
        mma_dt = cute.make_tiled_mma(op)
        mma_x = cute.make_tiled_mma(op)
        mma_dm = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(mma_dp, _MNK_TILER, m_att.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(mma_dp, _MNK_TILER, m_bdu.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        at_att1, t_att1 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_att, a1, _MNK_TILER, mma_dp)
        at_att2, t_att2 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_att, a1, _MNK_TILER, mma_qwy)
        at_att6, t_att6 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_att, a1, _MNK_TILER, mma_dm)
        at_adu, t_adu = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_adu, a1, _MNK_TILER, mma_dt)
        at_adwy, t_adwy = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_adwy, a1, _MNK_TILER, mma_dt)
        at_sdt, t_sdt = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_sdt, a1, _MNK_TILER, mma_x)
        at_bdu, t_bdu = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bdu, b1, _MNK_TILER, mma_dp)
        at_bdwy, t_bdwy = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bdwy, b1, _MNK_TILER, mma_qwy)
        at_bpwv, t_bpwv = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bpwv, b1, _MNK_TILER, mma_dt)
        at_bqkbg, t_bqkbg = cute.nvgpu.make_tiled_tma_atom_B(
            op_tma, m_bqkbg, b1, _MNK_TILER, mma_dt
        )
        at_bt, t_bt = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bt, b1, _MNK_TILER, mma_x)
        at_bsx, t_bsx = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bsx, b1, _MNK_TILER, mma_dm)
        _kern_k2f(
            mma_dp,
            mma_qwy,
            mma_dt,
            mma_x,
            mma_dm,
            at_att1,
            t_att1,
            at_att2,
            t_att2,
            at_att6,
            t_att6,
            at_adu,
            t_adu,
            at_adwy,
            t_adwy,
            at_sdt,
            t_sdt,
            at_bdu,
            t_bdu,
            at_bdwy,
            t_bdwy,
            at_bpwv,
            t_bpwv,
            at_bqkbg,
            t_bqkbg,
            at_bt,
            t_bt,
            at_bsx,
            t_bsx,
            m_dp_c,
            m_dq_c,
            m_dt_c,
            m_xn_c,
            m_dm_c,
            m_dp_s,
            m_dq_s,
            m_dm_s,
            m_xn_s,
            m_sx_s,
            m_k,
            m_bg,
            m_v,
            m_w,
            m_g2,
            m_dbkm,
            m_dkm,
            m_dk,
            m_dv,
            m_db,
            m_dw,
            m_dg,
            a_layout,
            b_layout,
            c,
            d_k,
            d_v,
            pred_tail,
        ).launch(grid=(1, 1, m_dp_c.layout.shape[2]), block=(THREADS, 1, 1), stream=stream)

    _k2f_cache: dict[tuple[int, ...], object] = {}


# ─── public launcher ──────────────────────────────────────────────────────────


def run_k2_fused(
    k: Tensor,
    v: Tensor,
    b: Tensor,
    w: Tensor,
    g2: Tensor,
    t_mat: Tensor,
    dwy: Tensor,
    du: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fused channel-wise WY-VJP — ONE launch for all Z=B·H·NT chunks.

    Packs via :func:`gdn2_bwd_wy_cw._k2f_pack`, launches ``_kern_k2f`` on torch's current
    stream (CUDA-graph-capturable; ``maybe_sync`` at the end only). Returns
    ``(dk2, dv, db, dw, dg2)`` head-major chunked, fp32. Tile contract: C=64, d_k=128,
    d_v=64 exactly (:func:`gdn2_bwd_wy_cw.k2f_dims_ok`). ~1.07 GB ``decay_rel`` is never
    materialized; staged operands + scratch total ~300 MB @B2/L2048/H8.
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    import os

    # FMR_K2F_PRED_TAIL=1 selects the predicated-tail fallback (the mamba2_ssd SegSum
    # shape) if the triangular loops' data-dependent bounds misbehave on this wheel.
    pred_tail = bool(os.environ.get("FMR_K2F_PRED_TAIL"))

    bsz, hh, nt, c, d_k = k.shape
    d_v = v.shape[-1]
    if not k2f_dims_ok(c, d_k, d_v):
        raise ValueError(
            f"run_k2_fused is single-N-tile (C=64, d_k=128, d_v=64); got "
            f"c={c}, d_k={d_k}, d_v={d_v} — d_v=128 is the N-tiling increment."
        )
    z = bsz * hh * nt
    dev = k.device
    f16, f32 = torch.float16, torch.float32

    buf = _k2f_pack(k, v, b, w, g2, t_mat, dwy, du)
    a_tt = buf["a_tt"].to(f16).contiguous()
    a_du = buf["a_du"].to(f16).contiguous()
    a_dwy = buf["a_dwy"].to(f16).contiguous()
    b_du = buf["b_du"].to(f16).contiguous()
    b_dwy = buf["b_dwy"].to(f16).contiguous()
    b_pwv = buf["b_pwv"].to(f16).contiguous()
    b_qkbg = buf["b_qkbg"].to(f16).contiguous()
    b_t = buf["b_t"].to(f16).contiguous()
    kf = buf["k"].to(f32).contiguous()
    bf = buf["b"].to(f32).contiguous()
    vf = buf["v"].to(f32).contiguous()
    wf = buf["w"].to(f32).contiguous()
    g2f = buf["g2"].to(f32).contiguous()

    # Device-written GEMM-operand scratch: zero-filled so K-pad lanes stay exact zeros.
    s_dt = torch.zeros(z, 128, 128, dtype=f16, device=dev)
    s_x = torch.zeros(z, 64, 128, dtype=f16, device=dev)
    s_xn = torch.zeros(z, 128, 64, dtype=f16, device=dev)
    s_dp32 = torch.zeros(z, 128, 64, dtype=f32, device=dev)
    s_dq32 = torch.zeros(z, 128, 128, dtype=f32, device=dev)
    s_dm32 = torch.zeros(z, 128, 64, dtype=f32, device=dev)
    s_dbkm = torch.zeros(z, c, d_k, dtype=f32, device=dev)
    s_dkm = torch.zeros(z, c, d_k, dtype=f32, device=dev)
    o_dk = torch.zeros(z, c, d_k, dtype=f32, device=dev)
    o_dv = torch.zeros(z, c, d_v, dtype=f32, device=dev)
    o_db = torch.zeros(z, c, d_k, dtype=f32, device=dev)
    o_dw = torch.zeros(z, c, d_v, dtype=f32, device=dev)
    o_dg = torch.zeros(z, c, d_k, dtype=f32, device=dev)

    stream = _cur_stream()
    args = (
        _mark_b(a_tt),
        _mark_b(a_du),
        _mark_b(a_dwy),
        _mark_b(s_dt),
        _mark_b(b_du),
        _mark_b(b_dwy),
        _mark_b(b_pwv),
        _mark_b(b_qkbg),
        _mark_b(b_t),
        _mark_b(s_x),
        _mark_b(s_dp32),
        _mark_b(s_dq32),
        _mark_b(s_dt),
        _mark_b(s_xn),
        _mark_b(s_dm32),
        _mark_simt(s_dp32),
        _mark_simt(s_dq32),
        _mark_simt(s_dm32),
        _mark_simt(s_xn),
        _mark_simt(s_x),
        _mark_simt(kf),
        _mark_simt(bf),
        _mark_simt(vf),
        _mark_simt(wf),
        _mark_simt(g2f),
        _mark_simt(s_dbkm),
        _mark_simt(s_dkm),
        _mark_simt(o_dk),
        _mark_simt(o_dv),
        _mark_simt(o_db),
        _mark_simt(o_dw),
        _mark_simt(o_dg),
        c,
        d_k,
        d_v,
        pred_tail,
        stream,
    )
    key = (z, c, d_k, d_v, int(pred_tail))
    ex = _k2f_cache.get(key)
    if ex is None:
        ex = cute.compile(_k2f_host, *args)
        _k2f_cache[key] = ex
    # Constexpr args (c, d_k, d_v, pred_tail) are baked at compile and DROPPED from the
    # runtime signature — the call tuple is the 32 marked tensors + the CUstream.
    ex(*args[:32], stream)
    maybe_sync()
    return (
        o_dk.reshape(bsz, hh, nt, c, d_k),
        o_dv.reshape(bsz, hh, nt, c, d_v),
        o_db.reshape(bsz, hh, nt, c, d_k),
        o_dw.reshape(bsz, hh, nt, c, d_v),
        o_dg.reshape(bsz, hh, nt, c, d_k),
    )
