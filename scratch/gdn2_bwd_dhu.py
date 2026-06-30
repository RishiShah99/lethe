"""K#1 — native Blackwell (sm_100) reverse inter-chunk state scan (GDN backward, B4).

Box bring-up file (→ ``src/flash_mamba_rl/kernels/cute/gdn2_bwd_dhu.py`` at integration).
The clean whitespace cuLA imports from fla (Triton); no native sm_100 impl exists. Built
on the proven single-tile GEMM idiom in ``scratch/tcgen05_gemm_smoke.py``.

Contract (``docs/gdn2_phase2_refs.md`` §B4; ground truth = ``references/gdn2_chunkwise.py``
via ``scratch/gen_k1_bundle.py``). One CTA owns one ``(b, hv)``; ``b_dh∈[d_k,d_v]`` fp32 is
the recurrent state-grad accumulator carried across the reverse chunk loop. Per chunk
``i_t=NT−1…0`` (gamma=exp2(g2), decay=exp2(g_last−g2), per-row over C):

  1. dh[i_t] = b_dh
  2. b_dv = (k @ b_dh)·decay + dv_local              store dv2[i_t]
  3. b_dh = exp2(g_last)·b_dh
  4. b_dh += (q·gamma)^T @ do − w^T @ b_dv          ;  dh0 = b_dh after the loop

Bring-up ladder (each micro-gated):
  A. NT=1 (THIS increment): b_dh starts 0 → dh=0, dv2=dv_local, and
     dh0 = (q·gamma)^T@do − w^T@dv_local = [qg | −w] @ [do | dv_local]^T — ONE tcgen05
     GEMM (M=d_k=128, N=d_v=64, K=2C=128) per (b,hv). Proves the MMA + bf16 epilogue.
  B. NT>1: in-kernel reverse loop + the b_dh carry (G1 reads resident b_dh as an operand).
  C. TMEM-resident b_dh (the headline traffic lever).

Numerics: bf16 I/O, fp32 accumulate; exp2 on g pre-scaled by RCP_LN2; deterministic, no
atomics. Off-box this imports cleanly and compiles nothing.
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes @cute.struct fields.

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

    @cute.jit
    def _gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
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
        _gemm_kernel(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout).launch(
            grid=grid, block=(THREADS, 1, 1)
        )

    _gemm_compiled = None  # cached cute.compile of _gemm_host (the fixed (128,128,64) tile)

    def _gemm_aa(a: Tensor, b: Tensor, out: Tensor) -> None:
        """Run increment A's proven (128,64,128) GEMM: ``out[128,64] = a[128,128] @ b[64,128]^T``.

        The whole increment-B reverse loop is expressed in this one config (natural
        orientation): GA is increment A's stacked GEMM verbatim, and G1 (k@b_dh) pads
        its M=C=64 output to 128 — both land on the M=128 TMEM fragment the epilogue
        was proven for (an M=64 accumulator trips ``make_tmem_copy``). ``out`` must be a
        fresh, standalone, contiguous tensor (the box write-back lesson).

        THE dispatch fix (the launch-overhead win). Calling the ``@cute.jit`` ``_gemm_host``
        directly re-traces its host body (tiled-MMA, smem layouts, TMA atoms) on EVERY call
        — ~190 ms to launch a ~5 us kernel (measured B200, scratch/probe_gemm_overhead.py).
        Every ``_gemm_aa`` call is the identical (128,128,64) tile, so ``cute.compile`` it
        ONCE and reuse the compiled executable: ~190 ms/call → ~0.04 ms/call
        (scratch/probe_gemm_compiled.py). No per-GEMM ``cuda.synchronize`` either — the
        launch is default-stream ordered against the loop's torch ops that read this ``out``
        and feed the next GEMM; the caller syncs ONCE at the end of its reverse loop.
        """
        global _gemm_compiled
        ca, cb, cc = _mark(a.contiguous()), _mark(b.contiguous()), _mark(out)
        if _gemm_compiled is None:
            _gemm_compiled = cute.compile(_gemm_host, ca, cb, cc)
        _gemm_compiled(ca, cb, cc)

    # ------------------------------------------------------------------
    # Lever B — batched (128,64,128) GEMM: L as grid-z, one CTA per batch element.
    # Strict generalization of _gemm_kernel/_gemm_host: tensors gain an L mode
    # ([M,K,L]/[N,K,L]/[M,N,L]); the coord gains bidz; the TMA atoms, proj, mainloop
    # and epilogue are byte-identical (local_tile bakes the L offset into gA/gB/gC).
    # Idiom: NVIDIA/cutlass CuTeDSL hopper/blackwell dense_gemm L-mode (grid-z).
    # ------------------------------------------------------------------
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

    @cute.jit
    def _gemm_host_b(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
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
        grid = (
            cute.ceil_div(c.layout.shape[0], _MNK_TILER[0]),
            cute.ceil_div(c.layout.shape[1], _MNK_TILER[1]),
            c.layout.shape[2],
        )
        _gemm_kernel_b(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout).launch(
            grid=grid, block=(THREADS, 1, 1)
        )

    _gemm_b_cache: dict[int, object] = {}  # cute.compile of _gemm_host_b, keyed by batch size Z

    def _gemm_batched(a: Tensor, b: Tensor, out: Tensor) -> None:
        """Batched ``out[z]=a[z]@b[z]^T`` on the proven (128,64,128) config, L=z as grid-z.

        ``a`` [Z,128,128], ``b`` [Z,64,128], ``out`` [Z,128,64] — contiguous, batch-first.
        Collapses Z single-CTA launches into ONE grid-z launch. Carries the dispatch win:
        ``cute.compile`` ONCE per batch size Z (M/N/K fixed at 128/64/128) and reuse.
        ``out`` must be a fresh contiguous tensor (the box write-back lesson).
        """
        z = int(a.shape[0])
        ca, cb, cc = _mark_b(a), _mark_b(b), _mark_b(out)
        ex = _gemm_b_cache.get(z)
        if ex is None:
            ex = cute.compile(_gemm_host_b, ca, cb, cc)
            _gemm_b_cache[z] = ex
        ex(ca, cb, cc)

    def _mark_b(t: Tensor) -> object:
        """Mark a batch-first [Z,R,S] tensor as a cute [R,S,L] tensor (L = largest stride).

        ``t`` contiguous → permute(1,2,0) presents it as [R,S,Z] with S unit-stride and Z the
        outermost (largest-stride) mode — the L-major batched layout the TMA descriptor wants.
        """
        v = t.contiguous().permute(1, 2, 0)
        return from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=1)


def _bmm_tc(x: Tensor, y: Tensor) -> Tensor:
    """Batched ``x[Z,M,K] @ y[Z,K,N] -> [Z,M,N]`` via the batched (128,64,128) a@b^T GEMM.

    The batched twin of :func:`_mm_tc`: per batch element M-pads to 128, K-pads to 128, and
    N-tiles by 64 — every step lands on the ONE proven config; fp16 operands / fp32 accumulate.
    Collapses the per-chunk Python loop into ONE launch per N-tile over all Z (lever B). Looked
    up via the module global so the desk orchestration check can swap a clean fp32 stand-in.
    """
    f16, dev = torch.float16, x.device
    z, m, kk = x.shape
    n = y.shape[-1]
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
    qg = (q * gamma[..., None])  # [B,HV,1,C,d_k]
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

    torch.cuda.synchronize()

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
    """Increment B, box step 1 — host-orchestrated reverse inter-chunk state scan.

    ``b_dh∈[d_k,d_v]`` fp32 is carried in a torch tensor across the reverse chunk loop
    (natural orientation). Each chunk runs two tcgen05 GEMMs through increment A's
    proven (128,64,128) config: GA is its stacked ``[qg|−w]@[do|b_dv]^T`` verbatim, and
    G1 (``k@b_dh``) pads its M=C output to 128. De-risks the full recurrence + the
    padded-G1 use of the proven GEMM on silicon before the in-kernel loop. Returns
    ``(dh, dv2, dh0)`` head-major chunked, matching the K#1 bundle's expected outputs.
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
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
            # G1: b_dv[C,d_v] = k[C,d_k] @ b_dh[d_k,d_v].  MMA out[128,d_v]=A[128,d_k]@B[d_v,d_k]^T
            #   A = k padded M=C->128 ; B = b_dh^T [d_v,d_k] ; take rows [:C].
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

    torch.cuda.synchronize()
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
    """Lever B — reverse-state scan with the (b,hv) groups batched into one GEMM per step.

    Same recurrence as :func:`run_k1_incB_host` (natural orientation, identical math), but the
    ``for i in range(n_bh)`` host loop collapses into the batch dim: per reverse chunk ``it`` the
    two GEMMs (G1 ``k@b_dh``, GA stacked ``[qg|−w]@[do|b_dv]^T``) run ONCE over all n_bh groups
    via :func:`_bmm_tc`. The ``it`` loop stays sequential — it carries ``b_dh``. Returns
    ``(dh, dv2, dh0)`` head-major chunked.
    """
    if not is_available():
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
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

    torch.cuda.synchronize()
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        b_dh.reshape(b, hv, d_k, d_v),
    )


# ------------------------------------------------------------------
# Lever D — inc-B2: the reverse it-loop fused into ONE persistent kernel per (b,hv).
#
# Lever B batched the (b,hv) groups but the reverse it-loop stays a sequential chain of
# batched-GEMM launches + torch glue (NT iterations → the residual gap that grows with NT).
# inc-B2 collapses that chain: one CTA owns one (b,hv) and runs the whole reverse scan
# in-kernel, b_dh resident across chunks. Two GEMMs/chunk land on the SAME proven (128,64,128)
# config as inc-A/inc-B-host — G1 (k@b_dh, M=C padded to 128) and GA ([qg|−w]@[do|b_dv]^T) —
# so the only new mechanics are loop fusion + the GMEM round-trip of b_dh / b_dv (no SIMT
# smem-fill of an MMA operand exists on Blackwell; scratch/gdn2_cfa_smoke.py is the evidence).
#
# The packing + glue are IDENTICAL in value to run_k1_incB_host (box-proven); what is new and
# desk-validatable is the flat-L operand marshalling and the b_ga round-trip layout the kernel
# reads. _run_k1_incB2_modelled runs that exact dataflow in torch — the spec the DSL kernel
# transcribes, and the target of scratch/k1_incB2_orchestration_check.py.
# ------------------------------------------------------------------


def _incb2_pack_scalar(
    q: Tensor,
    k: Tensor,
    w: Tensor,
    g2: Tensor,
    g_last: Tensor,
    do: Tensor,
    dv_local: Tensor,
) -> dict[str, Tensor]:
    """Host-precompute inc-B2's per-chunk operand buffers, flattened over L = n_bh·NT.

    Returns the four static GEMM-operand / glue buffers the persistent kernel TMAs in its
    reverse loop (chunk ``it`` of group ``i`` is the flat tile ``i·NT + it``):

      a_g1  [L, 128, 128]  k padded M=C→128, K=d_k          (G1 operand A)
      a_ga  [L, 128, 128]  [qg | −w]^T = [d_k, 2C]          (GA operand A)
      b_ga  [L, d_v, 2C]   [do^T | 0]; the 0 half is b_dv^T written in-kernel (GA operand B)
      decay [L, C] · dv_local [L, C, d_v] · glast [L]       (the per-chunk SIMT glue)

    Everything is the same orientation run_k1_incB_host feeds ``_gemm_aa`` (out = a @ b^T),
    so the kernel inherits the box-proven packing; only the flat-L stacking is new.
    """
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
        "a_g1": a_g1, "a_ga": a_ga, "b_ga": b_ga,
        "decay": decL, "dv_local": dvlL, "glast": geL,
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
    """Pure-torch model of inc-B2's exact in-kernel dataflow (the kernel's spec).

    Runs the reverse scan on the flat-L packed buffers in the same statement order the
    persistent kernel uses, with the b_ga round-trip (write b_dv^T into b_ga's second half,
    then read the full [do^T | b_dv^T] tile back as GA's B operand). The two ``@`` stand in
    for the kernel's two tcgen05 GEMMs (out = a @ b^T). Device/dtype-agnostic; in fp64 it
    reproduces the K#1 bundle to roundoff. Returns ``(dh, dv2, dh0)`` head-major chunked.
    """
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
    # ------------------------------------------------------------------
    # inc-B2 — the fused persistent reverse-loop kernel (box-burst v0).
    #
    # One CTA per (b,hv) (grid-z = n_bh); the reverse it-loop runs INSIDE. Each chunk fires
    # the two proven (128,64,128) GEMMs (G1, GA) through _gemm_step — byte-identical mainloop +
    # epilogue to the silicon-verified _gemm_kernel_b, only the L coordinate is selected
    # dynamically per chunk. b_dh is resident across chunks via a GMEM round-trip: the SIMT carry
    # writes it back (natural b_dh for output/carry + transposed b_dhT for G1's TMA operand), and
    # b_dv^T is written into b_ga's second half for GA — both re-read by TMA next, so a proxy
    # fence + barrier gates each round-trip (no SIMT smem-fill of an MMA operand exists on
    # Blackwell; gdn2_cfa_smoke.py is the evidence). Values == run_k1_incB_host == the fp64 oracle
    # (locked by scratch/k1_incB2_orchestration_check.py); the kernel only transcribes that flow.
    #
    # BOX STATUS (2026-06-30, B200): COMPILES (cute.compile succeeds); FAILS at launch with a hard
    # host SIGSEGV (core dumped) that survives CUDA_LAUNCH_BLOCKING=1 and is NOT caught by the
    # microgate try/except → it fires inside ex(*args) (the launch), not a recoverable CUDA error.
    # BISECTED via the `bisect` Constexpr knob (--bisect {0,1,2,3,4}; const_expr-guarded so each
    # value compiles only the taken branch). The lever-B batched GEMM is GO=True on the same box
    # (toolchain healthy). RULED OUT — all of these still SIGSEGV at NT=1:
    #   #1 two GEMMs sharing one pipeline + TMEM accumulator  (bisect=1 G1-only crashes)
    #   #2 the b_ga / b_dhT round-trip + proxy fence          (bisect=2 no-round-trip crashes)
    #   #3 the dynamic-L local_tile coord (0,0,None,lid)      (bisect=3 plain `bh` coord crashes)
    #      relinquish_alloc_permit ordering                   (bisect=4 relinquish-after-mainloop crashes)
    # ⇒ The fault is the MINIMAL machinery: ONE tcgen05 GEMM (TMEM acc + async pipelines, allocated
    #   OUTSIDE the loop and used INSIDE) wrapped in `for it in cutlass.range(nt)`, even at NT=1. The
    #   proven _gemm_kernel_b runs the byte-identical body straight-line and works. NEXT (desk-first):
    #   minimal repro = wrap _gemm_kernel_b's mainloop+epilogue in `cutlass.range(1)`; study a
    #   reference looped-tcgen05 kernel (CuTeDSL persistent-GEMM / flash-attn) for the correct in-loop
    #   TMEM/pipeline idiom (likely: retrieve the TMEM ptr / make_fragment_C INSIDE the loop so the
    #   accumulator handle is region-local, not captured across the scf.for boundary). DO NOT grind
    #   blind on the spot box. Debugger tooling (compute-sanitizer + cuda-gdb) is UNUSABLE on this box:
    #   both halt on the DSL's import-time cuGetProcAddress_v2 broken hook, never reaching the kernel.
    # Silicon-fixed already: SIMT-store dtype casts (fp32→_io); the if→scf.if AST preprocess (GEMMs
    # inlined, not a helper; ALSO: bisect `if`s need cutlass.const_expr or they ICE as scf.if with the
    # make_tmem_copy atom live across scf.yield); the dual TMA / SIMT views over shared storage
    # (_mark_b vs _mark_simt); the fence API name (fence_proxy).
    # ------------------------------------------------------------------
    # The two GEMMs are INLINED in _incb2_kernel's loop (below): the DSL's if→scf.if AST
    # preprocess only reaches the @cute.kernel's own body, so `if warp_idx == 0` cannot live
    # in a plain device helper called from inside the scf.for (it raises "dynamic Boolean").
    # ------------------------------------------------------------------
    @cute.kernel
    def _incb2_kernel(
        tiled_mma: cute.TiledMma,
        tma_ag1: cute.CopyAtom, m_ag1: cute.Tensor,
        tma_aga: cute.CopyAtom, m_aga: cute.Tensor,
        tma_bdhT: cute.CopyAtom, m_bdhT: cute.Tensor,
        tma_bga: cute.CopyAtom, m_bga: cute.Tensor,
        m_bdv: cute.Tensor, m_t: cute.Tensor,
        m_bdh: cute.Tensor, m_dh: cute.Tensor, m_dv2: cute.Tensor,
        m_decay: cute.Tensor, m_dvl: cute.Tensor, m_glast: cute.Tensor,
        m_bdhT_s: cute.Tensor, m_bga_s: cute.Tensor, m_bdv_s: cute.Tensor, m_t_s: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
        nt: cutlass.Constexpr, c: cutlass.Constexpr, d_v: cutlass.Constexpr,
        bisect: cutlass.Constexpr,
    ) -> None:
        # bisect: 0 full · 1 G1-only · 2 G1+GA no round-trips (the launch-SIGSEGV bisection knob).
        # m_bdhT / m_bga / m_bdv / m_t are the TMA / epilogue views (mark_b, permuted [R,S,L]);
        # the _s twins are natural-layout SIMT views over the SAME storage (the glue indexes [L,R,S]).
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

        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)
        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc.layout)
        if cutlass.const_expr(bisect != 4):
            tmem.relinquish_alloc_permit()

        for it in cutlass.range(nt):
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
                tma_ag1, 0, cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_bdhT, 0, cute.make_layout(1),
                cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
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
                    cute.copy(tma_ag1, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    cute.copy(tma_bdhT, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
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
                cute.arch.barrier()
                if cutlass.const_expr(bisect == 0):
                    cute.arch.fence_proxy("async.global")  # SIMT store → TMA (async-proxy) read

                # ===== GA: m_t[·,bh] = m_aga[·,lid] @ m_bga[·,lid]^T =====
                gA = cute.local_tile(m_aga, _MNK_TILER, (0, 0, None, lid), proj=(1, None, 1))
                gB = cute.local_tile(m_bga, _MNK_TILER, (0, 0, None, lid), proj=(None, 1, 1))
                gC = cute.local_tile(m_t, _MNK_TILER, (0, 0, None, bh), proj=(1, 1, None))
                thr_mma = tiled_mma.get_slice(0)
                tCgC = thr_mma.partition_C(gC)
                tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                    tma_aga, 0, cute.make_layout(1),
                    cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
                )
                tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                    tma_bga, 0, cute.make_layout(1),
                    cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
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
                    for _kt in cutlass.range(cute.size(gA, mode=[2]), prefetch_stages=AB_STAGES - 1):
                        ab_e = ab_prod.acquire_and_advance()
                        cute.copy(tma_aga, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                        cute.copy(tma_bga, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
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
                cute.arch.barrier()
                if cutlass.const_expr(bisect == 0):
                    cute.arch.fence_proxy("async.global")  # b_dhT round-trip → next chunk's G1 TMA

        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.jit
    def _incb2_host(
        m_ag1: cute.Tensor, m_aga: cute.Tensor, m_bdhT: cute.Tensor, m_bga: cute.Tensor,
        m_bdv: cute.Tensor, m_t: cute.Tensor, m_bdh: cute.Tensor,
        m_dh: cute.Tensor, m_dv2: cute.Tensor,
        m_decay: cute.Tensor, m_dvl: cute.Tensor, m_glast: cute.Tensor,
        m_bdhT_s: cute.Tensor, m_bga_s: cute.Tensor, m_bdv_s: cute.Tensor, m_t_s: cute.Tensor,
        n_bh: cutlass.Constexpr, nt: cutlass.Constexpr, c: cutlass.Constexpr, d_v: cutlass.Constexpr,
        bisect: cutlass.Constexpr,
    ) -> None:
        op = tcgen05.MmaF16BF16Op(
            _io, _acc, _MNK_INST, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, _MNK_TILER, m_ag1.element_type, AB_STAGES)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, _MNK_TILER, m_bga.element_type, AB_STAGES)
        a1 = cute.select(a_layout, mode=[0, 1, 2])
        b1 = cute.select(b_layout, mode=[0, 1, 2])
        op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        at_ag1, t_ag1 = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_ag1, a1, _MNK_TILER, tiled_mma)
        at_aga, t_aga = cute.nvgpu.make_tiled_tma_atom_A(op_tma, m_aga, a1, _MNK_TILER, tiled_mma)
        at_bdhT, t_bdhT = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bdhT, b1, _MNK_TILER, tiled_mma)
        at_bga, t_bga = cute.nvgpu.make_tiled_tma_atom_B(op_tma, m_bga, b1, _MNK_TILER, tiled_mma)
        _incb2_kernel(
            tiled_mma, at_ag1, t_ag1, at_aga, t_aga, at_bdhT, t_bdhT, at_bga, t_bga,
            m_bdv, m_t, m_bdh, m_dh, m_dv2, m_decay, m_dvl, m_glast,
            m_bdhT_s, m_bga_s, m_bdv_s, m_t_s,
            a_layout, b_layout, nt, c, d_v, bisect,
        ).launch(grid=(1, 1, n_bh), block=(THREADS, 1, 1))

    _incb2_cache: dict[tuple[int, ...], object] = {}  # cute.compile keyed (n_bh,nt,c,d_v,bisect)


def _mark_simt(t: Tensor) -> object:
    """Mark a plain contiguous tensor for in-kernel SIMT element access (no TMA atom)."""
    return from_dlpack(t.contiguous(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)


def _incb2_launch(
    a_g1: Tensor, a_ga: Tensor, b_ga: Tensor, b_dh: Tensor,
    decay: Tensor, dvl: Tensor, glast: Tensor,
    n_bh: int, nt: int, c: int, d_k: int, d_v: int,
    *,
    bisect: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mark the inc-B2 operands, compile-cache :func:`_incb2_host`, launch once, return flat.

    Shared by the scalar and channel-wise launchers — the regime difference lives entirely in
    the packed buffers (cw folds decay into ``a_g1`` and passes ``decay = 1``; ``glast`` is
    [L, d_k] in both, broadcast for scalar). ``b_dh`` is the resident state (mutated in place).
    """
    dev = a_g1.device
    f16, f32 = torch.float16, torch.float32
    ll = n_bh * nt

    # The 4 dual-use scratch tensors (bdhT/bga TMA operands + glue-written; bdv/t epilogue
    # outputs + glue-read) get BOTH a _mark_b (permuted [R,S,L]) TMA view and a _mark_simt
    # (natural [L,R,S]) glue view over the SAME contiguous f16 storage.
    a_g1_16 = a_g1.to(f16).contiguous()  # [L,128,128] TMA operand A (G1)
    a_ga_16 = a_ga.to(f16).contiguous()  # [L,128,128] TMA operand A (GA)
    b_ga_16 = b_ga.to(f16).contiguous()  # [L,d_v,2C] TMA operand B (GA); 2nd half written in-kernel
    b_dhT = b_dh.transpose(-1, -2).contiguous().to(f16)  # [n_bh,d_v,d_k] resident G1 operand
    bdv_raw = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)  # G1 epilogue landing
    t_scr = torch.zeros(n_bh, d_k, d_v, dtype=f16, device=dev)  # GA epilogue landing
    dh = torch.zeros(ll, d_k, d_v, dtype=f16, device=dev)
    dv2 = torch.zeros(ll, c, d_v, dtype=f16, device=dev)

    args = (
        _mark_b(a_g1_16), _mark_b(a_ga_16), _mark_b(b_dhT), _mark_b(b_ga_16),
        _mark_b(bdv_raw), _mark_b(t_scr),
        _mark_simt(b_dh), _mark_simt(dh), _mark_simt(dv2),
        _mark_simt(decay.to(f32)), _mark_simt(dvl.to(f32)), _mark_simt(glast.to(f32)),
        _mark_simt(b_dhT), _mark_simt(b_ga_16), _mark_simt(bdv_raw), _mark_simt(t_scr),
        n_bh, nt, c, d_v, bisect,
    )
    key = (n_bh, nt, c, d_v, bisect)
    ex = _incb2_cache.get(key)
    if ex is None:
        ex = cute.compile(_incb2_host, *args)
        _incb2_cache[key] = ex
    ex(*args)
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
    """Lever D — the fused persistent reverse-state scan (one CTA per (b,hv), loop in-kernel).

    Host-packs the flat-L operand buffers (:func:`_incb2_pack_scalar`) and launches
    :func:`_incb2_kernel` ONCE for all n_bh groups; the reverse it-loop and the b_dh / b_dv
    round-trips run in-kernel. The per-chunk values are the box-proven inc-B-host packing,
    desk-gated bit-exact vs the fp64 oracle by scratch/k1_incB2_orchestration_check.py. Returns
    ``(dh, dv2, dh0)`` head-major chunked. BOX-UNTESTED — staged for the focused B200 burst;
    ``run_k1_incB`` stays the proven batched default until this is micro-gated GO.
    """
    if not _HAVE:
        raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
    b, hv, nt, c, d_k = q.shape
    d_v = do.shape[-1]
    n_bh = b * hv
    buf = _incb2_pack_scalar(q, k, w, g2, g_last, do, dv_local)
    b_dh = dht.reshape(n_bh, d_k, d_v).contiguous().float()
    glast = buf["glast"][:, None].expand(n_bh * nt, d_k).contiguous()  # [L] scalar → [L,d_k]
    dh, dv2, dh0 = _incb2_launch(
        buf["a_g1"], buf["a_ga"], buf["b_ga"], b_dh, buf["decay"], buf["dv_local"], glast,
        n_bh, nt, c, d_k, d_v,
        bisect=bisect,
    )
    return (
        dh.reshape(b, hv, nt, d_k, d_v),
        dv2.reshape(b, hv, nt, c, d_v),
        dh0.reshape(b, hv, d_k, d_v),
    )


# Lever B is the default reverse-state path (batched over the (b,hv) groups). The per-group
# host loop (run_k1_incB_host) stays as the proven fallback; --mode incB_host exercises it.
# inc-B2 (the fused persistent reverse loop, run_k1_incB2 above) supersedes both once box-green.
run_k1_incB = run_k1_incB_batched
