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


# run_k1_incB (the in-kernel fused reverse loop, inc-B2) is authored separately; until it
# lands, `--mode incB` falls back to the verified host-orchestrated loop so the harness wiring
# stays exercised. See HANDOFF "inc-B2" + project_gdn2_phase2_k1_staged.
run_k1_incB = run_k1_incB_host
