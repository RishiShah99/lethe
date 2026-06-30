"""Isolate the inc-B2 launch-SIGSEGV: in-loop dynamic-L local_tile + tma_partition.

loop_gemm_repro.py established that loop-wrapping a proven tcgen05 GEMM is fine (`loop` GO)
and that the in-loop TMEM-handle idiom is fine (`loopfix` GO). Those keep the tile/TMA
derivation OUTSIDE the scf.for. inc-B2 cannot: G1's source tile is selected per chunk by a
loop-dependent L coordinate (lid = bh·NT + i_t), so ``local_tile`` + ``tma_partition`` +
``make_fragment`` are re-derived INSIDE the loop every iteration. This file tests exactly that
on the clean proven body (no dh-copy, no glue, no round-trip):

  static  in-loop local_tile/tma_partition with a CONSTANT L coord (0) — derivation in-loop,
          index static.  out[it] = a[0] @ b[0]^T  for all it.
  dyn     in-loop local_tile/tma_partition with the loop var as the L coord — the EXACT inc-B2
          G1 pattern.  out[it] = a[it] @ b[it]^T.

Interpretation:
  static GO + dyn crash  ⇒ the in-loop DYNAMIC-L TMA coord is the fault (fix: pre-derive the
                           partition outside, index the L mode by the loop var inside).
  static crash           ⇒ in-loop tile/TMA derivation itself is the fault, independent of coord.
  both GO                ⇒ neither is the fault; the residual is the dh-copy preamble or the
                           two-GEMM TMEM sharing / round-trip — look there next.

Run on the box (one process per mode):
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/loop_tile_repro.py --mode static --L 4
    PYTHONPATH=src:. ~/cuteenv/bin/python scratch/loop_tile_repro.py --mode dyn --L 4
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

    @cute.kernel
    def _kern_tile(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
        ll: cutlass.Constexpr, dyn: cutlass.Constexpr,
    ) -> None:
        # One CTA; reverse-free forward loop over L tiles. local_tile / tma_partition / fragments
        # / epilogue setup are ALL re-derived INSIDE the loop (the inc-B2 G1 structure). dyn=True
        # selects tile `it` (loop var) per iteration; dyn=False fixes the L coord at 0.
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

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

        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)
        tmem.wait_for_alloc()
        tmem.relinquish_alloc_permit()

        # range_constexpr UNROLLS the reverse loop (ll is Constexpr) → NO scf.for. The tcgen05
        # epilogue copy atom (make_tmem_copy) cannot live in a dynamic scf.for in this DSL
        # (inside → launch-SIGSEGV; hoisted-and-used-inside → "live after conversion" ICE);
        # unrolling makes each iteration the proven straight-line body.
        for it in cutlass.range_constexpr(ll):
            tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc_frag.layout)
            if cutlass.const_expr(dyn):
                lid = it
            else:
                lid = 0
            gA = cute.local_tile(mA, _MNK_TILER, (0, 0, None, lid), proj=(1, None, 1))
            gB = cute.local_tile(mB, _MNK_TILER, (0, 0, None, lid), proj=(None, 1, 1))
            gC = cute.local_tile(mC, _MNK_TILER, (0, 0, None, it), proj=(1, 1, None))
            thr_mma = tiled_mma.get_slice(0)
            tCgC = thr_mma.partition_C(gC)
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_b, 0, cute.make_layout(1),
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
                    cute.copy(tma_a, tAgA[(None, ab_e.count)], tAsA[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
                    cute.copy(tma_b, tBgB[(None, ab_e.count)], tBsB[(None, ab_e.index)], tma_bar_ptr=ab_e.barrier)
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
        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.kernel
    def _kern_tile_hoist(
        tiled_mma: cute.TiledMma,
        tma_a: cute.CopyAtom, mA: cute.Tensor,
        tma_b: cute.CopyAtom, mB: cute.Tensor,
        mC: cute.Tensor,
        a_layout: cute.ComposedLayout, b_layout: cute.ComposedLayout,
        ll: cutlass.Constexpr,
    ) -> None:
        # Candidate fix: every `make_*` atom/fragment is created OUTSIDE the loop (they are
        # L-independent). Inside the loop only the L-dependent VIEWS (local_tile, tma_partition,
        # partition_C/D, zipped_divide) + the TMA copies + MMA + epilogue copies. dyn coord = it.
        # If this is GO and `dyn` is 139, the launch fault is the in-loop make_* creation, and the
        # minimal inc-B2 fix is to hoist tCrA/tCrB/tmem_atom/tmem_copy/tDtC/rmem out of the loop.
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

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

        # HOISTED make_*: fragments over the fixed sA/sB, the TMEM accumulator + its copy atoms,
        # the rmem staging — all L-independent.
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(_MNK_TILER[:2])
        tCtAcc_frag = tiled_mma.make_fragment_C(acc_shape)
        tmem.wait_for_alloc()
        tCtAcc = cute.make_tensor(tmem.retrieve_ptr(_acc), tCtAcc_frag.layout)
        tmem.relinquish_alloc_permit()
        epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1])),)
        tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
        tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), _acc)
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
        tmem_thr = tmem_copy.get_slice(tidx)
        tDtC = tmem_thr.partition_S(tCtAcc_epi)
        thr_mma = tiled_mma.get_slice(0)
        # rmem epilogue fragments sized from a static tile-0 GMEM dest (per-tile shape is
        # L-independent); the in-loop tDgC has the same shape, only a different L offset.
        gC0 = cute.local_tile(mC, _MNK_TILER, (0, 0, None, 0), proj=(1, 1, None))
        gC0_epi = cute.zipped_divide(thr_mma.partition_C(gC0), epi_tiler)
        tDgC0 = tmem_thr.partition_D(gC0_epi)
        tCrAcc = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _acc)
        tCrC = cute.make_rmem_tensor(tDgC0[None, None, 0].shape, _io)

        # range_constexpr UNROLLS (no scf.for) AND all make_* atoms are hoisted ONCE above and
        # reused: dodges make_tmem_copy-in-scf.for (SIGSEGV/ICE) and per-GEMM re-create (device
        # fault) simultaneously — the last untested cell of the multi-GEMM matrix.
        for it in cutlass.range_constexpr(ll):
            gA = cute.local_tile(mA, _MNK_TILER, (0, 0, None, it), proj=(1, None, 1))
            gB = cute.local_tile(mB, _MNK_TILER, (0, 0, None, it), proj=(None, 1, 1))
            gC = cute.local_tile(mC, _MNK_TILER, (0, 0, None, it), proj=(1, 1, None))
            tCgC = thr_mma.partition_C(gC)
            tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
                tma_a, 0, cute.make_layout(1),
                cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
                tma_b, 0, cute.make_layout(1),
                cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            gC_epi = cute.zipped_divide(tCgC, epi_tiler)
            tDgC = tmem_thr.partition_D(gC_epi)

            if warp_idx == 0:
                acc_empty = acc_prod.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for _kt in cutlass.range(cute.size(gA, mode=[2]), prefetch_stages=AB_STAGES - 1):
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

            acc_f = acc_cons.wait_and_advance()
            for i in cutlass.range(cute.size(tDtC, mode=[2])):
                cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
                tCrC.store(tCrAcc.load().to(_io))
                cute.autovec_copy(tCrC, tDgC[None, None, i])
            acc_f.release()
            pipeline.sync(barrier_id=1)
        tmem.free(tmem.retrieve_ptr(_acc))

    @cute.jit
    def _host_tile(
        a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
        ll: cutlass.Constexpr, dyn: cutlass.Constexpr, hoist: cutlass.Constexpr,
    ) -> None:
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
        if cutlass.const_expr(hoist):
            _kern_tile_hoist(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout, ll).launch(
                grid=(1, 1, 1), block=(THREADS, 1, 1)
            )
        else:
            _kern_tile(tiled_mma, a_atom, a_t, b_atom, b_t, c, a_layout, b_layout, ll, dyn).launch(
                grid=(1, 1, 1), block=(THREADS, 1, 1)
            )

    def _mark_b(t: Tensor) -> object:
        v = t.contiguous().permute(1, 2, 0)  # [Z,R,S] -> [R,S,Z], S unit-stride, Z largest
        return from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    def run(mode: str, ll: int) -> tuple[Tensor, Tensor, Tensor]:
        dev = "cuda"
        torch.manual_seed(0)
        a = torch.randn(ll, D_K, D_K, dtype=torch.float16, device=dev)
        b = torch.randn(ll, D_V, D_K, dtype=torch.float16, device=dev)
        out = torch.zeros(ll, D_K, D_V, dtype=torch.float16, device=dev)
        dyn = mode != "static"  # static fixes the L coord at 0; dyn/hoist select tile `it`
        hoist = mode == "hoist"
        ca, cb, cc = _mark_b(a), _mark_b(b), _mark_b(out)
        ex = cute.compile(_host_tile, ca, cb, cc, ll, dyn, hoist)
        ex(ca, cb, cc, ll, dyn, hoist)
        torch.cuda.synchronize()
        return a, b, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "dyn", "hoist"], required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    res: dict[str, Any] = {"mode": args.mode, "L": args.L}
    try:
        if not _HAVE:
            raise RuntimeError("CuTe DSL toolchain unavailable (not an sm_100 box)")
        a, b, out = run(args.mode, args.L)
        got = out.float()
        if args.mode in ("dyn", "hoist"):
            ref = torch.bmm(a.float(), b.float().transpose(-1, -2))  # [L,128,64]
        else:
            one = a[0].float() @ b[0].float().t()  # [128,64]
            ref = one.unsqueeze(0).expand(args.L, -1, -1)
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
    print(f"\nGO={res.get('GO')} mode={args.mode} L={args.L}")


if __name__ == "__main__":
    main()
