"""Compute-fed-operand tcgen05 GEMM probe — RECON RESULT: the SIMT smem-fill is NOT a
Blackwell idiom. Keep as evidence; do NOT pursue this path for inc-B2.

Goal was to write a *compute*/SIMT value into an MMA SMEM operand's swizzled layout
(``make_smem_layout_a``) and have tcgen05 read it. Findings (2026-06-28, B200):
  * The MMA smem operand layout is MMA-PRE-TILED + swizzled, e.g. ``((128,16),1,(4,2),1)``
    for tiler (128,64,128) — NOT a flat [M,K]. So a flat gmem tile won't index/copy into it.
  * ``cute.copy(cp, partition_S(gmem), partition_D(smem))`` → rank 4 vs 5; the swizzled smem
    partition carries an extra structural mode. ``cute.filter_zeros`` on both (the library's
    own _tma_copy_impl technique) does NOT reconcile it (the extra mode is non-zero-stride).
  * ``utils.block_copy`` supports ONLY TMA and S2T (smem→tmem) ops — a generic/SIMT copy
    raises NotImplementedError.
  * The shipped ``utils/gemm/sm100.py`` stages MMA operands ONLY via TMA (gmem→smem); its
    register→smem copy is epilogue-only (``make_tiled_copy_D`` tied to the TMEM→reg tiling),
    not an MMA A/B operand fill.
CONCLUSION: on Blackwell you do not SIMT-fill an MMA smem operand. A resident
accumulator-that's-also-an-operand must be either (a) GMEM round-tripped (write→gmem,
TMA back — proven mechanics; the fused inc-B v0) or (b) TMEM-resident read via a TS-mode
MMA (``a_src=OperandSource.TMEM``; inc-C, and ``block_copy`` supports the S2T staging).

Run on the box (reproduces the rank-4-vs-5 finding):
    PYTHONPATH=src uv run --no-sync python scratch/gdn2_cfa_smoke.py
"""
# NB: no `from __future__ import annotations` — PEP 563 breaks @cute.struct.

import json
import traceback
from pathlib import Path
from typing import Any

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mnk_inst = (128, 64, 16)
mnk_tiler = (128, 64, 128)
THREADS = 128


@cute.struct
class SharedStorage:
    acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    tmem_buf: cutlass.Int32


@cute.kernel
def _kern(
    tiled_mma: cute.TiledMma,
    mA: cute.Tensor,
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
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(io_dtype, a_layout.outer, byte_alignment=128, swizzle=a_layout.inner)
    sB = smem.allocate_tensor(io_dtype, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)

    gA = cute.local_tile(mA, mnk_tiler, coord, proj=(1, None, 1))  # [M, K, nk]
    gB = cute.local_tile(mB, mnk_tiler, coord, proj=(None, 1, 1))  # [N, K, nk]
    gC = cute.local_tile(mC, mnk_tiler, coord, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mnk_tiler[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    # --- the keystone: fill sA, sB from GMEM via the MMA-consistent partition ---
    # block_copy handles the gmem(1 atom-mode)/swizzled-smem(2 atom-mode) partition
    # split internally; the MMA then reads the same swizzle. gA[None,None,0] / sA[..,0]
    # drop the size-1 nk / stage modes to the bare [M,K] tile.
    # block_copy is TMA/S2T-only; for a SIMT gmem->swizzled-smem fill the library's own
    # technique (see _tma_copy_impl) is filter_zeros on both partitions — it strips the
    # swizzle's zero-stride modes so partition_S (gmem, dense) and partition_D (smem,
    # swizzled) become congruent. This is the keystone B2 needs: a compute/SIMT value
    # written into the MMA operand's swizzle, then read by the MMA.
    cp = cute.make_copy_atom(sm100_utils.CopyUniversalOp(), io_dtype)
    tca = cute.make_tiled_copy_A(cp, tiled_mma)
    tcb = cute.make_tiled_copy_B(cp, tiled_mma)
    thr_a = tca.get_slice(tidx)
    thr_b = tcb.get_slice(tidx)
    cute.copy(cp, cute.filter_zeros(thr_a.partition_S(gA)), cute.filter_zeros(thr_a.partition_D(sA)))
    cute.copy(cp, cute.filter_zeros(thr_b.partition_S(gB)), cute.filter_zeros(thr_b.partition_D(sB)))
    pipeline.sync(barrier_id=1)

    acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar.data_ptr(),
    ).make_participants()

    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)

    sub = mnk_tiler[1] // 64
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // sub),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), acc_dtype)
    tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr = tmem_copy.get_slice(tidx)
    tDtC = tmem_thr.partition_S(tCtAcc_epi)
    tDgC = tmem_thr.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    if warp_idx == 0:
        acc_empty = acc_prod.acquire_and_advance()
        for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
            cc = (None, None, kb)
            cute.gemm(tiled_mma, tCtAcc, tCrA[cc], tCrB[cc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_f = acc_cons.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_f.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def _host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
    op = tcgen05.MmaF16BF16Op(
        io_dtype, acc_dtype, mnk_inst, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_layout = sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, a.element_type, 1)
    b_layout = sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, b.element_type, 1)
    grid = cute.ceil_div((*c.layout.shape, 1), mnk_tiler[:2])
    _kern(tiled_mma, a, b, c, a_layout, b_layout).launch(grid=grid, block=(THREADS, 1, 1))


def _mark(t: torch.Tensor) -> object:
    return (
        from_dlpack(t.contiguous(), assumed_align=16)
        .mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=16)
    )


def _run(m: int, n: int, k: int) -> dict[str, Any]:
    torch.manual_seed(0)
    td = cutlass_torch.dtype(io_dtype)
    a = torch.empty(m, k, dtype=torch.int32).random_(-2, 2).to(dtype=td, device="cuda")
    b = torch.empty(n, k, dtype=torch.int32).random_(-2, 2).to(dtype=td, device="cuda")
    c = torch.zeros(m, n, dtype=td, device="cuda")
    _host(_mark(a), _mark(b), _mark(c))
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    err = (c.float() - ref).abs().max().item()
    return {"mnk": [m, n, k], "max_err": err, "passed": err <= 0.5}


def main() -> None:
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        out["gemm"] = _run(mnk_tiler[0], mnk_tiler[1], mnk_tiler[2])
    except Exception as exc:  # noqa: BLE001
        out["gemm"] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
    out["GO"] = isinstance(out.get("gemm"), dict) and out["gemm"].get("passed", False)
    dest = Path("results/gdn2_cfa_smoke.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nGO={out['GO']}")


if __name__ == "__main__":
    main()
