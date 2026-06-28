"""Single-tile tcgen05 GEMM smoke — the K#1/K#2 MMA building block (Phase-2 Inc-0).

Validates the exact sm_100 primitives the GDN-2 backward kernels need: a tcgen05
``MmaF16BF16Op`` with SMEM operands and an fp32 TMEM accumulator, TMA G2S loads, the
``PipelineTmaUmma``/``PipelineUmmaAsync`` handshake, and the TMEM->RMEM->GMEM epilogue.
One CTA, one output tile, ``C = A @ B^T`` (B is N-major / K-contiguous), checked vs torch.

Idiom adapted from the CUTLASS CuTeDSL tutorial ``fp16_gemm_0.py`` (BSD-3-Clause,
NVIDIA) — studied, re-expressed here as our own minimal smoke. Once green, the three
reverse-state-scan GEMMs (k@dh, (g·q)^T@do, w^T@dv) are this pattern at C=64/d_k=128/d_v=64.

Run on the box: PYTHONPATH=src uv run --no-sync python scratch/tcgen05_gemm_smoke.py
"""
# NB: no `from __future__ import annotations` — PEP 563 stringizes the @cute.struct
# field annotations and the DSL decorator needs the real MemRange objects.

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
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float16
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 16)
mma_tiler_mnk = (128, 256, 64)
threads_per_cta = 128
ab_stages = 2
acc_stage = 1
# 4.5.2's @cute.struct annotation parser accepts MemRange[dtype, NAME] but not an
# arithmetic expression in the subscript — precompute the mbar slot counts as names.
_AB_MBAR = ab_stages * 2
_ACC_MBAR = acc_stage * 2


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, _AB_MBAR]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, _ACC_MBAR]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    mC_mnl: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx, bidy, _ = cute.arch.block_idx()
    mma_coord_mnk = (bidx, bidy, None)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sA = smem.allocate_tensor(
        element_type=io_dtype, layout=a_smem_layout.outer, byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    sB = smem.allocate_tensor(
        element_type=io_dtype, layout=b_smem_layout.outer, byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    num_tma_copy_bytes = cute.size_in_bytes(
        io_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0, 1, 2]))
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=num_tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
    gB = cute.local_tile(mB_nkl, mma_tiler_mnk, mma_coord_mnk, proj=(None, 1, 1))
    gC = cute.local_tile(mC_mnl, mma_tiler_mnk, mma_coord_mnk, proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgC = thr_mma.partition_C(gC)
    tCrA = tiled_mma.make_fragment_A(sA)
    tCrB = tiled_mma.make_fragment_B(sB)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a, 0, cute.make_layout(1),
        cute.group_modes(sA, 0, 3), cute.group_modes(thr_mma.partition_A(gA), 0, 3),
    )
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b, 0, cute.make_layout(1),
        cute.group_modes(sB, 0, 3), cute.group_modes(thr_mma.partition_B(gB), 0, 3),
    )

    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)

    subtile_cnt = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile_cnt),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gC_epi = cute.zipped_divide(tCgC, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gC_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    tCrC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)

    num_k_tiles = cute.size(gA, mode=[2])
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        for _k_tile_idx in cutlass.range(num_k_tiles, prefetch_stages=ab_stages - 1):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier)
            cute.copy(tma_atom_b, tBgB[(None, ab_empty.count)], tBsB[(None, ab_empty.index)],
                      tma_bar_ptr=ab_empty.barrier)
            ab_full = ab_consumer.wait_and_advance()
            for k_block_idx in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                k_block_coord = (None, None, k_block_idx, ab_full.index)
                cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for i in cutlass.range(cute.size(tDtC, mode=[2])):
        cute.copy(tmem_tiled_copy, tDtC[None, None, i], tCrAcc)
        tCrC.store(tCrAcc.load().to(io_dtype))
        cute.autovec_copy(tCrC, tDgC[None, None, i])
    acc_full.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def _gemm_host(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor) -> None:
    op = tcgen05.MmaF16BF16Op(
        io_dtype, acc_dtype, mma_inst_shape_mnk, tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
    )
    tiled_mma = cute.make_tiled_mma(op)
    a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a.element_type, ab_stages)
    b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b.element_type, ab_stages)
    a1 = cute.select(a_smem_layout, mode=[0, 1, 2])
    b1 = cute.select(b_smem_layout, mode=[0, 1, 2])
    op_tma = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(op_tma, a, a1, mma_tiler_mnk, tiled_mma)
    b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(op_tma, b, b1, mma_tiler_mnk, tiled_mma)
    grid_shape = cute.ceil_div((*c.layout.shape, 1), mma_tiler_mnk[:2])
    _gemm_kernel(
        tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor, c,
        a_smem_layout, b_smem_layout,
    ).launch(grid=grid_shape, block=(threads_per_cta, 1, 1))


def _run(m: int, n: int, k: int) -> dict[str, Any]:
    torch.manual_seed(0)
    td = cutlass_torch.dtype(io_dtype)
    a = torch.empty(m, k, dtype=torch.int32).random_(-2, 2).to(dtype=td, device="cuda")
    b = torch.empty(n, k, dtype=torch.int32).random_(-2, 2).to(dtype=td, device="cuda")
    c = torch.zeros(m, n, dtype=td, device="cuda")
    mark = lambda t, d: (  # noqa: E731
        from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=1)
        .mark_compact_shape_dynamic(mode=1, divisibility=d)
    )
    _gemm_host(mark(a, k), mark(b, k), mark(c, n), no_cache=True)
    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a.float(), b.float())
    err = (c.float() - ref).abs().max().item()
    return {"mnk": [m, n, k], "max_err": err, "passed": err <= 0.5}


def main() -> None:
    out: dict[str, Any] = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:
        out["gemm"] = _run(128, 256, 64)
    except Exception as exc:  # noqa: BLE001
        out["gemm"] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
    out["GO"] = isinstance(out.get("gemm"), dict) and out["gemm"].get("passed", False)
    dest = Path("results/tcgen05_gemm_smoke.json")
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(f"\nGO={out['GO']}  ->  {dest}")


if __name__ == "__main__":
    main()
