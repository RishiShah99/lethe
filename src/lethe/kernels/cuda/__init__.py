"""Hand-written CUDA-core kernels (warp-shuffle / cub scan, no tensor cores).

The Triton kernels (``ops/_triton_*``) avoid ``tl.dot`` so they survive the
sm_100 ``#904`` TMEM cliff, but they walk the L axis serially. These CUDA
kernels parallelise L as an O(log L) block scan (cub::BlockScan forward, a
warp-shuffle reverse scan for the backward), the same approach the official
Mamba-1 backward uses, with zero tensor cores / ``mma`` / ``tl.dot``.

JIT-compiled via ``torch.utils.cpp_extension`` (see ``_loader``); import the
launchers lazily so a machine without nvcc/CUDA can still import the package.
"""
