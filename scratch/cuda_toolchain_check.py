"""Increment 0 — de-risk the CUDA toolchain before writing the warp-shuffle scan.

Confirms on the B200 box that we can JIT-compile + run CUDA (incl. cub) via
torch.utils.cpp_extension.load_inline, with NO tensor cores / no tl.dot. Run
this FIRST next session; only once it prints ALL-OK do we write the real
forward/backward warp-shuffle scan kernels (docs/cuda_warpscan_plan.md).

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scratch/cuda_toolchain_check.py
"""

from __future__ import annotations

import torch
from torch.utils.cpp_extension import load_inline

# (1) trivial elementwise kernel — validates nvcc + cpp_extension + ninja.
# (2) cub::BlockScan with the SSM linear-recurrence monoid over one row —
#     validates cub availability + the warp-shuffle scan primitive (the engine
#     the real kernel uses). float2 = (decay a, value b); combine =
#     (a1*a0, a1*b0 + b1) gives h_t = a_t*h_{t-1} + b_t. No tensor cores.
_CUDA = r"""
#include <torch/extension.h>
#include <cub/block/block_scan.cuh>

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    add_kernel<<<(n + threads - 1) / threads, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

struct SSMScanOp {
    __device__ __forceinline__ float2 operator()(const float2& l, const float2& r) const {
        // l older, r newer: combined decay r.x*l.x; value r.x*l.y + r.y
        return make_float2(r.x * l.x, r.x * l.y + r.y);
    }
};

template <int kNThreads>
__global__ void scan_kernel(const float* a, const float* bval, float* h, int L) {
    using BlockScan = cub::BlockScan<float2, kNThreads>;
    __shared__ typename BlockScan::TempStorage temp;
    int t = threadIdx.x;
    float2 v = (t < L) ? make_float2(a[t], bval[t]) : make_float2(1.0f, 0.0f);
    float2 out;
    BlockScan(temp).InclusiveScan(v, out, SSMScanOp());
    if (t < L) h[t] = out.y;  // h_t
}

torch::Tensor ssm_scan(torch::Tensor a, torch::Tensor bval) {
    auto h = torch::empty_like(a);
    int L = a.numel();
    scan_kernel<256><<<1, 256>>>(
        a.data_ptr<float>(), bval.data_ptr<float>(), h.data_ptr<float>(), L);
    return h;
}
"""

_CPP = "torch::Tensor add(torch::Tensor, torch::Tensor);\ntorch::Tensor ssm_scan(torch::Tensor, torch::Tensor);"


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required")
    mod = load_inline(
        name="fmr_cuda_toolcheck",
        cpp_sources=[_CPP],
        cuda_sources=[_CUDA],
        functions=["add", "ssm_scan"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        verbose=True,
    )
    dev = "cuda"

    a = torch.randn(4096, device=dev)
    b = torch.randn(4096, device=dev)
    out = mod.add(a, b)
    add_ok = torch.allclose(out, a + b, atol=1e-5)
    print(f"[1] elementwise add: {'OK' if add_ok else 'FAIL'}")

    # SSM scan parity vs a serial reference (h_t = a_t*h_{t-1} + b_t).
    length = 200
    aa = torch.rand(length, device=dev) * 0.9 + 0.05
    bb = torch.randn(length, device=dev)
    h = mod.ssm_scan(aa, bb)
    ref = torch.empty_like(aa)
    carry = torch.zeros((), device=dev)
    for t in range(length):
        carry = aa[t] * carry + bb[t]
        ref[t] = carry
    scan_ok = torch.allclose(h, ref, atol=1e-4, rtol=1e-4)
    print(
        f"[2] cub BlockScan SSM monoid: {'OK' if scan_ok else 'FAIL'} (max err {(h - ref).abs().max().item():.2e})"
    )

    print(
        f"torch {torch.__version__}, cuda {torch.version.cuda}, gpu {torch.cuda.get_device_name(0)}"
    )
    print("ALL-OK" if (add_ok and scan_ok) else "TOOLCHAIN-PROBLEM")


if __name__ == "__main__":
    main()
