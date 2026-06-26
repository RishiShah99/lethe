// SISO selective-scan CUDA kernels (cub::BlockScan, no tensor cores).
//
// The SSM recurrence h_t = a_t * h_{t-1} + b_t is a linear (first-order)
// recurrence, so it scans under the monoid op((a0,b0),(a1,b1)) =
// (a1*a0, a1*b0 + b1) with (a1,b1) the newer element. cub::BlockScan runs
// that scan across the L axis as an O(log L) warp-shuffle prefix scan — the
// engine the fast official Mamba-1 backward uses, and the reason this beats a
// serial-L walk. No tl.dot / mma / tensor cores anywhere: cub::BlockScan is
// pure __shfl, so this is #904-safe by construction.
//
// Inc 1 (this file's forward): correct first. One block per (batch, d_model);
// the block owns the L axis (chunked over blockDim, a running prefix carried
// in shared memory per state index n) and loops the d_state axis n serially.
// That serial-n loop is the N=128 bottleneck the 2-D decomposition (Inc 2)
// removes; here it is the simplest correct structure to de-risk the build,
// the scan primitive, and parity vs reference_forward_chunked_scan.
//
// All math is fp32. softplus and exp match the references (softplus identity
// above 20; plain expf, no fast-math approximation) so the verifier's
// tolerance and EXC-01 denormal masks line up with the Triton kernels.

#include <torch/extension.h>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

namespace {

struct SSMScanOp {
  // older = the earlier-in-time aggregate, newer = the later element.
  __device__ __forceinline__ float2 operator()(const float2 &older,
                                               const float2 &newer) const {
    return make_float2(newer.x * older.x, newer.x * older.y + newer.y);
  }
};

// Carries the scan prefix across L-chunks: returns the prefix to seed the
// current chunk (everything older than it), then folds the chunk's aggregate
// in for the next chunk. cub invokes operator() in thread 0 only.
struct ChunkPrefixOp {
  float2 running;
  __device__ __forceinline__ explicit ChunkPrefixOp(float2 r) : running(r) {}
  __device__ __forceinline__ float2 operator()(float2 block_aggregate) {
    const float2 prefix = running;
    running = SSMScanOp()(prefix, block_aggregate);
    return prefix;
  }
};

__device__ __forceinline__ float softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

template <int kThreads>
__global__ void forward_scan_kernel(const float *__restrict__ u,      // [B,L,D]
                                    const float *__restrict__ delta,  // [B,L,D]
                                    const float *__restrict__ A,      // [D,N]
                                    const float *__restrict__ Bmat,   // [B,L,N]
                                    const float *__restrict__ Cmat,   // [B,L,N]
                                    const float *__restrict__ Dskip,  // [D]
                                    float *__restrict__ y,            // [B,L,D]
                                    int L, int D, int N) {
  using BlockScan = cub::BlockScan<float2, kThreads>;
  __shared__ typename BlockScan::TempStorage temp;
  extern __shared__ float2 carry[];  // [N], the per-state running prefix

  const int b = blockIdx.x / D;
  const int d = blockIdx.x % D;
  const int tid = threadIdx.x;

  for (int n = tid; n < N; n += kThreads) carry[n] = make_float2(1.0f, 0.0f);
  __syncthreads();

  const float Dd = Dskip[d];
  const int nchunks = (L + kThreads - 1) / kThreads;

  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int t = chunk * kThreads + tid;
    const bool valid = t < L;
    const long ud = (static_cast<long>(b) * L + t) * D + d;
    const long bln = (static_cast<long>(b) * L + t) * N;
    const float u_t = valid ? u[ud] : 0.0f;
    const float dbar = valid ? softplus(delta[ud]) : 0.0f;
    float y_acc = 0.0f;

    for (int n = 0; n < N; ++n) {
      const float a_dn = A[static_cast<long>(d) * N + n];
      // Invalid (padding) lanes must scan as the identity (1,0) so they leave
      // the running prefix untouched; their y is never stored.
      const float a_bar = valid ? expf(dbar * a_dn) : 1.0f;
      const float b_tn = valid ? Bmat[bln + n] : 0.0f;
      const float b_val = valid ? dbar * b_tn * u_t : 0.0f;

      float2 out;
      ChunkPrefixOp prefix(carry[n]);
      BlockScan(temp).InclusiveScan(make_float2(a_bar, b_val), out, SSMScanOp(), prefix);
      const float c_tn = valid ? Cmat[bln + n] : 0.0f;
      y_acc += out.y * c_tn;
      if (tid == 0) carry[n] = prefix.running;
      __syncthreads();  // guard temp + carry[n] before the next n reuses them
    }

    if (valid) y[ud] = y_acc + Dd * u_t;
  }
}

// ---- Inc 2: the 2-D decomposition (the N=128 win) -------------------------
//
// Inc 1 loops d_state serially, so its cost scales with N — fine at N=16,
// the bottleneck at Mamba-3's N=128. Here the block is (32, W): the 32 lanes
// own a 32-step L-chunk (warp-shuffle scan, carry across chunks per state),
// and the W warps split the d_state axis (warp w owns n = w, w+W, ...). The
// y_t = sum_n h_t[n]*C_t[n] contraction becomes a cross-warp reduction over
// the W warps for each lane. Parallel over BOTH L and d_state — neither the
// Mamba-1 CUDA backward (serial d_state) nor the Triton kernels (serial L) do
// this; it is the genuine contribution. Still pure __shfl: #904-safe.

__device__ __forceinline__ float warp_inclusive_ssm(float a, float b, float2 &carry, int lane) {
  // Inclusive Kogge-Stone scan of (a,b) across the 32 lanes under the SSM
  // monoid, then fold the cross-chunk carry (older than the whole warp) in.
  // Returns h_t for this lane; updates carry to the running prefix after the
  // chunk (lane 31's full aggregate, broadcast).
  float va = a, vb = b;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    const float pa = __shfl_up_sync(0xffffffffu, va, off);
    const float pb = __shfl_up_sync(0xffffffffu, vb, off);
    if (lane >= off) {
      vb = va * pb + vb;
      va = va * pa;
    }
  }
  const float hc = va * carry.y + vb;  // SSMScanOp(carry, v).y
  const float ac = va * carry.x;       // SSMScanOp(carry, v).x
  carry.x = __shfl_sync(0xffffffffu, ac, 31);
  carry.y = __shfl_sync(0xffffffffu, hc, 31);
  return hc;
}

template <int kWarps>
__global__ void forward_scan_2d_kernel(const float *__restrict__ u,      // [B,L,D]
                                       const float *__restrict__ delta,  // [B,L,D]
                                       const float *__restrict__ A,      // [D,N]
                                       const float *__restrict__ Bmat,   // [B,L,N]
                                       const float *__restrict__ Cmat,   // [B,L,N]
                                       const float *__restrict__ Dskip,  // [D]
                                       float *__restrict__ y,            // [B,L,D]
                                       int L, int D, int N) {
  constexpr int kMaxN = 128;
  constexpr int kJ = (kMaxN + kWarps - 1) / kWarps;  // states per warp (compile cap)
  __shared__ float y_red[32 * kWarps];

  const int b = blockIdx.x / D;
  const int d = blockIdx.x % D;
  const int lane = threadIdx.x;
  const int warp = threadIdx.y;
  const float Dd = Dskip[d];

  float2 carry[kJ];
#pragma unroll
  for (int j = 0; j < kJ; ++j) carry[j] = make_float2(1.0f, 0.0f);

  const int nchunks = (L + 31) / 32;
  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int t = chunk * 32 + lane;
    const bool valid = t < L;
    const long ud = (static_cast<long>(b) * L + t) * D + d;
    const long bln = (static_cast<long>(b) * L + t) * N;
    const float u_t = valid ? u[ud] : 0.0f;
    const float dbar = valid ? softplus(delta[ud]) : 0.0f;

    float y_partial = 0.0f;
#pragma unroll
    for (int j = 0; j < kJ; ++j) {
      const int n = warp + j * kWarps;  // uniform across the warp -> shfl-safe
      if (n < N) {
        const float a_dn = A[static_cast<long>(d) * N + n];
        const float a_bar = valid ? expf(dbar * a_dn) : 1.0f;
        const float b_tn = valid ? Bmat[bln + n] : 0.0f;
        const float b_val = valid ? dbar * b_tn * u_t : 0.0f;
        const float h = warp_inclusive_ssm(a_bar, b_val, carry[j], lane);
        const float c_tn = valid ? Cmat[bln + n] : 0.0f;
        y_partial += h * c_tn;
      }
    }

    y_red[lane * kWarps + warp] = y_partial;
    __syncthreads();
    if (warp == 0) {
      float ys = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) ys += y_red[lane * kWarps + w];
      if (valid) y[ud] = ys + Dd * u_t;
    }
    __syncthreads();
  }
}

// ---- Inc 2b: the efficient forward (d-major layout, kNItems time-tiling) --
//
// Measurement (Inc 1/2) showed the N=128 cost is NOT d_state serialisation but
// (a) the [B,L,D] layout — a d-fixed L-scan reads u/delta strided by D — and
// (b) one timestep per thread (too many block-scans + syncs). This kernel is
// the official structure, owned line-by-line: inputs are d-major [B,D,L] /
// [B,N,L] (the launcher transposes), so loads coalesce; each thread owns
// kNItems consecutive timesteps via cub::BlockLoad (WARP_TRANSPOSE), does a
// thread-local serial scan, and a single block-scan over the per-thread
// aggregates carries the prefix — so the block-scan count drops by kNItems.
// d_state stays serial (the official does too, and it is not the bottleneck).
// Still no tl.dot / mma: cub scan is pure __shfl. #904-safe.

template <int kNThreads, int kNItems>
__global__ void forward_scan_tiled_kernel(const float *__restrict__ u,      // [B,D,L]
                                          const float *__restrict__ delta,  // [B,D,L]
                                          const float *__restrict__ A,      // [D,N]
                                          const float *__restrict__ Bmat,   // [B,N,L]
                                          const float *__restrict__ Cmat,   // [B,N,L]
                                          const float *__restrict__ Dskip,  // [D]
                                          float *__restrict__ y,            // [B,D,L]
                                          int L, int D, int N) {
  constexpr int kChunk = kNThreads * kNItems;
  constexpr int kMaxN = 128;
  using BlockLoadT = cub::BlockLoad<float, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<float, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockScanT = cub::BlockScan<float2, kNThreads>;
  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } smem;
  __shared__ float2 carry[kMaxN];

  const int b = blockIdx.x / D;
  const int d = blockIdx.x % D;
  const int tid = threadIdx.x;
  const float Dd = Dskip[d];

  const float *u_bd = u + (static_cast<long>(b) * D + d) * L;
  const float *delta_bd = delta + (static_cast<long>(b) * D + d) * L;
  float *y_bd = y + (static_cast<long>(b) * D + d) * L;
  const float *B_b = Bmat + static_cast<long>(b) * N * L;
  const float *C_b = Cmat + static_cast<long>(b) * N * L;

  for (int n = tid; n < N; n += kNThreads) carry[n] = make_float2(1.0f, 0.0f);
  __syncthreads();

  const int nchunks = (L + kChunk - 1) / kChunk;
  for (int chunk = 0; chunk < nchunks; ++chunk) {
    const int t0 = chunk * kChunk;
    const int valid_items = min(kChunk, L - t0);

    float u_it[kNItems];
    float dbar_it[kNItems];
    BlockLoadT(smem.load).Load(u_bd + t0, u_it, valid_items, 0.0f);
    __syncthreads();
    BlockLoadT(smem.load).Load(delta_bd + t0, dbar_it, valid_items, 0.0f);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kNItems; ++i) dbar_it[i] = softplus(dbar_it[i]);

    float y_acc[kNItems];
#pragma unroll
    for (int i = 0; i < kNItems; ++i) y_acc[i] = 0.0f;

    for (int n = 0; n < N; ++n) {
      const float a_dn = A[static_cast<long>(d) * N + n];
      float b_it[kNItems];
      float c_it[kNItems];
      BlockLoadT(smem.load).Load(B_b + static_cast<long>(n) * L + t0, b_it, valid_items, 0.0f);
      __syncthreads();
      BlockLoadT(smem.load).Load(C_b + static_cast<long>(n) * L + t0, c_it, valid_items, 0.0f);
      __syncthreads();

      float2 local[kNItems];
      float2 agg = make_float2(1.0f, 0.0f);
#pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        const int t = t0 + tid * kNItems + i;
        const bool v = t < L;
        const float a_bar = v ? expf(dbar_it[i] * a_dn) : 1.0f;
        const float b_val = v ? dbar_it[i] * b_it[i] * u_it[i] : 0.0f;
        agg = SSMScanOp()(agg, make_float2(a_bar, b_val));
        local[i] = agg;
      }

      float2 thread_prefix;
      ChunkPrefixOp pfx(carry[n]);
      BlockScanT(smem.scan).ExclusiveScan(agg, thread_prefix, SSMScanOp(), pfx);
      if (tid == 0) carry[n] = pfx.running;
      __syncthreads();

#pragma unroll
      for (int i = 0; i < kNItems; ++i) {
        const float2 hf = SSMScanOp()(thread_prefix, local[i]);
        y_acc[i] += hf.y * c_it[i];
      }
    }

    float y_it[kNItems];
#pragma unroll
    for (int i = 0; i < kNItems; ++i) y_it[i] = y_acc[i] + Dd * u_it[i];
    BlockStoreT(smem.store).Store(y_bd + t0, y_it, valid_items);
    __syncthreads();
  }
}

}  // namespace

torch::Tensor forward_scan(torch::Tensor u, torch::Tensor delta, torch::Tensor A,
                           torch::Tensor Bmat, torch::Tensor Cmat, torch::Tensor Dskip) {
  const int Bsz = u.size(0);
  const int L = u.size(1);
  const int D = u.size(2);
  const int N = A.size(1);
  auto y = torch::empty_like(u);

  constexpr int kThreads = 256;
  const dim3 grid(static_cast<unsigned>(Bsz) * static_cast<unsigned>(D));
  const size_t shmem = static_cast<size_t>(N) * sizeof(float2);
  forward_scan_kernel<kThreads><<<grid, kThreads, shmem>>>(
      u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
      Bmat.data_ptr<float>(), Cmat.data_ptr<float>(), Dskip.data_ptr<float>(),
      y.data_ptr<float>(), L, D, N);
  return y;
}

torch::Tensor forward_scan_2d(torch::Tensor u, torch::Tensor delta, torch::Tensor A,
                              torch::Tensor Bmat, torch::Tensor Cmat, torch::Tensor Dskip,
                              int64_t warps) {
  const int Bsz = u.size(0);
  const int L = u.size(1);
  const int D = u.size(2);
  const int N = A.size(1);
  TORCH_CHECK(N <= 128, "forward_scan_2d: n_state must be <= 128");
  auto y = torch::empty_like(u);

  const dim3 grid(static_cast<unsigned>(Bsz) * static_cast<unsigned>(D));
  auto launch = [&](auto warps_tag) {
    constexpr int W = decltype(warps_tag)::value;
    const dim3 block(32, W);
    const size_t shmem = static_cast<size_t>(32 * W) * sizeof(float);
    forward_scan_2d_kernel<W><<<grid, block, shmem>>>(
        u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
        Bmat.data_ptr<float>(), Cmat.data_ptr<float>(), Dskip.data_ptr<float>(),
        y.data_ptr<float>(), L, D, N);
  };
  switch (warps) {
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    case 32: launch(std::integral_constant<int, 32>{}); break;
    default: TORCH_CHECK(false, "forward_scan_2d: warps must be one of {4,8,16,32}");
  }
  return y;
}

// d-major inputs: u,delta,y [B,D,L]; B,C [B,N,L]; A [D,N]; Dskip [D].
torch::Tensor forward_scan_tiled(torch::Tensor u, torch::Tensor delta, torch::Tensor A,
                                 torch::Tensor Bmat, torch::Tensor Cmat, torch::Tensor Dskip,
                                 int64_t items) {
  const int Bsz = u.size(0);
  const int D = u.size(1);
  const int L = u.size(2);
  const int N = A.size(1);
  TORCH_CHECK(N <= 128, "forward_scan_tiled: n_state must be <= 128");
  auto y = torch::empty_like(u);

  constexpr int kNThreads = 128;
  const dim3 grid(static_cast<unsigned>(Bsz) * static_cast<unsigned>(D));
  auto launch = [&](auto items_tag) {
    constexpr int I = decltype(items_tag)::value;
    forward_scan_tiled_kernel<kNThreads, I><<<grid, kNThreads>>>(
        u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(),
        Bmat.data_ptr<float>(), Cmat.data_ptr<float>(), Dskip.data_ptr<float>(),
        y.data_ptr<float>(), L, D, N);
  };
  switch (items) {
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    case 16: launch(std::integral_constant<int, 16>{}); break;
    default: TORCH_CHECK(false, "forward_scan_tiled: items must be one of {4,8,16}");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_scan", &forward_scan, "SISO selective scan forward (CUDA cub::BlockScan)");
  m.def("forward_scan_2d", &forward_scan_2d, "SISO forward, 2-D (warps split d_state)");
  m.def("forward_scan_tiled", &forward_scan_tiled, "SISO forward, d-major + kNItems tiling");
}
