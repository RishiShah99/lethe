"""Diagnostic 2: does ``cute.compile`` kill the ~190 ms/call ``@cute.jit`` dispatch?

probe_gemm_overhead.py showed each ``_gemm_aa`` spends ~190 ms in host dispatch (re-tracing the
host body every call) to launch a 5 us kernel. Every call is the SAME (128,128)@(64,128) shape, so
a single pre-compiled artifact should serve all of them. This compiles ``_gemm_host`` once and times
the compiled callable. If per-call drops to ~ms/us, the host-orchestrated loop gets fast with NO new
kernel — just cache the compiled GEMM.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/probe_gemm_compiled.py
"""

import time

import torch

from lethe.kernels.cute.gdn2_bwd_dhu import _gemm_host, _mark, is_available

if not is_available():
    raise SystemExit("not an sm_100 box")

import cutlass.cute as cute

dev = torch.device("cuda")
f16 = torch.float16
a = torch.randn(128, 128, device=dev, dtype=f16)
b = torch.randn(64, 128, device=dev, dtype=f16)
out = torch.zeros(128, 64, device=dev, dtype=f16)

print("cute has compile:", hasattr(cute, "compile"))
t0 = time.perf_counter()
compiled = cute.compile(_gemm_host, _mark(a), _mark(b), _mark(out))
torch.cuda.synchronize()
print(f"cute.compile once: {(time.perf_counter() - t0) * 1e3:.1f} ms ; type={type(compiled)}")

N = 20
print(f"\ncompiled callable, {N} calls (fresh out each, sync each):")
torch.cuda.synchronize()
for i in range(N):
    out_i = torch.zeros(128, 64, device=dev, dtype=f16)
    t0 = time.perf_counter()
    compiled(_mark(a), _mark(b), _mark(out_i))
    torch.cuda.synchronize()
    print(f"  call {i:2d}: {(time.perf_counter() - t0) * 1e3:9.4f} ms")

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    out_i = torch.zeros(128, 64, device=dev, dtype=f16)
    compiled(_mark(a), _mark(b), _mark(out_i))
host = (time.perf_counter() - t0) * 1e3
torch.cuda.synchronize()
total = (time.perf_counter() - t0) * 1e3
print(
    f"\ncompiled {N} calls: host(no per-call sync)={host:.2f} ms ({host / N:.4f} ms/call) "
    f"total={total:.2f} ms ({total / N:.4f} ms/call)"
)

# correctness: compiled GEMM vs torch reference (out = a @ b^T)
out_c = torch.zeros(128, 64, device=dev, dtype=f16)
compiled(_mark(a), _mark(b), _mark(out_c))
torch.cuda.synchronize()
ref = (a.float() @ b.float().t()).to(f16)
rel = ((out_c.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-9)).item()
print(f"compiled GEMM scale_rel vs torch a@b^T: {rel:.3e}")
