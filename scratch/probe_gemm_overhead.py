"""Diagnostic: per-launch cost of the proven (128,64,128) tcgen05 GEMM.

Lever A (dropping the per-GEMM ``cuda.synchronize``) was speed-neutral, so the ~146 ms/launch
the bench implies is host/DSL dispatch, not device sync. This isolates it: time identical
back-to-back ``_gemm_aa`` calls. Call 0 = JIT compile; calls 1+ = steady state. If steady-state
per-call stays ~100 ms it is pure dispatch/marshalling (→ B batching + D fusion are the fix); if
it drops to ~ms the first call was the only compile (→ caching already works, only launch count
matters). Also reports host-only dispatch time (no per-call sync) vs device time.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/probe_gemm_overhead.py
"""

import time

import torch

from scratch.gdn2_bwd_dhu import is_available

if not is_available():
    raise SystemExit("not an sm_100 box")

from scratch.gdn2_bwd_dhu import _gemm_aa  # noqa: E402

dev = torch.device("cuda")
f16 = torch.float16
a = torch.randn(128, 128, device=dev, dtype=f16)
b = torch.randn(64, 128, device=dev, dtype=f16)
N = 20

print(f"per-call (host+device, sync each): {N} identical (128,128)@(64,128)^T GEMMs")
torch.cuda.synchronize()
for i in range(N):
    out = torch.zeros(128, 64, device=dev, dtype=f16)
    t0 = time.perf_counter()
    _gemm_aa(a, b, out)
    torch.cuda.synchronize()
    print(f"  call {i:2d}: {(time.perf_counter() - t0) * 1e3:9.2f} ms")

# pure host dispatch: queue N launches, no per-call sync, one sync at the end.
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(N):
    out = torch.zeros(128, 64, device=dev, dtype=f16)
    _gemm_aa(a, b, out)
host_ms = (time.perf_counter() - t0) * 1e3
torch.cuda.synchronize()
total_ms = (time.perf_counter() - t0) * 1e3
print(
    f"\n{N} queued launches: host-dispatch(no per-call sync)={host_ms:8.1f} ms "
    f"({host_ms / N:6.2f} ms/call); total inc. final sync={total_ms:8.1f} ms "
    f"({total_ms / N:6.2f} ms/call)"
)
print(f"device-only (total-host) ~= {total_ms - host_ms:.1f} ms for {N} GEMMs")
