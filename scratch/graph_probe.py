"""#47 — CUDA-graph capture-safety probe (make-or-break for the graph plan).

The cw backward for a fixed shape is an IDENTICAL launch sequence every call, so a
``torch.cuda.graph`` replay would collapse ALL host dispatch + the per-chunk Python loop
+ torch glue into ONE graph launch — leaving only device time. That ONLY works if a
``cute.compile``'d tcgen05 launch is capture-safe: no per-launch ``cudaMalloc`` /
``synchronize`` / device-property query (any of those throws "operation not permitted
during stream capture").

This warms up one ``_gemm_batched`` (so ``cute.compile`` + the executable cache happen
OUTSIDE capture), captures a replay of it into a CUDA graph, replays N times feeding fresh
inputs via ``copy_`` into the static buffers, and grades each replay vs an eager ``bmm``.
GO = capture didn't throw AND every replay matches eager. GO opens lever #48 (graph the
whole cw backward); NO-GO falls back to #49/#50 (incremental launch-count cuts).

A hard host SIGSEGV during capture/replay = exit 139, NO JSON. A caught CUDA/python error
= exit 0 + JSON with GO=False. The run script greps for ``GO=`` / ``EXIT_``.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_probe.py
"""

import json
import sys
import time

import torch
from scratch.gdn2_bwd_dhu import is_available

RESULT: dict[str, object] = {"probe": "graph_capture_safety", "available": is_available()}


def _emit_and_exit(go: bool, code: int = 0) -> None:
    RESULT["GO"] = go
    print("GO=" + str(go))
    print(json.dumps(RESULT, indent=2))
    with open("results/graph_probe.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    sys.exit(code)


if not is_available():
    RESULT["reason"] = "not an sm_100 box"
    _emit_and_exit(False)

from scratch.gdn2_bwd_dhu import _gemm_batched  # noqa: E402 — guarded; defined only on-box

dev = torch.device("cuda")
f16 = torch.float16
Z = 8  # representative batch (b*hv); the GEMM is the fixed M/N/K = 128/64/128 config
torch.manual_seed(0)


def ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.float(), b.float().transpose(-1, -2)).to(f16)


def rel(o: torch.Tensor, r: torch.Tensor) -> float:
    return ((o.float() - r.float()).abs().max() / r.float().abs().max().clamp_min(1e-9)).item()


# Static buffers — contiguous, batch-first, reused across every replay (copy_ in place;
# never reassign, so the pointers baked into the captured launch stay valid).
a_s = torch.randn(Z, 128, 128, device=dev, dtype=f16)
b_s = torch.randn(Z, 64, 128, device=dev, dtype=f16)
o_s = torch.zeros(Z, 128, 64, device=dev, dtype=f16)

# Warm up OUTSIDE capture: triggers cute.compile + populates the _gemm_b_cache[Z] entry,
# so capture hits the cached executable (no compile, no alloc) on the captured launch.
_gemm_batched(a_s, b_s, o_s)
torch.cuda.synchronize()
RESULT["warmup_rel"] = rel(o_s, ref(a_s, b_s))

# Attempt capture. A per-launch alloc/sync/device-query inside the cutlass-dsl launch path
# throws here ("operation not permitted during stream capture").
g = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
try:
    with torch.cuda.graph(g):
        _gemm_batched(a_s, b_s, o_s)
    RESULT["capture_ok"] = True
except Exception as e:
    RESULT["capture_ok"] = False
    RESULT["capture_err"] = repr(e)[:400]
    RESULT["reason"] = "capture threw (cutlass-dsl launch not capture-safe)"
    _emit_and_exit(False)

# Replay with fresh inputs, grade each vs eager. Proves the graph re-reads the static
# buffers and the tcgen05 GEMM stays correct under replay.
N = 10
worst = 0.0
for _ in range(N):
    a_new = torch.randn(Z, 128, 128, device=dev, dtype=f16)
    b_new = torch.randn(Z, 64, 128, device=dev, dtype=f16)
    a_s.copy_(a_new)
    b_s.copy_(b_new)
    g.replay()
    torch.cuda.synchronize()
    worst = max(worst, rel(o_s, ref(a_new, b_new)))
RESULT["replay_n"] = N
RESULT["replay_worst_rel"] = worst

# Timing signal: graph replay vs the eager call, same iteration count. Informative only;
# the GO gate is correctness + capture-safety, not this number.
ITERS = 50
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    _gemm_batched(a_s, b_s, o_s)
torch.cuda.synchronize()
eager_ms = (time.perf_counter() - t0) * 1e3 / ITERS

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(ITERS):
    g.replay()
torch.cuda.synchronize()
replay_ms = (time.perf_counter() - t0) * 1e3 / ITERS
RESULT["eager_ms_per_call"] = eager_ms
RESULT["replay_ms_per_call"] = replay_ms
RESULT["replay_speedup"] = eager_ms / replay_ms if replay_ms > 0 else 0.0

_emit_and_exit(bool(RESULT["capture_ok"]) and worst < 2e-2)
