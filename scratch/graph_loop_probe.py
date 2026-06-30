"""#48 de-risk — capture a MULTI-launch reverse loop (the K#1 shape) into ONE graph.

#47 proved a single ``_gemm_batched`` is capture-safe. The full cw backward is a reverse
chunk loop carrying a resident ``b_dh``, with a tcgen05 GEMM + torch glue (decay scale,
add) + intermediate allocations per chunk. This probe mimics that structure to confirm the
three remaining #48 unknowns capture + replay correctly in ONE graph:
  (a) MANY launches recorded in one capture,
  (b) torch glue ops (mul/add) and ``torch.zeros`` allocations inside the capture region,
  (c) a cross-iteration carry (``b_dh``) threaded through the captured launches.

GO = capture didn't throw AND replay (fresh inputs via copy_) matches the eager loop. GO
means #48's full-backward graphing is low-risk. A hard SIGSEGV = exit 139, no JSON.

  PYTHONPATH=src:. ~/cuteenv/bin/python scratch/graph_loop_probe.py
"""

import json
import sys

import torch
from scratch.gdn2_bwd_dhu import is_available

RESULT: dict[str, object] = {"probe": "graph_loop_capture", "available": is_available()}


def _emit_and_exit(go: bool, code: int = 0) -> None:
    RESULT["GO"] = go
    print("GO=" + str(go))
    print(json.dumps(RESULT, indent=2))
    with open("results/graph_loop_probe.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    sys.exit(code)


if not is_available():
    RESULT["reason"] = "not an sm_100 box"
    _emit_and_exit(False)

from scratch.gdn2_bwd_dhu import _gemm_batched  # noqa: E402 — guarded; defined only on-box

dev = torch.device("cuda")
f16 = torch.float16
Z, NT = 8, 4  # n_bh batch, reverse chunks
torch.manual_seed(0)


def gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a.float(), b.float().transpose(-1, -2)).to(f16)


def eager_loop(a_all: torch.Tensor, b_all: torch.Tensor, decay: torch.Tensor) -> torch.Tensor:
    """Reverse loop with a resident b_dh carry, mirroring K#1's structure (torch GEMM ref)."""
    b_dh = torch.zeros(Z, 128, 64, device=dev, dtype=f16)
    dh = torch.zeros(NT, Z, 128, 64, device=dev, dtype=f16)
    for it in range(NT - 1, -1, -1):
        dh[it] = b_dh
        g = gemm_ref(a_all[it], b_all[it])
        b_dh = (decay[it] * b_dh.float() + g.float()).to(f16)
    return dh


# Static buffers — reused across replays (copy_ inputs in place; never reassign).
a_s = torch.randn(NT, Z, 128, 128, device=dev, dtype=f16)
b_s = torch.randn(NT, Z, 64, 128, device=dev, dtype=f16)
decay_s = torch.rand(NT, Z, 1, 1, device=dev, dtype=torch.float32) * 0.5 + 0.5
b_dh_s = torch.zeros(Z, 128, 64, device=dev, dtype=f16)
dh_s = torch.zeros(NT, Z, 128, 64, device=dev, dtype=f16)


def captured_loop() -> None:
    """Same loop, writing into the static dh_s, using the cute GEMM (one launch per chunk)."""
    b_dh_s.zero_()
    b_dh = b_dh_s
    for it in range(NT - 1, -1, -1):
        dh_s[it].copy_(b_dh)
        g = torch.zeros(Z, 128, 64, device=dev, dtype=f16)  # alloc inside capture (graph pool)
        _gemm_batched(a_s[it].contiguous(), b_s[it].contiguous(), g)
        b_dh = (decay_s[it] * b_dh.float() + g.float()).to(f16)


# Warm up OUTSIDE capture (cute.compile + cache the executable for batch Z).
captured_loop()
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
try:
    with torch.cuda.graph(g):
        captured_loop()
    RESULT["capture_ok"] = True
except Exception as e:
    RESULT["capture_ok"] = False
    RESULT["capture_err"] = repr(e)[:400]
    RESULT["reason"] = "loop capture threw"
    _emit_and_exit(False)

# Replay with fresh inputs; grade dh_s vs the eager loop.
N = 5
worst = 0.0
for _ in range(N):
    a_new = torch.randn(NT, Z, 128, 128, device=dev, dtype=f16)
    b_new = torch.randn(NT, Z, 64, 128, device=dev, dtype=f16)
    d_new = torch.rand(NT, Z, 1, 1, device=dev, dtype=torch.float32) * 0.5 + 0.5
    a_s.copy_(a_new)
    b_s.copy_(b_new)
    decay_s.copy_(d_new)
    g.replay()
    torch.cuda.synchronize()
    ref = eager_loop(a_new, b_new, d_new)
    rel = (
        (dh_s.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-9)
    ).item()
    worst = max(worst, rel)
RESULT["replay_n"] = N
RESULT["replay_worst_rel"] = worst
_emit_and_exit(bool(RESULT["capture_ok"]) and worst < 2e-2)
