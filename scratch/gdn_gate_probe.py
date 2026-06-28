"""GDN winnability gate — Part A.1 (compile floor) + Part B (determinism / ORD-02).

Run on a B200 (sm_100). Tests the two cheapest, highest-signal facts the
combined-thesis investigation flagged as UNMEASURED:

  A.1  Does fla's chunk_gated_delta_rule BACKWARD compile + run on Blackwell?
       - FAILS  -> supports the "first open correct Blackwell-native backward" floor.
       - RUNS   -> that floor claim does NOT hold; reconsider.
  B    Is the fla GDN backward deterministic? (fla #889 — atomic dg accumulation)
       - nonzero run-to-run grad diff on identical inputs -> a real ORD-02 finding.

No kernel is written here. This only measures the existing baseline.
Every step is wrapped so partial failure still yields a usable log.
"""

import os
import sys
import traceback

print("FLA_USE_TMA =", os.environ.get("FLA_USE_TMA", "(unset/default)"), flush=True)


def section(t: str) -> None:
    print(f"\n===== {t} =====", flush=True)


def show_exc() -> str:
    tb = traceback.format_exc()
    print(tb, flush=True)
    return tb


section("ENV")
print("python", sys.version.replace("\n", " "), flush=True)
torch = None
try:
    import torch  # noqa: F811

    print("torch", torch.__version__, "| cuda", torch.version.cuda, flush=True)
    print("cuda_available", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0), flush=True)
        print("capability", torch.cuda.get_device_capability(0), flush=True)
except Exception:
    show_exc()

try:
    import triton

    print("triton", triton.__version__, flush=True)
except Exception:
    show_exc()

section("FLA IMPORT")
chunk_fn = None
try:
    import fla

    print("fla", getattr(fla, "__version__", "unknown"), flush=True)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_fn

    print("imported chunk_gated_delta_rule OK", flush=True)
    try:
        import inspect

        print("signature:", inspect.signature(chunk_fn), flush=True)
    except Exception:
        pass
except Exception:
    show_exc()


def make_inputs(B=1, T=8192, H=16, D=128, seed=0):
    g_dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=g_dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, T, H, D, device=g_dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, T, H, D, device=g_dev, dtype=dt, requires_grad=True)
    # gate g: log-forget-gate, fp32, per (B,T,H)
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, device=g_dev, dtype=torch.float32)
    ).requires_grad_(True)
    beta = torch.rand(B, T, H, device=g_dev, dtype=dt).requires_grad_(True)
    return q, k, v, g, beta


def run_once(seed=0):
    q, k, v, g, beta = make_inputs(seed=seed)
    out = chunk_fn(q, k, v, g, beta)
    o = out[0] if isinstance(out, (tuple, list)) else out
    loss = o.float().square().mean()
    loss.backward()
    return {
        "dq": q.grad.detach().float().clone(),
        "dk": k.grad.detach().float().clone(),
        "dv": v.grad.detach().float().clone(),
        "dg": g.grad.detach().float().clone(),
        "dbeta": beta.grad.detach().float().clone(),
        "o": o.detach().float().clone(),
    }


section("PART A.1 — fla GDN BACKWARD COMPILE/RUN ON B200")
a1_ok = False
if chunk_fn is not None and torch is not None and torch.cuda.is_available():
    try:
        r = run_once(seed=0)
        a1_ok = True
        print("FWD+BWD ran. ||dq||=%.4e ||dg||=%.4e" % (
            r["dq"].norm().item(), r["dg"].norm().item()), flush=True)
        print("RESULT_A1=COMPILES  -> first-correct floor does NOT hold; reconsider", flush=True)
    except Exception:
        show_exc()
        print("RESULT_A1=FAILS  -> supports 'first open correct Blackwell-native backward' floor", flush=True)
else:
    print("RESULT_A1=SKIPPED (import or cuda unavailable)", flush=True)


section("PART B — DETERMINISM / ORD-02 (fla #889 class)")
if a1_ok:
    try:
        runs = [run_once(seed=0) for _ in range(3)]
        worst = {}
        for key in ("dq", "dk", "dv", "dg", "dbeta", "o"):
            d01 = (runs[0][key] - runs[1][key]).abs().max().item()
            d02 = (runs[0][key] - runs[2][key]).abs().max().item()
            worst[key] = max(d01, d02)
        for key, val in worst.items():
            print(f"max run-to-run |Δ| {key}: {val:.3e}", flush=True)
        nondet = {k: v for k, v in worst.items() if v > 0.0}
        if nondet:
            print("RESULT_B=NONDETERMINISTIC  -> ORD-02 finding:", list(nondet), flush=True)
        else:
            print("RESULT_B=DETERMINISTIC  -> no ORD-02 finding on these inputs", flush=True)
    except Exception:
        show_exc()
        print("RESULT_B=ERROR", flush=True)
else:
    print("RESULT_B=SKIPPED (A.1 did not run a successful backward)", flush=True)

print("\n===== PROBE DONE =====", flush=True)
