"""KernelBench's OWN correctness check, replicated for the differential cross-val.

The 62.1% finding says our contract battery flags kernels a source system's
harness accepted. The obvious rebuttal — "your gates are stricter than the
field's, of course foreigners fail" — is answered by running the SAME flagged
kernels through the FIELD's own check and quantifying the load-bearing cell:
{external PASS} x {our FAIL}. That count is "the field's harness accepts broken
kernels," in the field's own terms.

Two variants (both implemented; the gap between them is part of the story):

  paper_era  — what the corpora were ACCEPTED under. KernelBench
      src/eval.py::run_and_check_correctness @48642c5c ~L612:
      seed the base RNG (42), draw N=5 per-trial seeds, and for each trial
      reseed + call get_inputs() at the task's FIXED shapes, run ref and
      candidate, require torch.allclose(atol=1e-2, rtol=1e-2) AND matching
      shape. All 5 trials must pass. NO dtype guard, NO adversarial / zeros /
      denormal / NaN inputs, no determinism/aliasing/resource checks.

  hardened   — current-main KernelBench's tighter variant: per-dtype tolerance
      (fp32 -> 1e-4, fp16/bf16 -> 1e-2). If our findings survive even the
      field's HARDENED allclose, that is the stronger statement. (KernelBench's
      10x-excessive-speedup flag is a TIMING signal, orthogonal to correctness;
      it is not evaluated here — this is a correctness cross-val.)

This is deliberately a WEAKER check than verifier/audit_harness.py: same model
loading convention, but only fixed-shape random-input allclose. Run it in the
sandbox (candidate kernels crash CUDA contexts): the driver dispatches each row
through lethe.verifier.sandbox.run_in_subprocess pointed at this file.

Usage (box, after the audit shards exist):
    uv run python scratch/external_kernelbench_check.py \
        scratch/audit_manifest_drkernel.jsonl.gz results/crossval_rows.jsonl \
        --device cuda --shard 0 --num-shards 8
    # then join with the cached audit rows into the 2x2:
    uv run python scratch/external_kernelbench_check.py --join \
        results/crossval_rows.jsonl "audit_out/results_shard*.jsonl" \
        --json results/crossval_kernelbench.json

Local logic check (CPU, no corpus):
    uv run python scratch/external_kernelbench_check.py --self-test
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import gzip
import importlib.util
import json
import os
import sys
import tempfile
from typing import Any

N_TRIALS = 5
BASE_SEED = 42
PAPER_TOL = 1e-2  # atol == rtol, KernelBench paper-era
HARDENED_TOL = {"fp32": 1e-4, "half": 1e-2}


def _exec_source(source: str, required: str) -> dict[str, Any]:
    # @triton.jit needs a real source file (inspect-based tracing) and the lazy
    # PTX compile re-reads it during the run — mirror audit_harness._exec_source.
    fd, path = tempfile.mkstemp(suffix=".py", prefix=f"xkb_{required.lower()}_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(source)
    name = f"_xkb_{required.lower()}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    ns = vars(mod)
    if required not in ns:
        raise KeyError(f"source defines no {required}")
    return dict(ns)


def _tensors(out: Any) -> list[Any]:
    import torch

    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, tuple | list):
        return [x for x in out if isinstance(x, torch.Tensor)]
    return []


def _compare(out_ref: Any, out_cand: Any, atol: float, rtol: float) -> tuple[bool, str]:
    import torch

    ts_ref, ts_cand = _tensors(out_ref), _tensors(out_cand)
    if not ts_ref:
        return False, "reference produced no tensor"
    if len(ts_ref) != len(ts_cand):
        return False, f"output count {len(ts_cand)} != ref {len(ts_ref)}"
    for a, b in zip(ts_ref, ts_cand, strict=True):
        if tuple(a.shape) != tuple(b.shape):
            return False, f"shape {tuple(b.shape)} != ref {tuple(a.shape)}"
        # KernelBench compares in fp32; equal_nan matches its allclose default (False)
        if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
            return False, "allclose failed"
    return True, ""


def _run_trials(
    model: Any, model_new: Any, get_inputs: Any, device: str, atol: float, rtol: float
) -> tuple[int, str]:
    import torch

    torch.manual_seed(BASE_SEED)
    seeds = torch.randint(0, 2**31 - 1, (N_TRIALS,)).tolist()
    passed = 0
    last = ""
    for s in seeds:
        torch.manual_seed(s)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in list(get_inputs())]
        try:
            with torch.no_grad():
                out_ref = model(*inputs)
                out_cand = model_new(*inputs)
        except Exception as exc:  # candidate/ref runtime error == not correct (KernelBench)
            return passed, f"{type(exc).__name__}: {exc}"[:200]
        ok, reason = _compare(out_ref, out_cand, atol, rtol)
        if ok:
            passed += 1
        else:
            last = reason
    return passed, last


def external_check_worker(
    ref_source: str, cand_source: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Run the KernelBench paper-era + hardened checks on one (ref, cand) pair.

    Returns primitive-only dict (torch.save/weights_only safe across the sandbox).
    """
    import torch

    device = config.get("device", "cuda")
    out: dict[str, Any] = {"paper_era": None, "hardened": None, "out_dtype": None}

    try:
        ref_ns = _exec_source(ref_source, "Model")
        get_inputs = ref_ns["get_inputs"]
        get_init = ref_ns.get("get_init_inputs")
        torch.manual_seed(BASE_SEED)
        init_args = list(get_init()) if callable(get_init) else []
        init_args = [x.to(device) if isinstance(x, torch.Tensor) else x for x in init_args]
        torch.manual_seed(BASE_SEED)
        model = ref_ns["Model"](*init_args).to(device).eval()
    except Exception as exc:
        out["status"] = "ref_broken"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    try:
        cand_ns = _exec_source(cand_source, "ModelNew")
        torch.manual_seed(BASE_SEED)
        model_new = cand_ns["ModelNew"](*init_args).to(device).eval()
    except Exception as exc:
        out["status"] = "cand_load_fail"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        # a load failure is "not correct" under KernelBench too
        out["paper_era"] = {"correct": False, "trials_passed": 0, "reason": "load_fail"}
        out["hardened"] = {"correct": False, "trials_passed": 0, "reason": "load_fail"}
        return out

    # dtype of a representative reference output drives the hardened tolerance
    try:
        torch.manual_seed(BASE_SEED)
        probe_inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x for x in list(get_inputs())
        ]
        with torch.no_grad():
            probe = model(*probe_inputs)
        ts = _tensors(probe)
        out["out_dtype"] = str(ts[0].dtype) if ts else None
        hardened_tol = (
            HARDENED_TOL["fp32"] if (ts and ts[0].dtype == torch.float32) else HARDENED_TOL["half"]
        )
    except Exception:
        hardened_tol = HARDENED_TOL["fp32"]

    p_pass, p_reason = _run_trials(model, model_new, get_inputs, device, PAPER_TOL, PAPER_TOL)
    out["paper_era"] = {
        "correct": p_pass == N_TRIALS,
        "trials_passed": p_pass,
        "reason": p_reason,
        "atol": PAPER_TOL,
    }
    h_pass, h_reason = _run_trials(model, model_new, get_inputs, device, hardened_tol, hardened_tol)
    out["hardened"] = {
        "correct": h_pass == N_TRIALS,
        "trials_passed": h_pass,
        "reason": h_reason,
        "atol": hardened_tol,
    }
    out["status"] = "checked"
    return out


# --------------------------------------------------------------------------- #
# driver + join
# --------------------------------------------------------------------------- #
SSM_ADJACENT_CLASSES = "matmul,attention,softmax,scan,norm,conv,reduction"


def _load_manifest(path: str, classes: set[str]) -> list[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["op_class"] in classes:
                rows.append(row)
    return rows


def run_driver(args: argparse.Namespace) -> None:
    from lethe.verifier.sandbox import run_in_subprocess

    classes = set(args.classes.split(","))
    rows = _load_manifest(args.manifest, classes)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    if args.only_accepted:
        rows = [r for r in rows if (r.get("final_speedup") or 0) > 0]
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                with contextlib.suppress(json.JSONDecodeError, KeyError):
                    done.add(json.loads(line)["id"])

    here = os.path.abspath(__file__)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(
        f"[shard {args.shard}] {len(rows)} rows, {len(done & {r['id'] for r in rows})} done",
        flush=True,
    )
    with open(args.out, "a", encoding="utf-8") as fout:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            res = run_in_subprocess(
                here,
                "external_check_worker",
                (row["ref"], row["cand"], {"device": args.device}),
                timeout_s=args.timeout,
                memory_limit_mb=0,
            )
            if res.success and isinstance(res.output, dict):
                payload: dict[str, Any] = dict(res.output)
            else:
                payload = {
                    "status": f"sandbox_{res.error_class.name.lower()}",
                    "error": res.stderr[-300:],
                    "paper_era": {"correct": False, "trials_passed": 0, "reason": "sandbox"},
                }
            payload["id"] = row["id"]
            payload["op_class"] = row["op_class"]
            payload["final_speedup"] = row.get("final_speedup")
            fout.write(json.dumps(payload) + "\n")
            fout.flush()
            if (i + 1) % 10 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(rows)}", flush=True)
    print(f"[shard {args.shard}] DONE", flush=True)


def _our_verdict(row: dict[str, Any]) -> str:
    """Reproduce audit_aggregate's finding logic for one row: 'fail' or 'pass'."""
    excluded = {"ref_broken", "not_auditable"}
    if row["status"] in excluded:
        return "excluded"
    # artifact?
    if row["status"] == "sandbox_other" and "unpickle error" in row.get("error", ""):
        return "artifact"
    if row["status"] == "cand_native_fail":
        err = row.get("error", "")
        if err.startswith(
            ("CompilationError", "UnsupportedLanguageConstruct", "OutOfResources", "PTXASError")
        ):
            return "artifact"
    if row["status"] != "gated":
        return "fail"  # pre-gate finding
    gates = row.get("gates", {})
    if any("CUDA error" in g.get("reason", "") for g in gates.values()):
        return "fail"
    if any(g.get("status") == "fail" for g in gates.values()):
        return "fail"
    return "pass"


def run_join(args: argparse.Namespace) -> None:
    # external rows
    ext: dict[str, dict[str, Any]] = {}
    with open(args.external, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ext[r["id"]] = r
    # audit rows (our verdict)
    ours: dict[str, dict[str, Any]] = {}
    for pat in args.audit:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ours.setdefault(r["id"], r)

    def blank() -> dict[str, int]:
        return {
            "ext_pass_our_pass": 0,
            "ext_pass_our_fail": 0,
            "ext_fail_our_pass": 0,
            "ext_fail_our_fail": 0,
        }

    cells_paper = blank()
    cells_hard = blank()
    ext_pass_our_fail_ids: list[str] = []
    by_gate_in_cell: dict[str, int] = {}
    money_tol_free = 0
    joined = 0
    for cid, e in ext.items():
        o = ours.get(cid)
        if o is None:
            continue
        verdict = _our_verdict(o)
        if verdict in ("excluded", "artifact"):
            continue
        if (o.get("final_speedup") or 0) <= 0:
            continue  # accepted-only population
        our_ok = verdict == "pass"
        joined += 1
        for variant, cells in (("paper_era", cells_paper), ("hardened", cells_hard)):
            ext_ok = bool((e.get(variant) or {}).get("correct"))
            cells[f"ext_{'pass' if ext_ok else 'fail'}_our_{'pass' if our_ok else 'fail'}"] += 1
        # money cell decomposition uses the paper-era external verdict
        if bool((e.get("paper_era") or {}).get("correct")) and not our_ok:
            ext_pass_our_fail_ids.append(cid)
            gates = o.get("gates", {})
            tol_free = False
            for gate, g in gates.items():
                if g.get("status") == "fail":
                    by_gate_in_cell[gate] = by_gate_in_cell.get(gate, 0) + 1
                    if gate in ("EXC-01", "ORD-02"):
                        tol_free = True
                    if gate == "CMP-03" and "exception" in g.get("reason", ""):
                        tol_free = True
            if (
                o["status"] != "gated"
                or o.get("output_aliasing")
                or any("CUDA error" in g.get("reason", "") for g in gates.values())
            ):
                by_gate_in_cell["pre_gate_or_crash_or_alias"] = (
                    by_gate_in_cell.get("pre_gate_or_crash_or_alias", 0) + 1
                )
                tol_free = True
            money_tol_free += int(tol_free)

    result = {
        "corpus": "hkust-nlp/drkernel-coldstart-8k",
        "population": "accepted_only (final_speedup>0)",
        "joined_rows": joined,
        "cells_paper_era": cells_paper,
        "cells_hardened": cells_hard,
        "money_cell_external_PASS_our_FAIL": cells_paper["ext_pass_our_fail"],
        "money_cell_rate": round(cells_paper["ext_pass_our_fail"] / max(1, joined), 4),
        "money_cell_tolerance_free": money_tol_free,
        "money_cell_by_our_gate": dict(sorted(by_gate_in_cell.items(), key=lambda kv: -kv[1])),
        "worked_example_ids": ext_pass_our_fail_ids[:20],
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


def self_test() -> None:
    """CPU logic check: a correct pair passes both; a wrong pair fails both."""
    import torch  # noqa: F401

    ref_src = (
        "import torch\n"
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, n):\n"
        "        super().__init__()\n"
        "        self.lin = nn.Linear(n, n)\n"
        "    def forward(self, x):\n"
        "        return torch.relu(self.lin(x))\n"
        "def get_init_inputs():\n"
        "    return [8]\n"
        "def get_inputs():\n"
        "    return [torch.randn(4, 8)]\n"
    )
    good = ref_src.replace("class Model", "class ModelNew")
    bad = good.replace("torch.relu(self.lin(x))", "torch.sigmoid(self.lin(x))")

    g = external_check_worker(ref_src, good, {"device": "cpu"})
    b = external_check_worker(ref_src, bad, {"device": "cpu"})
    print("good pair:", json.dumps(g))
    print("bad  pair:", json.dumps(b))
    assert g["paper_era"]["correct"] is True, "correct kernel must pass paper-era"
    assert g["hardened"]["correct"] is True, "correct kernel must pass hardened"
    assert b["paper_era"]["correct"] is False, "wrong kernel must fail paper-era"
    print("SELF-TEST PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--join", action="store_true")
    ap.add_argument(
        "manifest", nargs="?", help="manifest (.jsonl[.gz]) OR external rows for --join"
    )
    ap.add_argument("out", nargs="?", help="output rows OR audit-shard glob for --join")
    ap.add_argument("audit", nargs="*", help="(--join) audit shard globs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--classes", default=SSM_ADJACENT_CLASSES)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--only-accepted", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if args.join:
        # reinterpret positionals: manifest=external rows, out+audit = shard globs
        args.external = args.manifest
        args.audit = ([args.out] if args.out else []) + args.audit
        run_join(args)
        return
    if not args.manifest or not args.out:
        ap.error("manifest and out required")
    run_driver(args)


if __name__ == "__main__":
    main()
