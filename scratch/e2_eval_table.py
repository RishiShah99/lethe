"""E2 eval table: learned config-RL/edit-RL policy vs the boundary-sweep oracle.

Reads a config-RL (or edit-RL) checkpoint dir of per-level
``<op>_B<b>L<l>W<w>/`` subdirs and, for each level, derives what the policy
*converged to* — the dominant ``scan_mode`` across the late-step, NON-seeded
emissions (the seeds are forced exploration, not policy choice) — and its best
measured speedup over the shipped default. Each shape is then scored against
``results/scan_mode_boundary.json`` (the best-tuned-vs-best-tuned oracle): does
the learned policy pick the mode the exhaustive sweep says wins here?

This is the #17 learned-vs-ceiling comparison and the #14 win check in one
table: the serial-seeding run "learns the crossover" iff the learned mode equals
the oracle winner at BOTH the long-L (chunk_parallel) and saturated (serial)
levels. Pure analysis over committed JSON — no torch, runs anywhere.

Usage:
    uv run python scratch/e2_eval_table.py <ckpt_dir> [--late N] [--out table.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re

_DIR_RE = re.compile(r"^(?P<op>.+)_B(?P<b>\d+)L(?P<l>\d+)W(?P<w>\d+)$")


def _scan_mode_of(source: str) -> str | None:
    """The scan_mode a config emission selected, or None if unparseable/unset."""
    try:
        obj = json.loads(source)
    except (json.JSONDecodeError, ValueError):
        return None
    mode = obj.get("scan_mode") if isinstance(obj, dict) else None
    return str(mode) if isinstance(mode, str) else None


def load_level(level_dir: str, late: int) -> dict[str, object] | None:
    """Summarise one level dir: converged mode (late non-seeded) + best speedup."""
    m = _DIR_RE.match(os.path.basename(level_dir))
    rollouts = os.path.join(level_dir, "rollouts.jsonl")
    if m is None or not os.path.exists(rollouts):
        return None
    with open(rollouts, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        return None
    max_step = max(int(r.get("step", 0)) for r in rows)
    cutoff = max_step - late + 1
    modes: dict[str, int] = {"serial": 0, "chunk_parallel": 0, "default": 0}
    best_speedup = 1.0
    for r in rows:
        s = r.get("speedup")
        if isinstance(s, (int, float)) and s > best_speedup:
            best_speedup = float(s)
        if r.get("seeded") or int(r.get("step", 0)) < cutoff or "source" not in r:
            continue
        mode = _scan_mode_of(str(r["source"]))
        modes[mode if mode in modes else "default"] += 1
    chosen = max(modes, key=lambda k: modes[k]) if sum(modes.values()) else None
    total = sum(modes.values())
    return {
        "op": m["op"],
        "batch": int(m["b"]),
        "seq_len": int(m["l"]),
        "width": int(m["w"]),
        "steps": max_step,
        "learned_mode": chosen,
        "learned_mode_frac": (modes.get(chosen, 0) / total) if (chosen and total) else 0.0,
        "best_speedup": round(best_speedup, 4),
    }


def _oracle(boundary: list[dict[str, object]], op: str, b: int, seqlen: int, w: int) -> dict | None:
    for row in boundary:
        if (
            row.get("op") == op
            and row.get("batch") == b
            and row.get("seq_len") == seqlen
            and row.get("width") == w
        ):
            return row
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--late", type=int, default=5, help="late steps that count as 'converged'")
    ap.add_argument("--boundary", default="results/scan_mode_boundary.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    boundary: list[dict[str, object]] = []
    if os.path.exists(args.boundary):
        with open(args.boundary, encoding="utf-8") as f:
            boundary = json.load(f)

    levels = sorted(
        d
        for d in (os.path.join(args.ckpt_dir, x) for x in os.listdir(args.ckpt_dir))
        if os.path.isdir(d) and _DIR_RE.match(os.path.basename(d))
    )
    table: list[dict[str, object]] = []
    for level_dir in levels:
        row = load_level(level_dir, args.late)
        if row is None:
            continue
        orc = _oracle(boundary, row["op"], row["batch"], row["seq_len"], row["width"])  # type: ignore[arg-type]
        row["oracle_winner"] = orc.get("winner") if orc else None
        row["matches_oracle"] = bool(orc) and row["learned_mode"] == orc.get("winner")
        table.append(row)

    hdr = (
        f"{'op':24} {'shape(BxLxW)':18} {'oracle':14} {'learned':14} {'frac':5} {'speedup':8} match"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        shape = f"{r['batch']}x{r['seq_len']}x{r['width']}"
        print(
            f"{r['op']:24} {shape:18} {r['oracle_winner']!s:14} "
            f"{r['learned_mode']!s:14} {r['learned_mode_frac']:.2f}  "
            f"{r['best_speedup']!s:<8} {'OK' if r['matches_oracle'] else 'x'}"
        )
    scored = [r for r in table if r["oracle_winner"] is not None]
    if scored:
        n_ok = sum(1 for r in scored if r["matches_oracle"])
        print(f"\nlearned mode == oracle winner: {n_ok}/{len(scored)} shapes")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
