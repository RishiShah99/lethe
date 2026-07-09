# lethe

Native Blackwell training kernels for the gated-linear-recurrence family, and a
contract-grade verifier that measures the correctness gap in LLM-generated GPU
kernels.

Package `lethe`, repository `flash-mamba-rl`. One project, one correctness spine:
a hand-written native `tcgen05` training backward for the GDN family, and the
contract verifier + at-scale audit that sits underneath it as ground truth.

## What this is

Two headline results and a stack of supporting work. Two of the pieces are done
and verified; one (RL that discovers faster kernels) is a deliberately reported
negative. Every claim below is backed by the reference oracles and the verifier,
not by a loose `allclose`.

### 1. A native `tcgen05` GDN-family training backward · flagship

The frontier labs shipped forward, decode, and Triton paths for the gated linear
recurrence (Mamba-2/SSD, GLA, KDA, DeltaNet/GDN-2). The **training backward**, a
reverse-state scan plus a WY / triangular-inverse VJP, had no hand-written native
Blackwell (`sm_100`) tensor-memory implementation. This builds it, in the CUTLASS
CuTe DSL, from two `tcgen05` kernels:

- **K#1, reverse-state scan.** Carries the adjoint recurrence state backward
  through the chunks, resident in TMEM. The novel piece.
- **K#2, WY / triangular-inverse VJP.** The backward of the chunked delta-rule
  update, 7 GEMMs per chunk on a `(128, 64, 128)` tile.

It ships scalar-GDN first, then the channel-wise crown (the full GDN-2 backward,
per-channel gates). Verified against an independent fp64 oracle at worst relative
error **3.29e-3 (scalar) / 3.31e-3 (channel-wise)**, under a 5e-3 pin,
bit-deterministic. It **trains**: a 300-step loop with zero NaN, and LA / GLA /
KDA / SSD each train through the native path (native fallback disabled), so "it
trains the family" is a checked reduction, not a slogan.

The engineering story is the TMEM lifecycle. The first attempt to run two GEMMs
in one kernel deadlocked as illegal PTX: it allocated the full 512-element TMEM
and relinquished it per GEMM. NVIDIA's own `mamba2_ssd.py`, in the same pinned
toolchain, runs four `tcgen05` MMAs with one allocation, static per-accumulator
column offsets, and a single relinquish. Porting that lifecycle unblocked the
fully-fused native backward.

### 2. A contract-grade verifier and a rigor-gap audit · working, the strongest asset

A 12-gate contract battery that separates a correct GPU kernel from a
plausible-looking wrong one: value parity, shape polymorphism, determinism,
non-finite dataflow, mixed-precision accumulation, subnormal handling, resource
limits. Built from scratch, calibrated on real B200 silicon, red-teamed against
our own cheating kernels before it was ever pointed at foreign code.

Run against the kernels a public RL-kernel-generation system's own harness
accepted (2,638-kernel denominator):

- **39.5% fail on a tolerance-free channel** (determinism, shape, non-finite):
  a floor no `allclose`-loosening argument can reach.
- **62.1% carry at least one contract violation** their own harness accepted.
- Fairness-hardened three ways: a **positive control** (7/7 of our own kernels
  pass the same battery), a threshold-calibration sweep (every band-gate trips
  only in the clean zone between honest noise and real error), and a
  **differential** against KernelBench's own correctness code (98.5% agreement
  with our replica; by its own check it accepts **1,487** kernels that fail our
  battery, 958 of them on a tolerance-free channel).

The battery is a runnable, at-scale operationalization of the Kernel Contracts
taxonomy (arXiv:2604.22032), cited as the taxonomy source, not our invention;
KernelBench (arXiv:2502.10517) is the field's acceptance signal we measure
against, not an endorser.

### 3. Six verified Mamba-3 Triton kernels · working

The references and six hand-written Triton kernels (C1 to C6) are the ground
truth the whole project stands on: forward chunked scan, backward selective scan,
MIMO backward, complex/RoPE scan, fused block forward and backward. The MIMO
backward is the **first open Mamba-3 MIMO backward anywhere**. All six are
contract-green on B200.

The calling card: the official Mamba-3 Triton backward fails to compile on B200
(`sm_100`) at every `num_warps >= 4` config, a tensor-memory budget overflow
(`Required: 544, Hardware limit: 512`), `state-spaces/mamba#904`, reproduced on
the box. Our backward kernels use no `tl.dot`, so the broken promotion pass never
engages and they compile at every config. An availability win, runs where theirs
crashes, not a raw-speed win (see the honest scoreboard).

### 4. RL that discovers faster kernels · reported negative

A GRPO trainer built from scratch (group-relative advantages, PPO clip, KL, LoRA
policy) that uses the verifier as reward. The reward is staged: 0 if it does not
compile, 0.1 if it fails contracts, 0.5 if correct but slower than the
hand-written kernel, and `1 + log(speedup)` only once it is faster. Speedup pays
only after correctness.

Two honest outcomes. Config-search is a **modest autotuner win** (1.167x at long
sequence over our own default, which is autotuning, not discovery). Source-editing
does **not** discover a faster verified kernel, and the verifier caught every
wrong edit. Both are reported as findings, not buried.

## Honest speed scoreboard (B200)

The speed story is an honest negative, told against three baselines that are never
mixed:

- **Native GDN backward vs the `fla` Triton path:** a **structural** gap that
  widens with length (about 8x at L=512, about 78x at L=2048). K#1 is a sequential
  reverse scan and the channel-wise crown does more work than a scalar path. We
  never claim `fla` parity.
- **A real crown on our own device time:** the save-forward `d_v=128` crown, an
  N-tiled dual-`dp` mainloop that holds TMEM at 384 and reuses the accumulator,
  cut the captured save-forward backward **24.50 -> 8.23 ms, 2.98x**. One
  disclosed caveat: the `d_v=128` whole-pipeline parity is 5.21e-3, marginally
  over the 5e-3 pin (fp16 accumulation over 2x the channels; the kernel itself is
  5.5e-4 vs the fp64 spec).
- **Six Triton kernels vs official CUDA:** forward scan is near parity at
  realistic widths (up to 1.04x at batch 8 / width 4096); the backward is slower;
  MIMO and complex/RoPE have no open comparator.

Faster kernels demonstrably exist, which is exactly the headroom the RL agent has
yet to reach.

## Applied demonstration

A 1.1B-parameter Mamba-3 ECG classifier trained end-to-end through the C5 fused
block on PTB-XL, **0.880 superclass macro-AUC**. A "the stack trains a real
model" demonstration, below the 0.93 to 0.95 recipe SOTA by recipe, not by
kernel. Not a SOTA claim.

## Layout

```
src/lethe/
  verifier/    12-gate contract battery, op-harness, candidate scoring, audit
  kernels/
    references/  fp64/torch oracles (ground truth)
    ops/         hand-written Triton kernels (C1 to C6)
    cute/        native tcgen05 GDN backward (K#1, K#2, assemble, dispatch)
  rl/          GRPO loss, HF + LoRA policy, trainer, curriculum, SFT warm-start
  medical/     PTB-XL loader + 1.1B Mamba-3 classifier
tests/         CPU + GPU suites, plus the cheating-kernel red-team battery
```

## Running

```bash
uv sync --extra dev                  # CPU: verifier + RL logic + references
uv run pytest -q                     # local gate suite (GPU tests auto-skip)
uv run ruff check src tests
uv run mypy src/lethe

uv sync --extra gpu --extra rl       # Blackwell box: Triton kernels + policy
uv sync --extra medical              # PTB-XL loader + classifier
```

Local gate suite: 1,083 passed / 117 CUDA-skipped, ruff and format and mypy
clean. Full GPU suite on the B200 box: 1,197 passed, positive control 7/7. The
`cute/` native backward requires the pinned CUTLASS CuTe DSL toolchain and
Blackwell silicon.

## What's novel

A verifier rigorous enough to falsify published kernels; the first hand-written
native `tcgen05` training backward for the GDN family, with the family reductions
checked, not asserted; the first open Mamba-3 MIMO backward; and an RL loop that
grades kernel generation on contract-grounded correctness before it ever rewards
speed.
