# Lethe

Native Blackwell (`sm_100`) training kernels for the gated-linear-recurrence family, and a contract-grade verifier that measures the correctness gap in LLM-generated GPU kernels. 

GPU kernels are the bottom of the AI stack. The leading post-transformer family
(Mamba-2/SSD, GLA, KDA, DeltaNet/GDN-2) shipped forward, decode, and Triton
paths, but the **training backward**, a reverse-state scan plus a WY /
triangular-inverse VJP, had no hand-written native Blackwell tensor-memory
implementation. This builds it, verifies it against fp64 oracles, and grades the
whole kernel space with a battery rigorous enough to falsify published work.

## 1. Native `tcgen05` GDN-family training backward · flagship

Two `tcgen05` kernels in the CUTLASS CuTe DSL:

- **K#1, reverse-state scan.** Carries the adjoint recurrence backward through the
chunks, resident in TMEM. The novel piece.
- **K#2, WY / triangular-inverse VJP.** Backward of the chunked delta-rule update,
7 GEMMs per chunk on a `(128, 64, 128)` tile.

Scalar-GDN first, then the channel-wise crown (full per-channel gates). Verified
against an independent fp64 oracle at worst relative error **3.29e-3 (scalar) /
3.31e-3 (channel-wise)** under a 5e-3 pin, bit-deterministic. It trains: 300
steps, zero NaN, and LA / GLA / KDA / SSD each train through the native path with
the fallback disabled, so "trains the family" is a checked reduction. The
engineering story is the TMEM lifecycle. The first two-GEMM kernel deadlocked as
illegal PTX (full 512-element allocation, relinquished per GEMM); porting NVIDIA's
`mamba2_ssd.py` lifecycle (one allocation, static column offsets, single
relinquish) unblocked the fully-fused backward.

## 2. Contract-grade verifier + rigor-gap audit · strongest asset

A 12-gate contract battery separating a correct kernel from a plausible-looking
wrong one: value parity, shape polymorphism, determinism, non-finite dataflow,
mixed-precision accumulation, subnormal handling, resource limits. Built from
scratch, calibrated on B200 silicon, red-teamed against our own cheating kernels
before it saw foreign code. Run against the kernels a public RL-kernel-gen
system's own harness accepted (2,638-kernel denominator):

- **39.5% fail a tolerance-free channel** (determinism, shape, non-finite), a
floor no `allclose`-loosening can reach.
- **62.1% carry at least one contract violation** their harness accepted.
- Hardened three ways: **positive control 7/7** (our kernels pass the same
battery), a threshold-calibration sweep, and a **differential** against
KernelBench's own correctness code (98.5% agreement; by its own check
KernelBench accepts **1,487** kernels that fail our battery, 958 on a
tolerance-free channel).

The battery is a runnable, at-scale operationalization of the Kernel Contracts
taxonomy (arXiv:2604.22032), cited as the source not our invention; KernelBench
(arXiv:2502.10517) is the acceptance signal we measure against.

## 3. Six verified Mamba-3 Triton kernels · working

The references and six Triton kernels (C1 to C6) are the ground truth the whole
project stands on: forward chunked scan, backward selective scan, MIMO backward,
complex/RoPE scan, fused block forward and backward. The MIMO backward is the
**first open Mamba-3 MIMO backward anywhere**. All six are contract-green on B200.

Calling card: the official Mamba-3 Triton backward fails to compile on `sm_100`
at every `num_warps >= 4` config, a tensor-memory overflow (`Required: 544, Hardware limit: 512`, `state-spaces/mamba#904`, reproduced on the box). Ours use
no `tl.dot`, so the broken promotion pass never engages and they compile at every
config. An availability win, runs where theirs crashes, not a raw-speed win.

## 4. RL that discovers faster kernels · reported negative

A from-scratch GRPO trainer (group-relative advantages, PPO clip, KL, LoRA policy)
using the verifier as reward, staged so speedup pays only after correctness: 0 if
it does not compile, 0.1 if it fails contracts, 0.5 if correct but slower,
`1 + log(speedup)` once faster. Two honest outcomes. Config-search is a **modest
autotuner win** (1.167x at long sequence over our own default, which is
autotuning, not discovery). Source-editing does **not** find a faster verified
kernel, and the verifier caught every wrong edit. Both are reported as findings.

## Honest speed scoreboard (B200)

Three baselines, never mixed:


| Comparison                                          | Result                                                                                                                                                                                                      |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native GDN backward vs the `fla` Triton path        | structural gap that widens with length, ~8x at L=512 to ~78x at L=2048 (K#1 is a sequential reverse scan; no `fla`-parity claim)                                                                            |
| Save-forward `d_v=128` crown vs our own device time | **24.50 -> 8.23 ms, 2.98x** (N-tiled dual-`dp` mainloop, TMEM held at 384). Caveat: whole-pipeline `d_v=128` parity is 5.21e-3, marginally over the 5e-3 pin (the kernel itself is 5.5e-4 vs the fp64 spec) |
| Six Triton kernels vs official CUDA                 | forward near parity (up to 1.04x at batch 8 / width 4096); backward slower; MIMO and complex/RoPE have no open comparator                                                                                   |


Faster kernels demonstrably exist, which is the headroom the RL agent has not
reached.

## Applied demonstration

A 1.1B-parameter Mamba-3 ECG classifier trained end-to-end through the C5 fused
block on PTB-XL, **0.880 superclass macro-AUC**. A "the stack trains a real model"
demonstration, below the 0.93 to 0.95 recipe SOTA by recipe, not by kernel. Not a
SOTA claim.

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
uv sync --extra dev                     # CPU: verifier + RL logic + references
uv run pytest -q                        # local gate suite (GPU tests auto-skip)
uv run ruff check src tests && uv run mypy src/lethe

uv sync --extra gpu --extra rl          # Blackwell box: Triton kernels + policy
uv sync --extra medical                 # PTB-XL loader + classifier
```

Local gate suite: 1,083 passed / 117 CUDA-skipped, ruff and format and mypy clean.
Full GPU suite on the B200 box: 1,197 passed, positive control 7/7. The `cute/`
native backward requires the pinned CUTLASS CuTe DSL toolchain and Blackwell
silicon.

## What's novel

A verifier rigorous enough to falsify published kernels; the first hand-written
native `tcgen05` training backward for the GDN family, with the family reductions
checked, not asserted; the first open Mamba-3 MIMO backward; and an RL loop that
grades kernel generation on contract-grounded correctness before it ever rewards
speed.

## Acknowledgements
Thank you to Google (GCP) and E3A Healthcare for providing the resources to run this project!
