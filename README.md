# Lethe

Paper: [arXiv:2608.12700](https://arxiv.org/abs/2608.12700) · archived at [doi.org/10.5281/zenodo.21563212](https://doi.org/10.5281/zenodo.21563212)

Native Blackwell (`sm_100`) training kernels for the gated-linear-recurrence family, and a contract-grade verifier that measures the correctness gap in LLM-generated GPU kernels. 

GPU kernels are the bottom of the AI stack. The leading post-transformer family
(Mamba-2/SSD, GLA, KDA, DeltaNet/GDN-2) shipped forward, decode, and Triton
paths, but the **training backward**, a reverse-state scan plus a WY /
triangular-inverse VJP, had no full native Blackwell tensor-memory
implementation: the reverse-state scan still ran in Triton everywhere. This builds
it, verifies it against an fp64-anchored oracle chain, and grades LLM-generated
kernels with a battery strict enough to catch what the standard check passes.

## 1. Native `tcgen05` GDN-family training backward · flagship

Two `tcgen05` kernels in the CUTLASS CuTe DSL:

- **K#1, reverse-state scan.** Carries the adjoint recurrence backward through the
chunks, with the fp32 `dS` carry held outside TMEM. At `d_v=64` it is one fused
kernel holding a single TMEM reservation for the MMA accumulators across the
whole reverse scan; other shapes launch one tensor-core GEMM per step. The novel
piece.
- **K#2, WY / triangular-inverse VJP.** Backward of the chunked delta-rule update
on a `(128, 64, 128)` tile, reusing the forward's triangular inverse instead of
re-inverting.

Scalar-GDN first, then the channel-wise crown (full per-channel gates). Checked
against the fp32 reference backward (autograd through a forward reference that
matches an independent fp64 implementation to ~1e-16)
at relative error **3.29e-3 (scalar) / 3.31e-3 to 3.67e-3 (channel-wise)**, all
under a 5e-3 pin. It trains: all five
family members (LA, GLA, SSD, KDA, full GDN) train through the native path with
the fallback disabled, zero NaN. Four clear the 300-step gate at L=256; LA is the
weak one, failing that gate and clearing it only at L=64 over 4000 steps. The
engineering story is the TMEM lifecycle. The first fused multi-GEMM kernel failed
with compiler ICEs, segfaults, and launch failures (full 512-column allocation,
relinquished per GEMM); porting NVIDIA's
`mamba2_ssd.py` lifecycle (one allocation, static column offsets, single
relinquish) unblocked the fully-fused backward.

## 2. Contract-grade verifier + rigor-gap audit · strongest asset

A 12-gate contract battery separating a correct kernel from a plausible-looking
wrong one: value parity, shape polymorphism, determinism, non-finite dataflow,
mixed-precision accumulation, subnormal handling, resource limits. Built from
scratch, thresholds derived from rounding-error models and calibrated against a
known-correct kernel, red-teamed against 19 of our own cheating kernels
before it saw foreign code. Run against the kernels a public kernel-generation
system's own harness accepted (2,638-kernel denominator):

- **39.5% fail a tolerance-free channel** (non-finite propagation, determinism,
aliasing, crash/run failure), a floor no `allclose`-loosening can reach. Shape
rejection is held out of the floor; adding CMP-03 raises it to 41.1%.
- **62.1% carry at least one contract violation** their harness accepted.
- Hardened four ways: a **positive control** (our six Triton kernels pass the same
battery), a threshold-calibration sweep, **98.5% agreement** with KernelBench's
own correctness code over the 1,030 pairs where both ran, and a stratified
31-row hand-audit.
- The **differential**, run with our replica of the standard `allclose` check
over all 2,638: the standard test accepts **1,487** kernels the battery rejects
(918 on the same tolerance-free channels as the 39.5% floor) against 14 in reverse.

The battery is a runnable, at-scale operationalization of the Kernel Contracts
taxonomy (arXiv:2604.22032), cited as the source not our invention; KernelBench
(arXiv:2502.10517) is the acceptance signal we measure against.

## 3. Six verified Mamba-3 Triton kernels · working

The references and six Triton kernels (C1 to C6) are the ground truth the whole
project stands on: forward chunked scan, backward selective scan, MIMO backward,
complex/RoPE scan, fused block forward and backward. Priority on the MIMO
backward is TileLang's, which ships the official one; ours is an independent
Triton implementation. All six are contract-green on B200.

Calling card: the official Mamba-3 Triton backward fails to compile on `sm_100`
at every `num_warps >= 4` config, a tensor-memory overflow (`Required: 544, Hardware limit: 512`, `state-spaces/mamba#904`, reproduced on the box). Ours use
no `tl.dot`, so the broken promotion pass never engages and they compile at every
config, where theirs loses 18 autotune configs and falls back to `num_warps=2`.
An availability win, not a raw-speed win.

## 4. RL that discovers faster kernels · reported negative

A from-scratch GRPO trainer (group-relative advantages, PPO clip, KL, LoRA policy)
using the verifier as reward, staged so speedup pays only after correctness: 0 if
it does not compile, 0.1 if it fails contracts, and once every contract passes,
`1 + log(speedup)` clamped to [0.5, 4.0], so a correct kernel always outscores a
wrong one. Two honest outcomes. Config-search is a **modest
autotuner win** (1.167x at long sequence over our own default, which is
autotuning, not discovery). Source-editing does **not** find a faster verified
kernel, and the verifier caught every wrong edit. Both are reported as findings.

## Honest speed scoreboard (B200)

Three baselines, never mixed:


| Comparison                                          | Result                                                                                                                                                                                                      |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native GDN backward vs the `fla` Triton path        | structural gap that widens with length, ~8x at L=512 to ~78x at L=2048 (K#1 is a sequential reverse scan; no `fla`-parity claim)                                                                            |
| Save-forward `d_v=128` crown vs our own device time | **24.50 -> 8.23 ms, 2.98x** (N-tiled dual-`dp` mainloop, 384 TMEM columns used). Caveat: at `d_v=128` the save-forward and default native paths differ by 5.21e-3, marginally over the 5e-3 pin; the K#2 kernel itself is 5.5e-4 vs fp64 |
| Six Triton kernels vs Mamba-1 CUDA                  | forward 0.21x to 1.04x (ahead only at the largest shape, batch 8 / width 4096); backward 0.13x to 0.75x; complex/RoPE has no open comparator                                                     |


Faster kernels demonstrably exist, which is the headroom the RL agent has not
reached.

## Applied demonstration

A 1.1B-parameter selective-scan (Mamba-1 style SISO) ECG classifier trained
end-to-end through the C5 fused block on PTB-XL, **0.880 superclass macro-AUC**. A "the stack trains a real model"
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
  medical/     PTB-XL loader + 1.1B selective-scan classifier
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
Last committed full GPU suite record on the B200 box: 1,191 passed / 3 skipped / 0 failed. The `cute/`
native backward requires the pinned CUTLASS CuTe DSL toolchain and Blackwell
silicon.

## What's novel

A verifier strict enough to catch the kernels the standard check passes; the first hand-written
native `tcgen05` training backward for the GDN family, with the family reductions
checked, not asserted; and an RL loop that grades kernel generation on
contract-grounded correctness before it ever rewards speed.

## Acknowledgements
Thank you to Google Cloud Singapore and E3A Healthcare for providing the compute for this project!
