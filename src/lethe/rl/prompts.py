"""Per-op prompt templates for kernel-generation policies."""

from __future__ import annotations

from lethe.kernels.autotune import SEARCH_GRID, ShapeSpec
from lethe.rl.sft_targets import target_source

_C1_PROMPT = """\
Write a high-performance Triton kernel implementing the Mamba selective \
state-space scan (SISO forward pass) for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def forward_chunked_scan(
    u,       # [B, L, D] float32/float16/bfloat16 CUDA tensor
    delta,   # [B, L, D] same dtype as u (pre-softplus timescale)
    A,       # [D, N] float32, negative log-magnitude SSM matrix
    B,       # [B, L, N] same dtype as u
    C,       # [B, L, N] same dtype as u
    D,       # [D] float32 skip weight
    *,
    chunk_size: int = 64,
) -> "Tensor":  # [B, L, D], same dtype as u
    ...
```

Semantics (zero-order-hold discretisation, then a linear recurrence over L):

    delta_bar = softplus(delta)                       # [B, L, D]
    A_bar     = exp(delta_bar[..., None] * A)         # [B, L, D, N]
    B_bar     = delta_bar[..., None] * B[:, :, None]  # [B, L, D, N]

    h_0 = 0                                           # [B, D, N]
    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t[..., None]
    y_t = (h_t * C_t).sum(-1) + D * u_t               # [B, L, D]

PITFALL: the arguments named B, C, D are TENSORS (input projection,
output projection, skip weight). Do NOT shadow them with shape variables:
write `batch, seqlen, dmodel = u.shape`, never `B, L, D = u.shape`.

The output must satisfy a 12-gate contract verifier, graded against a
float32 eager reference:

- Value parity on random and adversarial inputs (zeros, +/-1e6, 1e-6,
  subnormals, 4x longer sequences) within reorder-noise tolerance
  (eps * sqrt(chain length) * output scale). Wrong math fails by orders
  of magnitude.
- Shape polymorphism: any B, any D, any L divisible by chunk_size.
- Byte-identical determinism across repeated calls: do NOT use atomics;
  fixed reduction order only.
- NaN and signed-Inf positions in the output must match the reference
  exactly (non-finites propagate through the recurrence; do not mask,
  clamp, or nan_to_num).
- Mixed precision: fp16/bf16 inputs must be computed with float32 internal
  state and accumulators, rounding once at the output. A half-precision
  state carry fails the gate.
- Subnormal inputs must not be flushed to zero (avoid ftz approximations
  for exp/log; guard softplus overflow with a linear branch above
  threshold ~20).
- Output device/dtype must match the input.

Performance is rewarded only after every gate passes (log-scale speedup
bonus vs the hand-written kernel). Compile failures and contract failures
earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the forward_chunked_scan
function). No prose after the code block.
"""

_ELEMENTWISE_SILU_PROMPT = """\
Write a high-performance Triton kernel implementing elementwise SiLU
(x * sigmoid(x)) for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and
helpers it needs):

```python
def elementwise_silu(x):  # x: [B, L, D] float32/float16/bfloat16 CUDA tensor
    ...                   # returns [B, L, D], same dtype and device as x
```

The output must satisfy a 12-gate contract verifier, graded against a
float32 eager reference computing x * sigmoid(x):

- Value parity on random and adversarial inputs (zeros, +/-1e6, 1e-6,
  subnormals, 4x longer sequences) within near-ULP tolerance.
- Shape polymorphism: any batch, any sequence length, any model dim.
- Byte-identical determinism across repeated calls.
- NaN and signed-Inf positions in the output must match the reference
  exactly (SiLU(-inf) = -inf * 0 = NaN; do not mask, clamp, or
  nan_to_num).
- Mixed precision: fp16/bf16 inputs must be computed in float32
  internally, rounding once at the output.
- Subnormal inputs must not be flushed to zero (avoid ftz exp
  approximations).
- Output device/dtype must match the input.

Performance is rewarded only after every gate passes. Compile failures
and contract failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the elementwise_silu function).
No prose after the code block.
"""

_C2_PROMPT = """\
Write a high-performance Triton kernel implementing the BACKWARD pass
(analytic gradients) of the Mamba selective state-space scan for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def backward_selective_scan(
    u,       # [B, L, D] CUDA tensor (forward input)
    delta,   # [B, L, D] (pre-softplus timescale)
    A,       # [D, N] negative log-magnitude SSM matrix
    B,       # [B, L, N] input projection
    C,       # [B, L, N] output projection
    D,       # [D] skip weight
    dy,      # [B, L, D] upstream gradient w.r.t. y
    *,
    chunk_size: int = 64,
):  # -> (grad_u, grad_delta, grad_A, grad_B, grad_C, grad_D)
    ...
```

All tensors arrive on the same CUDA device with the same dtype
(float32/float16/bfloat16). Return the six gradients in exactly that
order, each matching its input's shape and dtype.

The forward pass being differentiated (zero-order-hold discretisation,
linear recurrence over L):

    delta_bar = softplus(delta)                       # [B, L, D]
    A_bar     = exp(delta_bar[..., None] * A)         # [B, L, D, N]
    B_bar     = delta_bar[..., None] * B[:, :, None]  # [B, L, D, N]

    h_0 = 0                                           # [B, D, N]
    h_t = A_bar_t * h_{t-1} + B_bar_t * u_t[..., None]
    y_t = (h_t * C_t).sum(-1) + D * u_t               # [B, L, D]

Your gradients are graded against torch.autograd run through a float32
eager reference. EACH of the six gradient outputs is run through the full
12-gate contract battery as its own view; the contract reward requires
ALL SIX views to pass.

- Value parity per gradient within reorder-noise tolerance
  (eps * sqrt(chain length) * output scale). Wrong math fails by orders
  of magnitude. The reverse-time carry never inverts the recurrence
  (dividing by A_bar underflows at saturated delta), recompute or
  checkpoint the forward state instead.
- Shape polymorphism: any B, any D, any L divisible by chunk_size.
- Byte-identical determinism across repeated calls: no atomics; fixed
  reduction order only (per-program partials + a deterministic final sum
  is the standard pattern).
- NaN/signed-Inf positions in every gradient must match autograd's
  dataflow EXACTLY when dy carries non-finites. Algebraically equivalent
  refactorings can mint different non-finites (e.g. at t=0 where
  h_{-1} = 0, a factored expression produces +/-Inf where autograd
  produces NaN), mirror autograd's grouping of products and keep the
  delta-path reductions separate.
- Mixed precision: fp16/bf16 inputs must be computed with float32 state
  and accumulators, rounding once at each output. A half-precision
  state or accumulator carry fails the gate.
- Subnormal inputs must not be flushed (no ftz exp approximations;
  guard softplus with a linear branch above threshold ~20).

PITFALL: the arguments named B, C, D are TENSORS. Do NOT shadow them
with shape variables: write `batch, seqlen, dmodel = u.shape`, never
`B, L, D = u.shape`.

Performance is rewarded only after every gate of every view passes
(log-scale speedup bonus vs the hand-written kernel). Compile failures
and contract failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the backward_selective_scan
function). No prose after the code block.
"""

_C3_PROMPT = """\
Write a high-performance Triton kernel implementing the BACKWARD pass of
the Mamba-3 MIMO (rank-R) selective state-space scan for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def mimo_backward(
    x,       # [B, L, H, P] CUDA tensor (forward input)
    B,       # [B, L, R, H, N] input projection (pre-rotated)
    C,       # [B, L, R, H, N] output projection (pre-rotated)
    dt,      # [B, L, H] positive timescale
    alpha,   # [B, L, H] decay, alpha = exp(dt * A), in (0, 1)
    mimo_x,  # [H, R, P] input mixing weights (psi)
    mimo_o,  # [H, R, P] output mixing weights (phi)
    dy,      # [B, L, H, P] upstream gradient w.r.t. y
):  # -> (grad_x, grad_B, grad_C, grad_dt, grad_alpha, grad_mimo_x, grad_mimo_o)
    ...
```

(B=batch, L=seqlen, H=nheads, P=headdim, R=rank, N=d_state.) All tensors
arrive on the same CUDA device with the same dtype
(float32/float16/bfloat16). Return the seven gradients in exactly that
order, each matching its input's shape and dtype.

The forward pass being differentiated (Mamba-3 Eqs 12-14):

    x_r[j]   = x * mimo_x[h, j, :]                  # rank-expanded input
    h_t^(j)  = alpha_t * h_{t-1}^(j) + dt_t * B_t^(j) * x_r_t^(j)
               # per-rank state, shape [B, H, P, N], h_0 = 0
    h_agg_t  = sum_j h_t^(j)
    y_raw^(i)= (C_t^(i) * h_agg_t).sum over N       # [B, L, R, H, P]
    y_t      = sum_i y_raw_t^(i) * mimo_o[h, i, :]  # [B, L, H, P]

Structural fact worth exploiting: alpha is rank-independent and the
readout distributes over the rank sum, so the reverse-time carry is
IDENTICAL for every rank, carry only the aggregated state's gradient,
never R per-rank states.

Your gradients are graded against torch.autograd run through a float32
eager reference. EACH of the seven gradient outputs is run through the
full 12-gate contract battery as its own view; the contract reward
requires ALL SEVEN views to pass.

- Value parity per gradient within reorder-noise tolerance
  (eps * sqrt(chain length) * output scale).
- Shape polymorphism: any batch, any L, any H, any P, any R, any N.
- Byte-identical determinism: no atomics; fixed reduction order
  (per-program partials + a deterministic final sum is the standard
  pattern for the parameter gradients summed over batch and L).
- NaN/signed-Inf positions in every gradient must match autograd's
  dataflow exactly when dy carries non-finites, mirror autograd's
  grouping of products.
- Mixed precision: fp16/bf16 inputs must be computed with float32 state
  and accumulators, rounding once at each output.

PITFALL: the arguments named B and C are TENSORS. Do NOT shadow them
with shape variables: write `batch, seqlen, nheads, headdim = x.shape`,
never `B, L, H, P = x.shape`.

Performance is rewarded only after every gate of every view passes
(log-scale speedup bonus vs the hand-written kernel). Compile failures
and contract failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the mimo_backward function).
No prose after the code block.
"""

_C4_PROMPT = """\
Write a high-performance Triton kernel implementing the Mamba-3
real-equivalent SSM scan with data-dependent RoPE rotation for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def complex_scan_rope(
    x,          # [B, L, H, P] CUDA tensor
    B,          # [B, L, H, N] input projection
    C,          # [B, L, H, N] output projection
    dt,         # [B, L, H] positive timescale
    A,          # [H] negative per-head decay rate
    angle_proj, # [B, L, H, S] rotation angle logits
) -> "Tensor":   # [B, L, H, P], same dtype as x
    ...
```

(B=batch, L=seqlen, H=nheads, P=headdim, N=d_state, S=num_rope_angles
with 2*S <= N.) All tensors arrive on the same CUDA device with the same
dtype (float32/float16/bfloat16).

Semantics: fold the cumulative rotation into B and C, then run a plain
decay scan (the hidden state itself is NEVER re-rotated):

    Theta_t  = cumsum_{s<=t}( tanh(angle_proj_s) * dt_s * pi )  mod 2*pi
               # [B, L, H, S], causal cumulative angle
    B_rot_t  = R(Theta_t) @ B_t,  C_rot_t = R(Theta_t) @ C_t
               # R = block-diagonal pairwise rotation on consecutive
               # lanes (2k, 2k+1) for k < S: with c = cos, s = sin,
               #   out_even = v_even * c - v_odd * s
               #   out_odd  = v_even * s + v_odd * c
               # SAME rotation sign for B and C (no conjugation);
               # lanes >= 2*S pass through unchanged (identity).
    alpha_t  = exp(dt_t * A)                        # [B, L, H]
    h_t      = alpha_t * h_{t-1} + dt_t * B_rot_t (outer) x_t
               # h: [B, H, P, N], h_0 = 0
    y_t      = (h_t * C_rot_t).sum over N           # [B, L, H, P]

The output must satisfy the 12-gate contract verifier, graded against a
float32 eager reference:

- Value parity on random and adversarial inputs within reorder-noise
  tolerance (eps * sqrt(chain length) * output scale). Keep cos/sin
  arguments small: accumulate the angle with a per-step remainder
  rather than one long fp32 cumsum.
- Shape polymorphism: any batch, any L, any H, any P, any N, any S
  with 2*S <= N.
- Byte-identical determinism across repeated calls: no atomics; fixed
  reduction order only.
- NaN and signed-Inf positions in the output must match the reference
  exactly (non-finites propagate through the recurrence; do not mask,
  clamp, or nan_to_num).
- Mixed precision: fp16/bf16 inputs must be computed with float32
  internal state and accumulators, rounding once at the output.
- Subnormal inputs must not be flushed (no ftz approximations for
  exp/cos/sin).

PITFALL: the arguments named B and C are TENSORS. Do NOT shadow them
with shape variables: write `batch, seqlen, nheads, headdim = x.shape`,
never `B, L, H, P = x.shape`.

Performance is rewarded only after every gate passes (log-scale speedup
bonus vs the hand-written kernel). Compile failures and contract
failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the complex_scan_rope
function). No prose after the code block.
"""

_C5_PROMPT = """\
Write a high-performance Triton kernel implementing the fused Mamba block
forward pass (conv1d + SiLU + selective scan + RMSNorm) for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def fused_block_forward(
    x,           # [B, L, D] CUDA tensor, ALREADY left-padded with
                  # conv_kernel_size - 1 zeros along L (causal padding)
    conv_weight, # [D, 1, K] depthwise conv kernel
    conv_bias,   # [D]
    delta,       # [B, L_out, D] (pre-softplus timescale), L_out = L - (K-1)
    A,           # [D, N] negative log-magnitude SSM matrix
    B,           # [B, L_out, N] input projection
    C,           # [B, L_out, N] output projection
    D,           # [D] skip weight
    norm_weight, # [D] RMSNorm gain
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
) -> "Tensor":    # [B, L_out, D], same dtype as x
    ...
```

All tensors arrive on the same CUDA device with the same dtype
(float32/float16/bfloat16).

Semantics, in order:

    1. Causal depthwise conv1d over L (groups = D, VALID convolution, 
       the input already carries the K-1 left padding):
       conv_out[b, t, d] = sum_k conv_weight[d, 0, k] * x[b, t+k, d]
                           + conv_bias[d]            # [B, L_out, D]
    2. z = silu(conv_out) = conv_out * sigmoid(conv_out)
    3. Selective scan over z (zero-order-hold discretisation):
       delta_bar = softplus(delta)
       A_bar     = exp(delta_bar[..., None] * A)
       B_bar     = delta_bar[..., None] * B[:, :, None]
       h_t = A_bar_t * h_{t-1} + B_bar_t * z_t[..., None]   # h_0 = 0
       y_scan_t = (h_t * C_t).sum(-1) + D * z_t             # [B, L_out, D]
    4. RMSNorm over the D dimension:
       y_t = y_scan_t / sqrt(mean_D(y_scan_t^2) + eps) * norm_weight

The output must satisfy the 12-gate contract verifier, graded against a
float32 eager reference:

- Value parity on random and adversarial inputs within reorder-noise
  tolerance (eps * sqrt(chain length) * output scale).
- Shape polymorphism: any batch, any D, any L_out divisible by
  chunk_size.
- Byte-identical determinism across repeated calls: no atomics; fixed
  reduction order only (the cross-D RMSNorm reduction must be
  deterministic, a full-D fp32 sum per (batch, t) program is the
  standard pattern).
- NaN and signed-Inf positions in the output must match the reference
  exactly (non-finites propagate through conv, SiLU, the recurrence and
  the norm; do not mask, clamp, or nan_to_num).
- Mixed precision: fp16/bf16 inputs must be computed with float32
  internal state and accumulators (including the RMSNorm sum-of-squares),
  rounding once at the output.
- Subnormal inputs must not be flushed (no ftz approximations; guard
  softplus with a linear branch above threshold ~20).

PITFALL: the arguments named B, C, D are TENSORS. Do NOT shadow them
with shape variables: write `batch, seqlen, dmodel = x.shape`, never
`B, L, D = x.shape`.

Performance is rewarded only after every gate passes (log-scale speedup
bonus vs the hand-written kernel). Compile failures and contract
failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the fused_block_forward
function). No prose after the code block.
"""

_C6_PROMPT = """\
Write a high-performance Triton kernel implementing the BACKWARD pass of
the fused Mamba block (conv1d + SiLU + selective scan + RMSNorm) for
CUDA GPUs. This is the hardest op in the curriculum.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def fused_block_backward(
    x,           # [B, L, D] CUDA tensor, ALREADY left-padded with
                  # conv_kernel_size - 1 zeros along L (causal padding)
    conv_weight, # [D, 1, K] depthwise conv kernel
    conv_bias,   # [D]
    delta,       # [B, L_out, D] (pre-softplus), L_out = L - (K-1)
    A,           # [D, N] negative log-magnitude SSM matrix
    B,           # [B, L_out, N] input projection
    C,           # [B, L_out, N] output projection
    D,           # [D] skip weight
    norm_weight, # [D] RMSNorm gain
    dy,          # [B, L_out, D] upstream gradient w.r.t. the block output
    *,
    conv_kernel_size: int = 4,
    eps: float = 1e-5,
    chunk_size: int = 64,
):  # -> (grad_x, grad_conv_weight, grad_conv_bias, grad_delta, grad_A,
    #     grad_B, grad_C, grad_D, grad_norm_weight)
    ...
```

All tensors arrive on the same CUDA device with the same dtype
(float32/float16/bfloat16). Return the nine gradients in exactly that
order, each matching its input's shape and dtype, grad_x matches the
PADDED x, shape [B, L, D].

The forward pass being differentiated, in order:

    1. conv_out = causal depthwise conv1d(x) + conv_bias   # [B, L_out, D]
    2. z = silu(conv_out)
    3. selective scan:  delta_bar = softplus(delta)
       h_t = exp(delta_bar*A)_t * h_{t-1} + (delta_bar*B)_t * z_t
       y_scan_t = (h_t * C_t).sum(-1) + D * z_t
    4. y = y_scan / sqrt(mean_D(y_scan^2) + eps) * norm_weight

Your gradients are graded against torch.autograd run through a float32
eager reference. EACH of the nine gradient outputs is run through the
full 12-gate contract battery as its own view; the contract reward
requires ALL NINE views to pass.

- Value parity per gradient within reorder-noise tolerance
  (eps * sqrt(chain length) * output scale). The reverse-time carry
  never inverts the recurrence (dividing by exp(delta_bar*A) underflows
  at saturated delta), checkpoint and recompute the forward state.
- Shape polymorphism: any batch, any D, any L_out divisible by
  chunk_size.
- Byte-identical determinism: no atomics; fixed reduction order only
  (per-program fp32 partials + a deterministic final sum for the
  parameter gradients summed over batch and L).
- NaN/signed-Inf positions in every gradient must match autograd's
  dataflow exactly when dy carries non-finites, mirror autograd's
  grouping of products (e.g. at t=0 where h_{-1} = 0, a factored
  expression can mint +/-Inf where autograd produces NaN).
- Mixed precision: fp16/bf16 inputs must be computed with float32 state
  and accumulators throughout (scan state, RMSNorm sums, conv partials),
  rounding once at each output.
- Subnormal inputs must not be flushed (no ftz approximations; guard
  softplus with a linear branch above threshold ~20).

PITFALL: the arguments named B, C, D are TENSORS. Do NOT shadow them
with shape variables: write `batch, seqlen, dmodel = x.shape`, never
`B, L, D = x.shape`.

Performance is rewarded only after every gate of every view passes
(log-scale speedup bonus vs the hand-written kernel). Compile failures
and contract failures earn near-zero reward, so correctness comes first.

Reply with ONE fenced ```python code block containing the complete,
self-contained module (imports, kernels, the fused_block_backward
function). No prose after the code block.
"""

_OP_PROMPTS: dict[str, str] = {
    "forward_chunked_scan": _C1_PROMPT,
    "elementwise_silu": _ELEMENTWISE_SILU_PROMPT,
    "backward_selective_scan": _C2_PROMPT,
    "mimo_backward": _C3_PROMPT,
    "complex_scan_rope": _C4_PROMPT,
    "fused_block_forward": _C5_PROMPT,
    "fused_block_backward": _C6_PROMPT,
}


def build_op_prompt(op_name: str) -> str:
    """Return the generation prompt for *op_name* (KeyError if unknown)."""
    return _OP_PROMPTS[op_name]


def available_ops() -> tuple[str, ...]:
    return tuple(_OP_PROMPTS)


# The config-emission action space.
_KNOB_DESC: dict[str, str] = {
    "block_d": "D (model-dim) tile width",
    "block_p": "P (head-dim) tile width",
    "chunk_k": "in-chunk recompute window of the checkpointed backward (must divide seq_len)",
    "num_warps": "warps per program (occupancy vs per-thread registers)",
    "num_stages": "software-pipelining depth of the kernel's main loop",
    "scan_mode": "SISO scan algorithm: 'serial' (a sequential O(L) walk over the "
    "sequence) or 'chunk_parallel' (an SSD chunked-carry reassociating the scan "
    "across L/chunk_len chunks). Both are correct; which is faster depends on the shape",
    "chunk_len": "chunk-parallel granularity (must divide seq_len; no-op in serial mode)",
}

_CONFIG_PROMPT_TEMPLATE = """\
You are tuning the CUDA launch configuration of a verified-correct Triton
kernel implementing `{op}` on an NVIDIA B200 GPU. The kernel's numerics are
fixed and already correct, you choose ONLY performance knobs that cannot
change the result. Pick the fastest configuration for this exact shape:

    batch = {batch}, seq_len = {seq_len}, d_model = {width}

Choose values for these launch knobs (a JSON object):

{knobs}

Rules:
- Omit a knob to use the kernel's shipped default heuristic for it; the empty
  object {{}} is the shipped default.
- A value outside the listed set is an illegal action and scores zero.
- A config that fails to compile, spills past the register budget, or OOMs
  scores zero; one whose output drifts outside the contract tolerance is
  rejected.
- Speedup over the shipped default is rewarded ONLY after the full gate
  battery re-passes at this shape. Correctness first.

Reply with ONE fenced ```json block holding the config object, e.g.
```json
{{"num_warps": 4, "block_d": 32}}
```
No prose after the block.
"""


def build_config_prompt(op_name: str, shape: ShapeSpec) -> str:
    """Config-emission prompt for *op_name* at *shape* (KeyError if unknown)."""
    grid = SEARCH_GRID[op_name]
    knobs = "\n".join(
        f"    {name}: one of {list(values)}  ({_KNOB_DESC[name]})" for name, values in grid.items()
    )
    return _CONFIG_PROMPT_TEMPLATE.format(
        op=op_name,
        batch=shape.batch,
        seq_len=shape.seq_len,
        width=shape.width,
        knobs=knobs,
    )


# The edit-emission action space.
_EDIT_PROMPT_TEMPLATE = """\
You are optimizing a verified-correct kernel implementing `{op}` on an NVIDIA
B200 GPU. Make it FASTER at this exact shape while keeping it numerically
faithful to the float32 reference it is graded against:

    batch = {batch}, seq_len = {seq_len}, d_model = {width}

Current kernel source:

```python
{base}
```

Express every change as a git-conflict-style SEARCH/REPLACE block:

<<<<<<< SEARCH
<exact lines copied from the source above>
=======
<replacement lines>
>>>>>>> REPLACE

Rules:
- Each SEARCH must match the source above EXACTLY and occur exactly once; you
  may emit several blocks and they apply in order.
- The edited kernel is re-graded by a 12-gate contract verifier against a
  float32 eager reference. An edit that changes the result, fails to compile,
  spills past the register budget, or OOMs scores zero.
- Speedup over the hand-written kernel at this shape is rewarded ONLY after the
  full gate battery re-passes. Correctness first.
- The module must stay self-contained: do NOT import the reference, the official
  Mamba kernels, or any project package.

Reply with ONLY the SEARCH/REPLACE blocks. No prose, no full-file rewrite.
"""


def build_edit_prompt(op_name: str, shape: ShapeSpec, *, base_variant: str = "triton") -> str:
    """Edit-emission prompt for *op_name* at *shape* (KeyError if unknown)."""
    base = target_source(op_name, base_variant)
    return _EDIT_PROMPT_TEMPLATE.format(
        op=op_name,
        batch=shape.batch,
        seq_len=shape.seq_len,
        width=shape.width,
        base=base,
    )
