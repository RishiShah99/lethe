"""Per-op prompt templates for kernel-generation policies.

One prompt per target op, consumed by the Phase D bakeoff (single-shot
base-model eval) and the GRPO trainer alike. The prompt states the exact
contract the verifier enforces — signature, math, and the gate battery —
so a policy is graded only on what it was told. The math sections restate
the reference docstrings (kernels/references/*) verbatim in substance;
they are the single source of truth.
"""

from __future__ import annotations

_C1_PROMPT = """\
Write a high-performance Triton kernel implementing the Mamba selective \
state-space scan (SISO forward pass) for CUDA GPUs.

Define exactly this Python function (plus any @triton.jit kernels and \
helpers it needs):

```python
def forward_chunked_scan(
    u,        # [B, L, D] float32/float16/bfloat16 CUDA tensor
    delta,    # [B, L, D] same dtype as u (pre-softplus timescale)
    A,        # [D, N] float32, negative log-magnitude SSM matrix
    B,        # [B, L, N] same dtype as u
    C,        # [B, L, N] same dtype as u
    D,        # [D] float32 skip weight
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

PITFALL — the arguments named B, C, D are TENSORS (input projection,
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

_OP_PROMPTS: dict[str, str] = {
    "forward_chunked_scan": _C1_PROMPT,
}


def build_op_prompt(op_name: str) -> str:
    """Return the generation prompt for *op_name* (KeyError if unknown)."""
    return _OP_PROMPTS[op_name]


def available_ops() -> tuple[str, ...]:
    return tuple(_OP_PROMPTS)
