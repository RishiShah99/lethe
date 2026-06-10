"""Triton smoke test 1: vector add.

Exact-match correctness vs torch.add, then benchmark. Writes
out/smoke_vector_add.json. Failure here = the box/toolchain is broken,
not the kernel.
"""

import json
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 1024


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    vector_add_kernel[grid](x, y, out, n, BLOCK=BLOCK_SIZE)
    return out


def main() -> None:
    torch.manual_seed(0)
    results: dict[str, dict[str, float]] = {}
    for n in (1 << 12, 1 << 20, 1 << 26):
        x = torch.rand(n, device="cuda")
        y = torch.rand(n, device="cuda")
        out = vector_add(x, y)
        ref = x + y
        max_err = (out - ref).abs().max().item()
        assert max_err == 0.0, f"vector_add wrong at n={n}: max_err={max_err}"
        t_triton = triton.testing.do_bench(partial(vector_add, x, y))
        t_torch = triton.testing.do_bench(partial(torch.add, x, y))
        results[f"n={n}"] = {"triton_ms": t_triton, "torch_ms": t_torch}
        print(f"n={n:>9}  triton {t_triton:.5f} ms   torch {t_torch:.5f} ms")

    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "smoke_vector_add.json").write_text(json.dumps(results, indent=2))
    print("OK: vector_add smoke passed")


if __name__ == "__main__":
    main()
