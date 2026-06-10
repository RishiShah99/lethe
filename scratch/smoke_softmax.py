"""Triton smoke test 2: row-wise fused softmax.

Correctness vs torch.softmax (fp32 tolerances), then benchmark.
Writes out/smoke_softmax.json.
"""

import json
from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    col_offsets = tl.arange(0, BLOCK)
    mask = col_offsets < n_cols
    x = tl.load(in_ptr + row * in_row_stride + col_offsets, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + col_offsets, num / den, mask=mask)


def fused_softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(n_cols)
    num_warps = 4 if block < 2048 else (8 if block < 8192 else 16)
    softmax_kernel[(n_rows,)](
        out, x, x.stride(0), out.stride(0), n_cols, BLOCK=block, num_warps=num_warps
    )
    return out


def main() -> None:
    torch.manual_seed(0)
    results: dict[str, dict[str, float]] = {}
    for n_rows, n_cols in ((4096, 1024), (8192, 4096), (1024, 8000)):
        x = torch.randn(n_rows, n_cols, device="cuda")
        out = fused_softmax(x)
        ref = torch.softmax(x, dim=-1)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6), (
            f"softmax wrong at ({n_rows},{n_cols}): max_err={(out - ref).abs().max().item():.3e}"
        )
        t_triton = triton.testing.do_bench(partial(fused_softmax, x))
        t_torch = triton.testing.do_bench(partial(torch.softmax, x, dim=-1))
        results[f"{n_rows}x{n_cols}"] = {"triton_ms": t_triton, "torch_ms": t_torch}
        print(f"{n_rows}x{n_cols:>5}  triton {t_triton:.5f} ms   torch {t_torch:.5f} ms")

    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "smoke_softmax.json").write_text(json.dumps(results, indent=2))
    print("OK: softmax smoke passed")


if __name__ == "__main__":
    main()
