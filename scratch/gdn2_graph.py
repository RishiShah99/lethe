"""#48 — CUDA-graph capture/replay wrapper for the fixed-shape cw GDN-2 backward.

The cw backward fires an IDENTICAL launch sequence for a given shape: forward re-stage +
stage-B (torch/cublas) → K#1 reverse scan + K#2 WY-VJP (tcgen05, stream-threaded by #47) →
chunk assembly + L2-norm VJP. #47 proved the tcgen05 launches are capture-safe on torch's
stream; graph_loop_probe proved a multi-launch loop + glue + carry captures. This wraps a
fixed-shape backward callable so ALL host dispatch + the per-chunk Python loop + torch glue
collapse into one ``graph.replay()`` -- leaving only device time (the 23-170x host-orchestration
gap vs fla).

Pattern (canonical torch manual capture): clone inputs into STATIC buffers, warm up on a SIDE
stream (so cublas/cute workspaces + the cute.compile caches are populated and NOT captured),
then capture the call under :func:`graph_capture` (which no-ops the per-kernel
``torch.cuda.synchronize`` in the cw run fns). Replays copy_ fresh inputs into the static
buffers and replay; outputs live in the graph pool, so each replay overwrites them — callers
get cloned copies.

Box-only (the cute kernels need sm_100). Imports cleanly off-box.
"""

from collections.abc import Callable

import torch
from scratch.gdn2_bwd_dhu import graph_capture
from torch import Tensor


class GraphedBackward:
    """Capture a fixed-shape backward ``fn(*tensors) -> tuple[Tensor, ...]`` into a CUDA graph.

    One graph is cached per distinct ``(shape, dtype)`` tuple of the inputs. ``fn`` must take
    only tensor args and return a tuple of tensors (wrap a dataclass-returning backward in a
    small adapter). All input/output shapes must be static across replays (true for a fixed
    GDN-2 shape — verified in the #48 call-graph map).
    """

    def __init__(self, fn: Callable[..., tuple[Tensor, ...]], warmup: int = 3) -> None:
        self.fn = fn
        self.warmup = warmup
        self._cache: dict[
            tuple[object, ...], tuple[list[Tensor], torch.cuda.CUDAGraph, list[Tensor]]
        ] = {}

    @staticmethod
    def _key(tensors: tuple[Tensor, ...]) -> tuple[object, ...]:
        return tuple((tuple(t.shape), t.dtype) for t in tensors)

    def _build(
        self, tensors: tuple[Tensor, ...]
    ) -> tuple[list[Tensor], "torch.cuda.CUDAGraph", list[Tensor]]:
        static_in = [t.detach().clone() for t in tensors]

        # Warm up on a side stream: runs the eager backward a few times so every cute.compile
        # cache + cublas workspace is allocated BEFORE capture (a workspace alloc inside capture
        # would abort it). The side stream + wait_stream is the canonical torch graph recipe.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self.warmup):
                self.fn(*static_in)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with graph_capture(), torch.cuda.graph(graph):
            out = self.fn(*static_in)
        static_out = list(out)
        return static_in, graph, static_out

    def __call__(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        key = self._key(tensors)
        entry = self._cache.get(key)
        if entry is None:
            entry = self._build(tensors)
            self._cache[key] = entry
        static_in, graph, static_out = entry
        for s, t in zip(static_in, tensors, strict=True):
            s.copy_(t)
        graph.replay()
        torch.cuda.synchronize()
        return tuple(o.clone() for o in static_out)

    def replay_only(self) -> None:
        """Replay the single cached graph with whatever is in the static input buffers.

        For timing the pure device-bound cost (caller stages inputs + syncs around a batch of
        replays). Assumes exactly one shape has been captured.
        """
        (_static_in, graph, _static_out) = next(iter(self._cache.values()))
        graph.replay()
