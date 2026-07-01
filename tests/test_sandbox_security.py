"""Regression for the CRITICAL verifier-sandbox pickle-RCE.

The child process runs untrusted candidate code; the parent runs the reward /
GO-verdict logic. A bare ``pickle.loads`` of the child's return value let a
reward-hacking candidate execute code in the parent via a ``__reduce__`` gadget
(forging rewards/verdicts). The fix serializes the result with ``torch.save`` and
loads it with ``torch.load(weights_only=True)``; these tests pin that a gadget is
refused with no side effect while legitimate outputs still round-trip.
"""

from __future__ import annotations

import io
import os
import pickle
import tempfile

import pytest
import torch

from flash_mamba_rl.verifier.sandbox import _deserialize_child_output, run_in_subprocess
from tests._sandbox_helpers import ReduceBomb


def _fresh_sentinel_path() -> str:
    fd, path = tempfile.mkstemp(prefix="fmrl_rce_sentinel_", suffix=".txt")
    os.close(fd)
    os.unlink(path)  # must not exist unless the gadget's reconstruction fires
    return path


def _torch_saved(obj: object) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf, pickle_protocol=2)
    return buf.getvalue()


def test_result_fd_swap_precedes_untrusted_deserialize() -> None:
    # The child must repoint fd 1 -> stderr BEFORE the stdin read, the task
    # unpickle, and the heavy imports: a C-level write to fd 1 during any of
    # those would prepend stray bytes to the torch.save result payload.
    from flash_mamba_rl.verifier.sandbox import _WORKER_SCRIPT

    swap = _WORKER_SCRIPT.index("os.dup2(2, 1)")
    for later in ("import importlib", "sys.stdin.buffer.read()", "pickle.loads(", "import torch"):
        assert swap < _WORKER_SCRIPT.index(later), f"fd-swap must precede: {later}"


class TestSafeDeserialize:
    def test_torch_saved_gadget_refused_no_side_effect(self) -> None:
        path = _fresh_sentinel_path()
        payload = _torch_saved(ReduceBomb(path))
        with pytest.raises(pickle.UnpicklingError):
            _deserialize_child_output(payload)
        assert not os.path.exists(path), "gadget reconstruction fired in the parent"

    def test_raw_pickle_gadget_refused_no_side_effect(self) -> None:
        # The exact bytes the OLD vulnerable path (pickle.loads) would have run.
        path = _fresh_sentinel_path()
        with pytest.raises(pickle.UnpicklingError):
            _deserialize_child_output(pickle.dumps(ReduceBomb(path), protocol=2))
        assert not os.path.exists(path)

    def test_gadget_is_genuinely_dangerous_under_bare_pickle(self) -> None:
        # Proves the payload is a real RCE proxy: bare pickle.loads DOES run it,
        # which is exactly the behaviour the safe loader removes from the parent.
        path = _fresh_sentinel_path()
        pickle.loads(pickle.dumps(ReduceBomb(path)))
        assert os.path.exists(path)
        os.unlink(path)

    def test_legitimate_outputs_round_trip(self) -> None:
        def rt(obj: object) -> object:
            return _deserialize_child_output(_torch_saved(obj))

        assert rt(42) == 42
        assert rt("7") == "7"
        assert rt({"status": "scored", "reward": 1.0, "speedup": None}) == {
            "status": "scored",
            "reward": 1.0,
            "speedup": None,
        }
        assert torch.equal(rt(torch.arange(4)), torch.arange(4))


class TestEndToEndSandbox:
    def test_reduce_bomb_never_fires_in_parent(self) -> None:
        path = _fresh_sentinel_path()
        res = run_in_subprocess(
            "tests._sandbox_helpers",
            "return_reduce_bomb",
            (path,),
            timeout_s=120.0,
        )
        assert res.success is False, "malicious return value must classify as failure"
        assert not os.path.exists(path), "gadget executed in the verifier parent"

    def test_benign_result_still_round_trips(self) -> None:
        res = run_in_subprocess(
            "tests._sandbox_helpers",
            "noisy_identity",
            (21,),
            timeout_s=120.0,
        )
        assert res.success is True, res.stderr
        assert res.output == 42
