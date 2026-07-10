"""Torch-free helpers for sandbox resource-limit tests."""


def alloc_8_gib() -> int:
    """Allocate an 8 GiB buffer; under a smaller RLIMIT_AS this raises MemoryError."""
    buf = bytearray(8 * 1024**3)
    return len(buf)


def env_echo(name: str) -> str | None:
    """Return the child process's value of environment variable *name*."""
    import os

    return os.environ.get(name)


def noisy_identity(value: int) -> int:
    """Write to fd 1 (Python print + raw os.write) then return ``value * 2``."""
    import os
    import sys

    print("python-level stdout noise")
    sys.stdout.flush()
    os.write(1, b"raw fd-1 bytes that would corrupt a pickle channel\n")
    return value * 2


def _write_sentinel(path: str) -> int:
    """Reconstruction side effect a ``__reduce__`` gadget runs in the UNPICKLER."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("pwned")
    return 0


class ReduceBomb:
    """A return value whose *reconstruction* writes a sentinel, a benign RCE proxy."""

    def __init__(self, sentinel_path: str) -> None:
        self.sentinel_path = sentinel_path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_write_sentinel, (self.sentinel_path,))


def return_reduce_bomb(sentinel_path: str) -> ReduceBomb:
    """Sandbox worker entry: return an object that would run code when deserialized."""
    return ReduceBomb(sentinel_path)
