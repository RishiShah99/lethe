"""Torch-free helpers for sandbox resource-limit tests.

No heavy imports at module level — the POSIX RLIMIT_AS test needs the
child's baseline address space small so the limit constrains the
allocation, not the interpreter startup.
"""


def alloc_8_gib() -> int:
    """Allocate an 8 GiB buffer; under a smaller RLIMIT_AS this raises MemoryError."""
    buf = bytearray(8 * 1024**3)
    return len(buf)


def env_echo(name: str) -> str | None:
    """Return the child process's value of environment variable *name*."""
    import os

    return os.environ.get(name)


def noisy_identity(value: int) -> int:
    """Write to fd 1 (Python print + raw os.write) then return ``value * 2``.

    Stands in for a kernel/CUDA printf or any os-level write to stdout: the
    raw bytes would corrupt the pickle channel if the result shared fd 1.
    The sandbox marshals the result over a private fd and points fd 1 at
    stderr, so the value must still round-trip.
    """
    import os
    import sys

    print("python-level stdout noise")
    sys.stdout.flush()
    os.write(1, b"raw fd-1 bytes that would corrupt a pickle channel\n")
    return value * 2
