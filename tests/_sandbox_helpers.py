"""Torch-free helpers for sandbox resource-limit tests.

No heavy imports at module level — the POSIX RLIMIT_AS test needs the
child's baseline address space small so the limit constrains the
allocation, not the interpreter startup.
"""


def alloc_8_gib() -> int:
    """Allocate an 8 GiB buffer; under a smaller RLIMIT_AS this raises MemoryError."""
    buf = bytearray(8 * 1024**3)
    return len(buf)
