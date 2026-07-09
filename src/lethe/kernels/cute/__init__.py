"""Native Blackwell (sm_100) CuTe / tcgen05 kernels (Phase 2/3).

Currently the GDN-2 training backward. Modules here load a compiled sm_100 kernel
lazily and expose a ``native_*`` entry point that returns ``None`` when the kernel
is unavailable, so callers fall back to the oracle-faithful eager path.
"""
