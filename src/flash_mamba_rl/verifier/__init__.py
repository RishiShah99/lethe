"""Contract-grounded verifier: compile sandbox, 12 Kernel-Contract gates, timing, staged reward."""

from .compile import CompileResult, ErrorClass, compile_kernel
from .contracts import (
    GateResult,
    gate_cmp_01_input_variation,
    gate_cmp_02_gradient_correctness,
    gate_cmp_03_shape_polymorphism,
    gate_exc_01_exceptional_values,
    gate_exc_02_subnormal_handling,
    gate_ord_01_reduction_order_tolerance,
    gate_ord_02_atomic_determinism,
    gate_ord_03_noncommutative_reduction,
    gate_prc_01_precision_regime,
    gate_prc_02_mixed_precision_accumulation,
    gate_res_01_memory_residency,
    gate_res_02_resource_limits,
    run_all_gates,
)
from .op_harness import (
    scan_candidate_adapter,
    scan_reference_adapter,
    verify_scan_op,
)
from .reward import compute_reward
from .sandbox import SubprocessResult, run_in_subprocess
from .timing import TimingResult, benchmark

__all__ = [
    "CompileResult",
    "ErrorClass",
    "GateResult",
    "SubprocessResult",
    "TimingResult",
    "benchmark",
    "compile_kernel",
    "compute_reward",
    "gate_cmp_01_input_variation",
    "gate_cmp_02_gradient_correctness",
    "gate_cmp_03_shape_polymorphism",
    "gate_exc_01_exceptional_values",
    "gate_exc_02_subnormal_handling",
    "gate_ord_01_reduction_order_tolerance",
    "gate_ord_02_atomic_determinism",
    "gate_ord_03_noncommutative_reduction",
    "gate_prc_01_precision_regime",
    "gate_prc_02_mixed_precision_accumulation",
    "gate_res_01_memory_residency",
    "gate_res_02_resource_limits",
    "run_all_gates",
    "run_in_subprocess",
    "scan_candidate_adapter",
    "scan_reference_adapter",
    "verify_scan_op",
]
