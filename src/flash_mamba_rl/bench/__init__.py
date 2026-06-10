"""Benchmark runners, result tables, ablation runners, plot generators.

Kernel benchmark modules are CLIs (``python -m flash_mamba_rl.bench.<module>``)
that write JSON reports; on the fleet box, write to ``~/out`` so
``fleet pull`` brings artifacts home.
"""
