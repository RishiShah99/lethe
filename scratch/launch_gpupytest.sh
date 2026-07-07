#!/usr/bin/env bash
# Detached GPU pytest from the fresh ~/lethe deploy (guard must not regress GPU paths).
mkdir -p ~/box_out_verifier
cd ~/lethe || exit 1
export PYTHONPATH=src:.
export TORCH_CUDA_ARCH_LIST=10.0
find tests -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
rm -f ~/box_out_verifier/PYTEST_DONE
nohup bash -c '~/cuteenv/bin/python -m pytest tests -q -p no:cacheprovider > ~/box_out_verifier/gpu_pytest.log 2>&1; echo "exit=$?" > ~/box_out_verifier/PYTEST_DONE' &
echo "launched pytest pid=$!"
