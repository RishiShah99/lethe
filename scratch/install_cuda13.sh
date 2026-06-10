#!/usr/bin/env bash
# Install CUDA toolkit 13.0 on the box. The image ships nvcc 12.9 but PyPI
# torch 2.12 is built for CUDA 13.0, and causal-conv1d/mamba-ssm builds
# refuse a toolkit/torch version mismatch. Driver 580.x supports CUDA 13.
set -euo pipefail

if [ -x /usr/local/cuda-13.0/bin/nvcc ]; then
  echo ">> cuda-13.0 already present"
else
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -q 2>&1 | tail -1
  sudo apt-get install -y -q cuda-toolkit-13-0 2>&1 | tail -2
fi

/usr/local/cuda-13.0/bin/nvcc --version | tail -1
echo ">> install_cuda13 complete"
