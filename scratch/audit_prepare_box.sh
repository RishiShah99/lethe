#!/usr/bin/env bash
# Box-side audit prep: download the Dr. Kernel parquet + build the manifest.
# Usage: fleet run "bash scratch/detach.sh bash scratch/audit_prepare_box.sh"
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "=== audit prepare: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
curl -sL --retry 5 --retry-all-errors -o scratch/drkernel_coldstart.parquet \
  "https://huggingface.co/api/datasets/hkust-nlp/drkernel-coldstart-8k/parquet/default/train/0.parquet"
ls -la scratch/drkernel_coldstart.parquet
uv run --with pyarrow python scratch/audit_extract_drkernel.py \
  scratch/drkernel_coldstart.parquet scratch/audit_manifest_drkernel.jsonl.gz
ls -la scratch/audit_manifest_drkernel.jsonl.gz
echo "=== audit prepare done ==="
