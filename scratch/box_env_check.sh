#!/bin/bash
# One-shot box health probe (run on the box, so heredoc quoting is local-only).
cd "$HOME/lethe" || exit 1
export PATH=$HOME/.local/bin:$PATH
echo "REPO files=$(ls | wc -l)"
uv run --no-sync python - <<'PY'
import torch, triton
print("torch", torch.__version__, "triton", triton.__version__)
try:
    import mamba_ssm
    print("mamba", mamba_ssm.__version__)
except Exception as e:
    print("mamba MISSING:", type(e).__name__, e)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
echo "HFCACHE=$(du -sh $HOME/.cache/huggingface 2>/dev/null | cut -f1)"
echo "ADAPTERS=$(ls -d sft_out_v2 sft_out 2>/dev/null | tr '\n' ' ')"
echo "ENVCHK_END"
