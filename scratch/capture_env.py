"""Capture the GPU box environment -> out/env_box.json.

Run on the box (`uv run python scratch/capture_env.py`), pull back with
`fleet pull`, then curate into results/ locally. This file is the
version-pin record the #904 reproduction depends on.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def nvidia_smi() -> dict[str, Any]:
    try:
        query = "name,driver_version,memory.total,compute_cap"
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"nvidia_smi_error": str(exc)}
    gpus = [
        dict(
            zip(
                ("name", "driver", "memory", "compute_cap"),
                (f.strip() for f in line.split(",")),
                strict=False,
            )
        )
        for line in out.splitlines()
    ]
    return {"gpus": gpus}


def module_version(name: str) -> str | None:
    try:
        mod = __import__(name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "unknown")


def main() -> None:
    info: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        **nvidia_smi(),
        "torch": module_version("torch"),
        "triton": module_version("triton"),
        "mamba_ssm": module_version("mamba_ssm"),
        "transformers": module_version("transformers"),
    }
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            info["device_0"] = torch.cuda.get_device_name(0)
            info["capability_0"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass

    out_dir = Path(__file__).resolve().parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "env_box.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
