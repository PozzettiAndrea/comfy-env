"""CUDA version detection. Priority: env -> torch -> nvcc"""

from __future__ import annotations

import os
import re
import subprocess

CUDA_VERSION_ENV_VAR = "COMFY_ENV_CUDA_VERSION"


def detect_cuda_version() -> str | None:
    """Detect CUDA version from available sources."""
    return get_cuda_from_env() or get_cuda_from_torch() or get_cuda_from_nvcc()


def get_cuda_from_env() -> str | None:
    """Get CUDA version from environment variable override."""
    override = os.environ.get(CUDA_VERSION_ENV_VAR, "").strip()
    if not override:
        return None
    if "." not in override and len(override) >= 2:
        return f"{override[:-1]}.{override[-1]}"
    return override


def get_cuda_from_torch() -> str | None:
    """Get CUDA version from PyTorch."""
    try:
        import torch
        if torch.cuda.is_available() and torch.version.cuda:
            return torch.version.cuda
    except Exception:
        pass
    return None


def get_cuda_from_nvml() -> str | None:
    """Get CUDA version from NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            return f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return None


def get_cuda_from_nvcc() -> str | None:
    """Get CUDA version from nvcc compiler."""
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and (m := re.search(r"release (\d+\.\d+)", r.stdout)):
            return m.group(1)
    except Exception:
        pass
    return None
