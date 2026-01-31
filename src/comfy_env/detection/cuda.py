"""
CUDA version detection with multiple fallback methods.

Detection priority: PyTorch -> nvcc -> env vars
"""

from __future__ import annotations

import os
import re
import subprocess

CUDA_VERSION_ENV_VAR = "COMFY_ENV_CUDA_VERSION"


def detect_cuda_version() -> str | None:
    """
    Detect CUDA version from available sources.

    Returns:
        CUDA version string (e.g., "12.8") or None if not detected.
    """
    # Check environment override first
    if version := get_cuda_from_env():
        return version

    # Try PyTorch
    if version := get_cuda_from_torch():
        return version

    # Try nvcc
    if version := get_cuda_from_nvcc():
        return version

    return None


def get_cuda_from_env() -> str | None:
    """Get CUDA version from environment variable override."""
    override = os.environ.get(CUDA_VERSION_ENV_VAR, "").strip()
    if not override:
        return None
    # Handle formats like "128" -> "12.8"
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
    """Get CUDA version from NVML (driver API version)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            # NVML reports driver CUDA version, not runtime
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            return f"{major}.{minor}"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return None


def get_cuda_from_nvcc() -> str | None:
    """Get CUDA version from nvcc compiler."""
    try:
        r = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if r.returncode == 0:
            if m := re.search(r"release (\d+\.\d+)", r.stdout):
                return m.group(1)
    except Exception:
        pass
    return None


def get_cuda_from_nvidia_smi() -> str | None:
    """Get CUDA version from nvidia-smi (driver-supported version)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if r.returncode == 0:
            # nvidia-smi shows driver-supported CUDA, not installed runtime
            # This is less reliable for our purposes
            pass
    except Exception:
        pass
    return None
