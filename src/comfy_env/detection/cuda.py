"""CUDA version detection. Priority: env -> nvml -> libcuda -> nvcc -> torch

NVML and libcuda detection use ctypes (same approach as rattler/pixi) to avoid
importing torch during ComfyUI prestartup.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
import sys

CUDA_VERSION_ENV_VAR = "COMFY_ENV_CUDA_VERSION"
TORCH_VERSION_ENV_VAR = "COMFY_ENV_TORCH_VERSION"


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU hardware is present without loading CUDA/torch."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return True
    except Exception:
        pass
    if sys.platform == "linux":
        from pathlib import Path
        try:
            for d in Path("/sys/bus/pci/devices").iterdir():
                vendor = (d / "vendor").read_text().strip().lower() if (d / "vendor").exists() else ""
                if "10de" in vendor:
                    cls = (d / "class").read_text().strip() if (d / "class").exists() else ""
                    if cls.startswith("0x0300") or cls.startswith("0x0302"):
                        return True
        except Exception:
            pass
    return False


def detect_cuda_version() -> str | None:
    """Detect CUDA version from available sources."""
    if env_cuda := get_cuda_from_env():
        return env_cuda
    if not has_nvidia_gpu():
        return None
    return (get_cuda_from_nvml() or get_cuda_from_libcuda()
            or get_cuda_from_nvcc() or get_cuda_from_torch())


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
    """Get CUDA version via NVML (ctypes, same approach as rattler/pixi)."""
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("nvml.dll")
        else:
            path = ctypes.util.find_library("nvidia-ml") or "libnvidia-ml.so.1"
            lib = ctypes.CDLL(path)
        if lib.nvmlInit_v2() != 0:
            return None
        try:
            cuda_ver = ctypes.c_int()
            if lib.nvmlSystemGetCudaDriverVersion_v2(ctypes.byref(cuda_ver)) != 0:
                return None
            v = cuda_ver.value
            return f"{v // 1000}.{(v % 1000) // 10}"
        finally:
            lib.nvmlShutdown()
    except Exception:
        return None


def get_cuda_from_libcuda() -> str | None:
    """Get CUDA version via libcuda (ctypes, same approach as rattler/pixi)."""
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("nvcuda.dll")
        else:
            path = ctypes.util.find_library("cuda") or "libcuda.so.1"
            lib = ctypes.CDLL(path)
        if lib.cuInit(0) != 0:
            return None
        ver = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(ver)) != 0:
            return None
        v = ver.value
        return f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:
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


def get_bootstrap_python_version() -> str:
    """Python version of the interpreter running comfy-env install (e.g. '3.10')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_bootstrap_torch_version() -> str | None:
    """Public torch version available in the bootstrap interpreter (e.g. '2.11.0').

    Priority: COMFY_ENV_TORCH_VERSION env var -> importlib.metadata.
    Uses metadata to avoid importing torch (which pollutes sys.modules during
    ComfyUI prestartup and triggers a spurious warning).
    """
    override = os.environ.get(TORCH_VERSION_ENV_VAR, "").strip()
    if override:
        return override
    try:
        from importlib.metadata import version
        v = version("torch")
        # Strip local label (e.g. "2.11.0+cu128" -> "2.11.0")
        return str(v).split("+", 1)[0]
    except Exception:
        return None


def get_bootstrap_torch_cuda() -> str | None:
    """CUDA version the bootstrap torch was built against (e.g. '12.8').

    Parsed from the torch package version's local label (e.g. '2.5.0+cu128' ->
    '12.8'). Uses importlib.metadata to avoid importing torch.
    """
    try:
        from importlib.metadata import version
        v = version("torch")
        # Parse local label: "2.5.0+cu128" -> "cu128" -> "12.8"
        if "+" not in v:
            return None
        local = v.split("+", 1)[1]
        if not local.startswith("cu"):
            return None
        cu_digits = local[2:]  # e.g. "128"
        if len(cu_digits) >= 2:
            return f"{cu_digits[:-1]}.{cu_digits[-1]}"
        return None
    except Exception:
        return None
