"""CUDA version detection. Priority: env -> torch -> nvcc"""

from __future__ import annotations

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
    return get_cuda_from_torch() or get_cuda_from_nvcc()


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


def get_bootstrap_python_version() -> str:
    """Python version of the interpreter running comfy-env install (e.g. '3.10')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_bootstrap_torch_version() -> str | None:
    """Public torch version available in the bootstrap interpreter (e.g. '2.11.0').

    Priority: COMFY_ENV_TORCH_VERSION env var -> `import torch; torch.__version__`.
    Returns None if torch isn't importable from the bootstrap.
    """
    override = os.environ.get(TORCH_VERSION_ENV_VAR, "").strip()
    if override:
        return override
    try:
        import torch
        v = getattr(torch, "__version__", None)
        if not v:
            return None
        # Strip local label (e.g. "2.11.0+cu128" -> "2.11.0")
        return str(v).split("+", 1)[0]
    except Exception:
        return None


def get_bootstrap_torch_cuda() -> str | None:
    """CUDA version the bootstrap torch was built against (e.g. '12.8').

    Read from `torch.version.cuda`. Returns None if torch isn't importable from
    the bootstrap or it's a CPU-only build. This is the authoritative answer
    when present -- it's what the bootstrap torch's binary ABI actually targets,
    independent of what nvidia-smi reports.
    """
    try:
        import torch
        cu = getattr(torch.version, "cuda", None)
        return str(cu) if cu else None
    except Exception:
        return None
