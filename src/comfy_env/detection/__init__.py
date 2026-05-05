"""
Detection layer - Pure functions for system detection.

No side effects. These functions gather information about the runtime environment.
"""

from .cuda import (
    has_nvidia_gpu,
    detect_cuda_version,
    get_bootstrap_python_version,
    get_bootstrap_torch_version,
    get_bootstrap_torch_cuda,
)
from .gpu import (
    GPUInfo,
    CUDAEnvironment,
    COMPUTE_TO_ARCH,
    detect_gpu,
    detect_gpus,
    detect_cuda_environment,
    get_compute_capability,
    compute_capability_to_architecture,
    get_recommended_cuda_version,
    get_gpu_summary,
)

# Platform helpers (minimal, inlined from former platform.py)
import platform as _platform_module
import sys as _sys

_PIXI_PLATFORMS = {
    ("linux", "x86_64"): "linux-64",
    ("linux", "aarch64"): "linux-aarch64",
    ("darwin", "x86_64"): "osx-64",
    ("darwin", "arm64"): "osx-arm64",
    ("windows", "amd64"): "win-64",
    ("windows", "x86_64"): "win-64",
}


def _get_os_name() -> str:
    if _sys.platform.startswith("linux"):
        return "linux"
    elif _sys.platform == "darwin":
        return "darwin"
    elif _sys.platform == "win32":
        return "windows"
    return _sys.platform


def get_pixi_platform() -> str:
    """Get pixi platform string (e.g. 'linux-64', 'osx-arm64')."""
    key = (_get_os_name(), _platform_module.machine().lower())
    return _PIXI_PLATFORMS.get(key, f"{key[0]}-{key[1]}")


def is_linux() -> bool:
    return _sys.platform.startswith("linux")


def is_windows() -> bool:
    return _sys.platform == "win32"


def is_macos() -> bool:
    return _sys.platform == "darwin"


__all__ = [
    # CUDA detection
    "has_nvidia_gpu",
    "detect_cuda_version",
    "get_bootstrap_python_version",
    "get_bootstrap_torch_version",
    "get_bootstrap_torch_cuda",
    # GPU detection
    "GPUInfo",
    "CUDAEnvironment",
    "COMPUTE_TO_ARCH",
    "detect_gpu",
    "detect_gpus",
    "detect_cuda_environment",
    "get_compute_capability",
    "compute_capability_to_architecture",
    "get_recommended_cuda_version",
    "get_gpu_summary",
    # Platform helpers
    "get_pixi_platform",
    "is_linux",
    "is_windows",
    "is_macos",
]
