"""
Pixi integration for comfy-env.

All dependencies go through pixi for unified management.
"""

from .core import (
    ensure_pixi,
    get_pixi_path,
    get_pixi_python,
    pixi_run,
    pixi_install,
    clean_pixi_artifacts,
    CUDA_WHEELS_INDEX,
)
from .registry import PACKAGE_REGISTRY
from .cuda_detection import (
    detect_cuda_version,
    detect_cuda_environment,
    detect_gpu_info,
    detect_gpus,
    get_gpu_summary,
    get_recommended_cuda_version,
    GPUInfo,
    CUDAEnvironment,
)

__all__ = [
    # Core pixi functions
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_run",
    "pixi_install",
    "clean_pixi_artifacts",
    "CUDA_WHEELS_INDEX",
    # Registry
    "PACKAGE_REGISTRY",
    # CUDA detection
    "detect_cuda_version",
    "detect_cuda_environment",
    "detect_gpu_info",
    "detect_gpus",
    "get_gpu_summary",
    "get_recommended_cuda_version",
    "GPUInfo",
    "CUDAEnvironment",
]
