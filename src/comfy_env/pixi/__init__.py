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
from .resolver import RuntimeEnv

__all__ = [
    # Core pixi functions
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_run",
    "pixi_install",
    "clean_pixi_artifacts",
    "CUDA_WHEELS_INDEX",
    # CUDA detection
    "detect_cuda_version",
    "detect_cuda_environment",
    "detect_gpu_info",
    "detect_gpus",
    "get_gpu_summary",
    "get_recommended_cuda_version",
    "GPUInfo",
    "CUDAEnvironment",
    # Resolver
    "RuntimeEnv",
]
