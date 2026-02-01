"""
Detection layer - Pure functions for system detection.

No side effects. These functions gather information about the runtime environment.
"""

from .cuda import (
    CUDA_VERSION_ENV_VAR,
    detect_cuda_version,
    get_cuda_from_torch,
    get_cuda_from_nvml,
    get_cuda_from_nvcc,
    get_cuda_from_env,
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
from .platform import (
    PlatformInfo,
    detect_platform,
    get_platform_tag,
    get_pixi_platform,
    get_library_extension,
    get_executable_suffix,
    is_linux,
    is_windows,
    is_macos,
)
from .runtime import (
    RuntimeEnv,
    detect_runtime,
    parse_wheel_requirement,
)

__all__ = [
    # CUDA detection
    "CUDA_VERSION_ENV_VAR",
    "detect_cuda_version",
    "get_cuda_from_torch",
    "get_cuda_from_nvml",
    "get_cuda_from_nvcc",
    "get_cuda_from_env",
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
    # Platform detection
    "PlatformInfo",
    "detect_platform",
    "get_platform_tag",
    "get_pixi_platform",
    "get_library_extension",
    "get_executable_suffix",
    "is_linux",
    "is_windows",
    "is_macos",
    # Runtime detection
    "RuntimeEnv",
    "detect_runtime",
    "parse_wheel_requirement",
]
