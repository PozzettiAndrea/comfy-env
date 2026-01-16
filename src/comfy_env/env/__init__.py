"""Environment management for comfyui-isolation."""

from .config import IsolatedEnv, ToolConfig
from .manager import IsolatedEnvManager
from .cuda_gpu_detection import (
    GPUInfo,
    CUDAEnvironment,
    detect_cuda_environment,
    detect_cuda_version,
    detect_gpu_info,
    detect_gpus,
    get_gpu_summary,
    get_recommended_cuda_version,
)
from .platform import get_platform, PlatformProvider, PlatformPaths
from .security import (
    normalize_env_name,
    validate_dependency,
    validate_dependencies,
    validate_path_within_root,
    validate_wheel_url,
)

__all__ = [
    "IsolatedEnv",
    "IsolatedEnvManager",
    "ToolConfig",
    # GPU Detection
    "GPUInfo",
    "CUDAEnvironment",
    "detect_cuda_environment",
    "detect_cuda_version",
    "detect_gpu_info",
    "detect_gpus",
    "get_gpu_summary",
    "get_recommended_cuda_version",
    # Platform
    "get_platform",
    "PlatformProvider",
    "PlatformPaths",
    # Security
    "normalize_env_name",
    "validate_dependency",
    "validate_dependencies",
    "validate_path_within_root",
    "validate_wheel_url",
]
