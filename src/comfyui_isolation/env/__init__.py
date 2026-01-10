"""Environment management for comfyui-isolation."""

from .config import IsolatedEnv
from .manager import IsolatedEnvManager
from .detection import detect_cuda_version, detect_gpu_info, get_gpu_summary
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
    "detect_cuda_version",
    "detect_gpu_info",
    "get_gpu_summary",
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
