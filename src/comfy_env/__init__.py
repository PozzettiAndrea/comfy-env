"""
comfy-env - Environment management for ComfyUI custom nodes.

Features:
- CUDA wheel resolution (pre-built wheels without compilation)
- Process isolation (run nodes in separate Python environments)
- Central environment cache (~/.comfy-env/envs/)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfy-env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


# =============================================================================
# Primary API (what most users need)
# =============================================================================

# Install API
from .install import install, verify_installation, USE_COMFY_ENV_VAR

# Prestartup helpers
from .environment.setup import setup_env
from .environment.paths import copy_files

# Isolation
from .isolation import wrap_isolated_nodes, wrap_nodes


# =============================================================================
# Config Layer
# =============================================================================

from .config import (
    ComfyEnvConfig,
    NodeDependency,
    NodeReq,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)


# =============================================================================
# Detection Layer
# =============================================================================

from .detection import (
    # CUDA detection
    detect_cuda_version,
    detect_cuda_environment,
    get_recommended_cuda_version,
    # GPU detection
    GPUInfo,
    CUDAEnvironment,
    detect_gpu,
    get_gpu_summary,
    # Platform detection
    detect_platform,
    get_platform_tag,
    # Runtime detection
    RuntimeEnv,
    detect_runtime,
)


# =============================================================================
# Packages Layer
# =============================================================================

from .packages import (
    # Pixi
    ensure_pixi,
    get_pixi_path,
    get_pixi_python,
    pixi_run,
    pixi_clean,
    # CUDA wheels
    CUDA_WHEELS_INDEX,
    get_wheel_url,
    get_cuda_torch_mapping,
)


# =============================================================================
# Environment Layer
# =============================================================================

from .environment import (
    # Cache management
    get_cache_dir,
    cleanup_orphaned_envs,
    resolve_env_path,
    CACHE_DIR,
    MARKER_FILE,
)


# =============================================================================
# Isolation Layer
# =============================================================================

from .isolation import (
    # Workers
    Worker,
    WorkerError,
    MPWorker,
    SubprocessWorker,
    # Tensor utilities
    TensorKeeper,
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Install API
    "install",
    "verify_installation",
    "USE_COMFY_ENV_VAR",
    # Prestartup
    "setup_env",
    "copy_files",
    # Isolation
    "wrap_isolated_nodes",
    "wrap_nodes",
    # Config
    "ComfyEnvConfig",
    "NodeDependency",
    "NodeReq",
    "load_config",
    "discover_config",
    "CONFIG_FILE_NAME",
    "ROOT_CONFIG_FILE_NAME",
    # Detection
    "detect_cuda_version",
    "detect_cuda_environment",
    "get_recommended_cuda_version",
    "GPUInfo",
    "CUDAEnvironment",
    "detect_gpu",
    "get_gpu_summary",
    "detect_platform",
    "get_platform_tag",
    "RuntimeEnv",
    "detect_runtime",
    # Packages
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_run",
    "pixi_clean",
    "CUDA_WHEELS_INDEX",
    "get_wheel_url",
    "get_cuda_torch_mapping",
    # Environment
    "get_cache_dir",
    "cleanup_orphaned_envs",
    "resolve_env_path",
    "CACHE_DIR",
    "MARKER_FILE",
    # Workers
    "Worker",
    "WorkerError",
    "MPWorker",
    "SubprocessWorker",
    "TensorKeeper",
]


# =============================================================================
# Startup cleanup
# =============================================================================

def _run_startup_cleanup():
    """Clean orphaned envs on startup."""
    try:
        cleanup_orphaned_envs(log=lambda x: None)  # Silent
    except Exception:
        pass  # Never fail startup due to cleanup

_run_startup_cleanup()
