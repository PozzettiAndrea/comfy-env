"""
comfy-env: Environment management for ComfyUI custom nodes.

All dependencies go through pixi for unified management.

Main APIs:
- install(): Install dependencies from comfy-env.toml
- wrap_isolated_nodes(): Wrap nodes for subprocess isolation
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfy-env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Config types and parsing
from .config import (
    ComfyEnvConfig,
    NodeReq,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
)

# Pixi integration
from .pixi import (
    ensure_pixi,
    get_pixi_path,
    get_pixi_python,
    pixi_run,
    pixi_install,
    CUDA_WHEELS_INDEX,
    detect_cuda_version,
    detect_cuda_environment,
    get_recommended_cuda_version,
    GPUInfo,
    CUDAEnvironment,
)

# Workers
from .workers import (
    Worker,
    WorkerError,
    MPWorker,
    SubprocessWorker,
)

# Isolation
from .isolation import wrap_isolated_nodes

# Install API
from .install import install, verify_installation

# Prestartup helpers
from .prestartup import setup_env

# Cache management
from .cache import (
    get_cache_dir,
    cleanup_orphaned_envs,
    resolve_env_path,
    CACHE_DIR,
    MARKER_FILE,
)

# Errors
from .errors import (
    EnvManagerError,
    ConfigError,
    WheelNotFoundError,
    DependencyError,
    CUDANotFoundError,
    InstallError,
)

__all__ = [
    # Install API
    "install",
    "verify_installation",
    # Prestartup
    "setup_env",
    # Isolation
    "wrap_isolated_nodes",
    # Config
    "ComfyEnvConfig",
    "NodeReq",
    "load_config",
    "discover_config",
    "CONFIG_FILE_NAME",
    # Pixi
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_run",
    "pixi_install",
    "CUDA_WHEELS_INDEX",
    # CUDA detection
    "detect_cuda_version",
    "detect_cuda_environment",
    "get_recommended_cuda_version",
    "GPUInfo",
    "CUDAEnvironment",
    # Workers
    "Worker",
    "WorkerError",
    "MPWorker",
    "SubprocessWorker",
    # Errors
    "EnvManagerError",
    "ConfigError",
    "WheelNotFoundError",
    "DependencyError",
    "CUDANotFoundError",
    "InstallError",
    # Cache
    "get_cache_dir",
    "cleanup_orphaned_envs",
    "resolve_env_path",
    "CACHE_DIR",
    "MARKER_FILE",
]

# Run orphan cleanup once on module load (silently)
def _run_startup_cleanup():
    """Clean orphaned envs on startup. Runs silently, never fails startup."""
    try:
        cleanup_orphaned_envs(log=lambda x: None)  # Silent
    except Exception:
        pass  # Never fail startup due to cleanup

_run_startup_cleanup()
