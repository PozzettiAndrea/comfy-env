"""Environment management for ComfyUI custom nodes."""

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
from .isolation import wrap_isolated_nodes, wrap_nodes

# Install API
from .install import install, verify_installation, USE_COMFY_ENV_VAR

# Prestartup helpers
from .prestartup import setup_env, copy_files

# Cache management
from .cache import (
    get_cache_dir,
    cleanup_orphaned_envs,
    resolve_env_path,
    CACHE_DIR,
    MARKER_FILE,
)

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
    # Cache
    "get_cache_dir",
    "cleanup_orphaned_envs",
    "resolve_env_path",
    "CACHE_DIR",
    "MARKER_FILE",
]

# Run orphan cleanup once on module load (silently)
def _run_startup_cleanup():
    """Clean orphaned envs on startup."""
    try:
        cleanup_orphaned_envs(log=lambda x: None)  # Silent
    except Exception:
        pass  # Never fail startup due to cleanup

_run_startup_cleanup()
