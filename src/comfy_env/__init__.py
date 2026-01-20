from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfy-env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"  # Fallback for editable installs

from .env.config import IsolatedEnv, EnvManagerConfig, LocalConfig, NodeReq, CondaConfig
from .env.config_file import (
    load_env_from_file,
    discover_env_config,
    load_config,
    discover_config,
    CONFIG_FILE_NAMES,
)
from .env.manager import IsolatedEnvManager
from .env.cuda_gpu_detection import (
    GPUInfo,
    CUDAEnvironment,
    detect_cuda_environment,
    detect_cuda_version,
    detect_gpu_info,
    detect_gpus,
    get_gpu_summary,
    get_recommended_cuda_version,
)
from .env.security import (
    normalize_env_name,
    validate_dependency,
    validate_dependencies,
    validate_path_within_root,
    validate_wheel_url,
)
from .ipc.bridge import WorkerBridge
from .ipc.worker import BaseWorker, register
from .decorator import isolated, shutdown_all_processes

# New in-place installation API
from .install import install, verify_installation
from .resolver import RuntimeEnv

# Pixi integration (for conda packages)
from .pixi import (
    ensure_pixi,
    get_pixi_path,
    pixi_install,
    create_pixi_toml,
    get_pixi_python,
    pixi_run,
)
from .errors import (
    EnvManagerError,
    ConfigError,
    WheelNotFoundError,
    DependencyError,
    CUDANotFoundError,
    InstallError,
)

# New workers module (recommended API)
from .workers import (
    Worker,
    TorchMPWorker,
    VenvWorker,
    WorkerPool,
    get_worker,
    register_worker,
    shutdown_workers,
)

# TorchBridge is optional (requires PyTorch)
try:
    from .ipc.torch_bridge import TorchBridge, TorchWorker
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# PersistentVenvWorker requires the ipc.transport module
try:
    from .workers.venv import PersistentVenvWorker
    _PERSISTENT_AVAILABLE = True
except ImportError:
    _PERSISTENT_AVAILABLE = False

__all__ = [
    # NEW: In-place installation API
    "install",
    "verify_installation",
    "RuntimeEnv",
    # Pixi integration (for conda packages)
    "ensure_pixi",
    "get_pixi_path",
    "pixi_install",
    "create_pixi_toml",
    "get_pixi_python",
    "pixi_run",
    # Errors
    "EnvManagerError",
    "ConfigError",
    "WheelNotFoundError",
    "DependencyError",
    "CUDANotFoundError",
    "InstallError",
    # Workers API (recommended for isolation)
    "Worker",
    "TorchMPWorker",
    "VenvWorker",
    "WorkerPool",
    "get_worker",
    "register_worker",
    "shutdown_workers",
    # Environment & Config
    "IsolatedEnv",
    "EnvManagerConfig",
    "LocalConfig",
    "NodeReq",
    "CondaConfig",
    "IsolatedEnvManager",
    # Config file loading
    "load_env_from_file",
    "discover_env_config",
    "load_config",
    "discover_config",
    "CONFIG_FILE_NAMES",
    # Detection
    "GPUInfo",
    "CUDAEnvironment",
    "detect_cuda_environment",
    "detect_cuda_version",
    "detect_gpu_info",
    "detect_gpus",
    "get_gpu_summary",
    "get_recommended_cuda_version",
    # Security validation
    "normalize_env_name",
    "validate_dependency",
    "validate_dependencies",
    "validate_path_within_root",
    "validate_wheel_url",
    # Legacy IPC (subprocess-based)
    "WorkerBridge",
    "BaseWorker",
    "register",
    # Legacy Decorator API
    "isolated",
    "shutdown_all_processes",
]

# Add torch-based IPC if available
if _TORCH_AVAILABLE:
    __all__ += ["TorchBridge", "TorchWorker"]

# Add PersistentVenvWorker if available
if _PERSISTENT_AVAILABLE:
    __all__ += ["PersistentVenvWorker"]
