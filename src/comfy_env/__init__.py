"""
comfy-env: Environment management for ComfyUI custom nodes.

This package provides:
- CUDA wheel resolution and in-place installation (Type 2 nodes)
- Process isolation with separate venvs (Type 1 nodes)

## Quick Start - In-Place Installation

    from comfy_env import install

    # Auto-discover config and install CUDA wheels
    install()

    # Or with explicit config
    install(config="comfyui_env.toml")

## Quick Start - Process Isolation

    from comfy_env.workers import get_worker, TorchMPWorker

    # Same-venv isolation (zero-copy tensors)
    worker = TorchMPWorker()
    result = worker.call(my_gpu_function, image=tensor)

    # Cross-venv isolation
    from comfy_env.workers import PersistentVenvWorker
    worker = PersistentVenvWorker(python="/path/to/venv/bin/python")
    result = worker.call_module("my_module", "my_func", image=tensor)

## CLI

    comfy-env install          # Install from config
    comfy-env info             # Show environment info
    comfy-env resolve pkg==1.0 # Show resolved wheel URL
    comfy-env doctor           # Verify installation

## Legacy APIs (still supported)

The @isolated decorator and WorkerBridge are still available.
"""

__version__ = "0.0.2"

from .env.config import IsolatedEnv, EnvManagerConfig, LocalConfig, NodeReq
from .env.config_file import (
    load_env_from_file,
    discover_env_config,
    load_config,
    discover_config,
    CONFIG_FILE_NAMES,
)
from .env.manager import IsolatedEnvManager
from .env.detection import detect_cuda_version, detect_gpu_info, get_gpu_summary
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
from .resolver import RuntimeEnv, WheelResolver
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
    "WheelResolver",
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
    "IsolatedEnvManager",
    # Config file loading
    "load_env_from_file",
    "discover_env_config",
    "load_config",
    "discover_config",
    "CONFIG_FILE_NAMES",
    # Detection
    "detect_cuda_version",
    "detect_gpu_info",
    "get_gpu_summary",
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
