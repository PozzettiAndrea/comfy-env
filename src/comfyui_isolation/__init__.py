"""
comfyui-isolation: Process isolation for ComfyUI custom nodes.

This package provides simple, explicit process isolation for ComfyUI nodes.

## Quick Start (Recommended API)

    from comfyui_isolation.workers import get_worker, TorchMPWorker

    # Same-venv isolation (zero-copy tensors)
    worker = TorchMPWorker()
    result = worker.call(my_gpu_function, image=tensor)

    # Cross-venv isolation
    from comfyui_isolation.workers import PersistentVenvWorker
    worker = PersistentVenvWorker(python="/path/to/venv/bin/python")
    result = worker.call_module("my_module", "my_func", image=tensor)

## Named Worker Pool

    from comfyui_isolation.workers import register_worker, get_worker

    # Register at startup
    register_worker("sam3d", factory=lambda: PersistentVenvWorker(...))

    # Use anywhere
    worker = get_worker("sam3d")
    result = worker.call_module("my_module", "process", image=tensor)

## Legacy APIs (still supported)

The @isolated decorator and WorkerBridge are still available but the
workers module above is simpler and more explicit.
"""

__version__ = "0.2.1"

from .env.config import IsolatedEnv
from .env.config_file import (
    load_env_from_file,
    discover_env_config,
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
    # NEW: Simple workers API (recommended)
    "Worker",
    "TorchMPWorker",
    "VenvWorker",
    "WorkerPool",
    "get_worker",
    "register_worker",
    "shutdown_workers",
    # Environment
    "IsolatedEnv",
    "IsolatedEnvManager",
    # Config file loading
    "load_env_from_file",
    "discover_env_config",
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
