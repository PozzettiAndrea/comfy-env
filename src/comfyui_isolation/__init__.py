"""
comfyui-isolation: Process isolation for ComfyUI custom nodes.

This package provides a clean API for running ComfyUI node code in isolated
Python virtual environments, solving dependency conflicts between nodes.

Example usage:

    from comfyui_isolation import IsolatedEnv, WorkerBridge
    from pathlib import Path

    # Define the isolated environment
    env = IsolatedEnv(
        name="my-node",
        python="3.10",
        cuda="12.8",
        requirements=["torch==2.8.0", "nvdiffrast"],
    )

    # Create bridge to communicate with isolated process
    bridge = WorkerBridge(env, worker_script=Path("worker.py"))

    # Ensure environment is set up
    bridge.ensure_environment()

    # Call functions in the isolated environment
    result = bridge.call("process_image", image=my_image)
"""

__version__ = "0.1.0"

from .env.config import IsolatedEnv
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

# TorchBridge is optional (requires PyTorch)
try:
    from .ipc.torch_bridge import TorchBridge, TorchWorker
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

__all__ = [
    # Environment
    "IsolatedEnv",
    "IsolatedEnvManager",
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
    # IPC (subprocess-based)
    "WorkerBridge",
    "BaseWorker",
    "register",
]

# Add torch-based IPC if available
if _TORCH_AVAILABLE:
    __all__ += ["TorchBridge", "TorchWorker"]
