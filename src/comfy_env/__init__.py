"""
comfy-env - Environment management for ComfyUI custom nodes.

Features:
- CUDA wheel resolution (pre-built wheels without compilation)
- Process isolation (run nodes in separate Python environments)
- Local _env_* folders (no central cache, no junctions)
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("comfy-env")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


# =============================================================================
# Public API -- this is the whole promised surface.
#
# Everything else in this package is INTERNAL: it moves between releases
# without notice (pixi.py relocated in 0.4.21, the worker pool moved out of
# wrap.py in 0.4.20, _ipc_shared absorbed the CUDA-IPC cache in the same
# release). Internals are deliberately NOT re-exported here, so
# `from comfy_env import SubprocessWorker` fails loudly instead of quietly
# becoming a dependency nobody agreed to. `__getattr__` below turns that
# failure into a signpost.
# =============================================================================

from .install import install
from .environment.setup import setup_env
from .environment.cache import copy_files
from .isolation import register_nodes
from .isolation.workers._ipc_shared import register_serializer

__all__ = [
    "install",
    "setup_env",
    "register_nodes",
    "copy_files",
    "register_serializer",
]


# Names that used to be re-exported here, with where they actually live. Kept
# as a signpost rather than a re-export: importing any of them binds you to an
# internal that carries no compatibility promise.
_INTERNAL = {
    "verify_installation": "comfy_env.install",
    "get_comfyui_dir": "comfy_env.environment.cache.find_comfyui_dir_from_node",
    "find_comfyui_dir_from_node": "comfy_env.environment.cache",
    "ComfyEnvConfig": "comfy_env.config",
    "load_config": "comfy_env.config",
    "discover_config": "comfy_env.config",
    "CONFIG_FILE_NAME": "comfy_env.config",
    "ROOT_CONFIG_FILE_NAME": "comfy_env.config",
    "detect_cuda_version": "comfy_env.detection",
    "has_nvidia_gpu": "comfy_env.detection",
    "get_bootstrap_torch_version": "comfy_env.detection",
    "get_bootstrap_torch_cuda": "comfy_env.detection",
    "get_bootstrap_python_version": "comfy_env.detection",
    "GPUInfo": "comfy_env.detection",
    "CUDAEnvironment": "comfy_env.detection",
    "detect_cuda_environment": "comfy_env.detection",
    "detect_gpu": "comfy_env.detection",
    "get_gpu_summary": "comfy_env.detection",
    "get_recommended_cuda_version": "comfy_env.detection",
    "get_pixi_platform": "comfy_env.detection",
    "is_linux": "comfy_env.detection",
    "is_windows": "comfy_env.detection",
    "is_macos": "comfy_env.detection",
    "PIXI": "comfy_env.pixi",
    "ensure_pixi": "comfy_env.pixi",
    "CUDA_WHEELS_INDEX": "comfy_env.packages.cuda_wheels",
    "cuda_wheels_index": "comfy_env.packages.cuda_wheels",
    "get_wheel_url": "comfy_env.packages.cuda_wheels",
    "get_cuda_torch_mapping": "comfy_env.packages.cuda_wheels",
    "get_env_name": "comfy_env.environment.cache",
    "get_workspace_dir": "comfy_env.environment.cache",
    "Worker": "comfy_env.isolation.workers",
    "WorkerError": "comfy_env.isolation.workers",
    "SubprocessWorker": "comfy_env.isolation.workers.subprocess",
    "TensorKeeper": "comfy_env.isolation.tensor_utils",
    "release_tensor": "comfy_env.isolation.tensor_utils",
    "release_tensors_recursive": "comfy_env.isolation.tensor_utils",
}


def __getattr__(name):
    """PEP 562 signpost for names that are no longer re-exported."""
    where = _INTERNAL.get(name)
    if where:
        raise AttributeError(
            f"comfy_env.{name} is internal and is not part of the public API. "
            f"It lives in {where} and may move or change in any release. "
            f"The public API is: {', '.join(__all__)}."
        )
    raise AttributeError(f"module 'comfy_env' has no attribute {name!r}")
