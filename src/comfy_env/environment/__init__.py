"""Environment layer - path resolution, setup, and platform workarounds."""

from .cache import (
    CE_WORKSPACE_DIR,
    get_env_name,
    get_workspace_dir,
    find_comfyui_dir_from_node,
    find_comfyui_source_dir,
    copy_files,
)
from .setup import (
    setup_env,
    USE_COMFY_ENV_VAR,
)
from .libomp import (
    dedupe_libomp,
)

__all__ = [
    "CE_WORKSPACE_DIR",
    "get_env_name",
    "get_workspace_dir",
    "find_comfyui_dir_from_node",
    "copy_files",
    "setup_env",
    "USE_COMFY_ENV_VAR",
    "dedupe_libomp",
]

from .runtime import RuntimeEnv  # noqa: E402,F401
