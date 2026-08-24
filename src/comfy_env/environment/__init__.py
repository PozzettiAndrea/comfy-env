"""Environment layer - path resolution, setup, and platform workarounds."""

from .cache import (
    CE_WORKSPACE_DIR,
    get_env_name,
    get_workspace_dir,
    find_comfyui_dir_from_node,
    copy_files,
)
from .setup import (
    setup_env,
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
    "dedupe_libomp",
]

from .runtime import RuntimeEnv  # noqa: E402,F401
