"""Packages layer - CUDA wheels, toml generation, node dependencies."""

from .pixi import PIXI, ensure_pixi
from .cuda_wheels import (
    CUDA_WHEELS_INDEX,
    get_wheel_url,
    find_available_wheels,
    get_cuda_torch_mapping,
)
from .toml_generator import (
    build_workspace_toml,
    write_workspace_pixi_toml,
    parse_comfyui_requirements,
)
from .node_dependencies import (
    install_node_dependencies,
    clone_node,
    normalize_repo_url,
)

__all__ = [
    "PIXI",
    # CUDA wheels
    "CUDA_WHEELS_INDEX",
    "get_wheel_url",
    "find_available_wheels",
    "get_cuda_torch_mapping",
    # TOML generation
    "build_workspace_toml",
    "write_workspace_pixi_toml",
    "parse_comfyui_requirements",
    # Node dependencies
    "install_node_dependencies",
    "clone_node",
    "normalize_repo_url",
]
