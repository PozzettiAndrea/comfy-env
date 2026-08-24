"""Packages layer - CUDA wheels, toml generation, node packs."""

from ..pixi import PIXI, ensure_pixi
from .cuda_wheels import (
    CUDA_WHEELS_INDEX,
    cuda_wheels_index,
    get_wheel_url,
    find_available_wheels,
    get_cuda_torch_mapping,
)
from .node_packs import (
    install_node_packs,
    clone_node,
    normalize_repo_url,
)

__all__ = [
    "PIXI",
    # CUDA wheels
    "CUDA_WHEELS_INDEX",
    "cuda_wheels_index",
    "get_wheel_url",
    "find_available_wheels",
    "get_cuda_torch_mapping",
    # Node dependencies
    "install_node_packs",
    "clone_node",
    "normalize_repo_url",
]
