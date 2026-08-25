"""Packages layer - CUDA wheels, toml generation, node packs."""

from ..pixi import PIXI
from .cuda_wheels import (
    CUDA_WHEELS_INDEX,
    cuda_wheels_index,
    get_wheel_url,
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
    # Node dependencies
    "install_node_packs",
    "clone_node",
    "normalize_repo_url",
]
