"""
Packages layer - Package installation with side effects.

Handles pixi, CUDA wheels, apt packages, and node dependencies.
"""

from .pixi import (
    ensure_pixi,
    get_pixi_path,
    get_pixi_python,
    pixi_install,
    pixi_run,
    pixi_clean,
)
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
from .apt import (
    apt_install,
    check_apt_packages,
)
from .node_dependencies import (
    install_node_dependencies,
    clone_node,
    normalize_repo_url,
)

__all__ = [
    # Pixi package manager
    "ensure_pixi",
    "get_pixi_path",
    "get_pixi_python",
    "pixi_install",
    "pixi_run",
    "pixi_clean",
    # CUDA wheels
    "CUDA_WHEELS_INDEX",
    "get_wheel_url",
    "find_available_wheels",
    "get_cuda_torch_mapping",
    # TOML generation
    "build_workspace_toml",
    "write_workspace_pixi_toml",
    "parse_comfyui_requirements",
    # APT packages
    "apt_install",
    "check_apt_packages",
    # Node dependencies
    "install_node_dependencies",
    "clone_node",
    "normalize_repo_url",
]
