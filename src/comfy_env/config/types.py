"""Configuration types for comfy-env."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeReq:
    """A node dependency (another ComfyUI custom node)."""
    name: str
    repo: str  # GitHub repo, e.g., "owner/repo"


@dataclass
class ComfyEnvConfig:
    """
    Configuration from comfy-env.toml.

    comfy-env.toml is a superset of pixi.toml. Custom sections we handle:
    - python = "3.11" - Python version for isolated envs
    - [cuda] packages = [...] - CUDA packages (triggers find-links + PyTorch detection)
    - [node_reqs] - Other ComfyUI nodes to clone

    Everything else passes through to pixi.toml directly:
    - [dependencies] - conda packages
    - [pypi-dependencies] - pip packages
    - [target.linux-64.pypi-dependencies] - platform-specific deps
    - Any other pixi.toml syntax

    Example config:
        python = "3.11"

        [cuda]
        packages = ["cumesh"]

        [dependencies]
        mesalib = "*"
        cgal = "*"

        [pypi-dependencies]
        numpy = ">=1.21.0,<2"
        trimesh = { version = ">=4.0.0", extras = ["easy"] }

        [target.linux-64.pypi-dependencies]
        embreex = "*"

        [node_reqs]
        SomeNode = "owner/repo"
    """
    # python = "3.11" - Python version (for isolated envs)
    python: Optional[str] = None

    # [cuda] - CUDA packages (installed via find-links index)
    cuda_packages: List[str] = field(default_factory=list)

    # [apt] - System packages to install via apt (Linux only)
    apt_packages: List[str] = field(default_factory=list)

    # [env_vars] - Environment variables to set early (in prestartup)
    env_vars: Dict[str, str] = field(default_factory=dict)

    # [node_reqs] - other ComfyUI nodes to clone
    node_reqs: List[NodeReq] = field(default_factory=list)

    # Everything else from comfy-env.toml passes through to pixi.toml
    pixi_passthrough: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        return bool(self.cuda_packages)
