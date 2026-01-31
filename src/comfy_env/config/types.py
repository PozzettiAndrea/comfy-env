"""
Configuration types for comfy-env.

Dataclasses representing configuration structures.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeDependency:
    """
    A node dependency (another ComfyUI custom node).

    Represents a required custom node that should be cloned and installed.
    """
    name: str
    repo: str  # GitHub repo, e.g., "owner/repo" or full URL


# Backwards compatibility alias
NodeReq = NodeDependency


@dataclass
class ComfyEnvConfig:
    """
    Configuration from comfy-env.toml.

    Represents the parsed configuration file with all sections.
    """
    # Python version for isolated environment
    python: Optional[str] = None

    # [cuda] section - packages to install from cuda-wheels index
    cuda_packages: List[str] = field(default_factory=list)

    # [apt] section - system packages to install (Linux only)
    apt_packages: List[str] = field(default_factory=list)

    # [env_vars] section - environment variables to set
    env_vars: Dict[str, str] = field(default_factory=dict)

    # [node_reqs] section - dependent custom nodes
    node_reqs: List[NodeDependency] = field(default_factory=list)

    # Everything else passes through to pixi.toml
    pixi_passthrough: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        """Check if config requires CUDA packages."""
        return bool(self.cuda_packages)

    @property
    def has_dependencies(self) -> bool:
        """Check if config has any dependencies to install."""
        return bool(
            self.cuda_packages
            or self.apt_packages
            or self.node_reqs
            or self.pixi_passthrough.get("dependencies")
            or self.pixi_passthrough.get("pypi-dependencies")
        )
