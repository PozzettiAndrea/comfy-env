"""Configuration types for comfy-env."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Default timeout for worker health checks (seconds)
DEFAULT_HEALTH_CHECK_TIMEOUT = 5.0


@dataclass
class NodeDependency:
    """A ComfyUI custom node dependency.

    Supports two sources:
    - GitHub: github="owner/repo", optional tag/branch/commit
    - Registry: registry="node-id", optional version
    """
    name: str
    # GitHub source
    github: Optional[str] = None   # "owner/repo" or full URL
    tag: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None
    # Registry source (api.comfy.org)
    registry: Optional[str] = None  # registry node ID
    version: Optional[str] = None   # semver for registry


@dataclass
class ComfyEnvOptions:
    """Runtime options for comfy-env."""
    health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT


@dataclass
class ComfyEnvConfig:
    """Parsed comfy-env.toml configuration."""
    python: Optional[str] = None
    cuda_packages: List[str] = field(default_factory=list)
    apt_packages: List[str] = field(default_factory=list)
    brew_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    node_reqs: List[NodeDependency] = field(default_factory=list)
    options: ComfyEnvOptions = field(default_factory=ComfyEnvOptions)
    pixi_passthrough: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        return bool(self.cuda_packages)

    @property
    def has_dependencies(self) -> bool:
        return bool(
            self.cuda_packages or self.apt_packages or self.brew_packages or self.node_reqs
            or self.pixi_passthrough.get("dependencies")
            or self.pixi_passthrough.get("pypi-dependencies")
        )
