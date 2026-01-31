"""Configuration types for comfy-env."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NodeDependency:
    """A ComfyUI custom node dependency."""
    name: str
    repo: str  # "owner/repo" or full URL


NodeReq = NodeDependency  # Backwards compat


@dataclass
class ComfyEnvConfig:
    """Parsed comfy-env.toml configuration."""
    python: Optional[str] = None
    cuda_packages: List[str] = field(default_factory=list)
    apt_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    node_reqs: List[NodeDependency] = field(default_factory=list)
    pixi_passthrough: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        return bool(self.cuda_packages)

    @property
    def has_dependencies(self) -> bool:
        return bool(
            self.cuda_packages or self.apt_packages or self.node_reqs
            or self.pixi_passthrough.get("dependencies")
            or self.pixi_passthrough.get("pypi-dependencies")
        )
