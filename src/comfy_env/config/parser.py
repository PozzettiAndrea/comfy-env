"""
Configuration parsing for comfy-env.

Loads comfy-env.toml (a superset of pixi.toml) and provides typed config objects.
"""

import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import tomli

# --- Types&Constants ---
CONFIG_FILE_NAME = "comfy-env.toml"

@dataclass
class NodeReq:
    """A node dependency (another ComfyUI custom node)."""
    name: str
    repo: str  # GitHub repo, e.g., "owner/repo"

@dataclass
class ComfyEnvConfig:
    """Configuration from comfy-env.toml."""
    python: Optional[str] = None
    cuda_packages: List[str] = field(default_factory=list)
    apt_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    node_reqs: List[NodeReq] = field(default_factory=list)
    pixi_passthrough: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_cuda(self) -> bool:
        return bool(self.cuda_packages)
# --- Types&Constants ---


def load_config(path: Path) -> ComfyEnvConfig:
    """Load config from a TOML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        data = tomli.load(f)
    return _parse_config(data)


def discover_config(node_dir: Path) -> Optional[ComfyEnvConfig]:
    """Find and load comfy-env.toml from a directory."""
    config_path = Path(node_dir) / CONFIG_FILE_NAME
    if config_path.exists():
        return load_config(config_path)

    return None


def _parse_config(data: Dict[str, Any]) -> ComfyEnvConfig:
    """Parse TOML data into ComfyEnvConfig."""
    # Make a copy so we can pop our custom sections
    data = copy.deepcopy(data)

    # Extract python version (top-level key)
    python_version = data.pop("python", None)
    if python_version is not None:
        python_version = str(python_version)

    # Extract [cuda] section
    cuda_data = data.pop("cuda", {})
    cuda_packages = _ensure_list(cuda_data.get("packages", []))

    # Extract [apt] section
    apt_data = data.pop("apt", {})
    apt_packages = _ensure_list(apt_data.get("packages", []))

    # Extract [env_vars] section
    env_vars_data = data.pop("env_vars", {})
    env_vars = {str(k): str(v) for k, v in env_vars_data.items()}

    # Extract [node_reqs] section
    node_reqs_data = data.pop("node_reqs", {})
    node_reqs = _parse_node_reqs(node_reqs_data)

    # Everything else passes through to pixi.toml
    pixi_passthrough = data

    return ComfyEnvConfig(
        python=python_version,
        cuda_packages=cuda_packages,
        apt_packages=apt_packages,
        env_vars=env_vars,
        node_reqs=node_reqs,
        pixi_passthrough=pixi_passthrough,
    )


def _parse_node_reqs(data: Dict[str, Any]) -> List[NodeReq]:
    """Parse [node_reqs] section."""
    node_reqs = []
    for name, value in data.items():
        if isinstance(value, str):
            node_reqs.append(NodeReq(name=name, repo=value))
        elif isinstance(value, dict):
            node_reqs.append(NodeReq(name=name, repo=value.get("repo", "")))
    return node_reqs


def _ensure_list(value) -> List:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []
