"""Configuration parsing for comfy-env."""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomli

from .types import ComfyEnvConfig, NodeDependency

ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"  # Main node config
CONFIG_FILE_NAME = "comfy-env.toml"  # Isolated folder config


def load_config(path: Path) -> ComfyEnvConfig:
    """Load and parse config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        return parse_config(tomli.load(f))


def discover_config(node_dir: Path, root: bool = True) -> Optional[ComfyEnvConfig]:
    """Find and load config from directory. Checks root config first if root=True."""
    node_dir = Path(node_dir)
    if root:
        root_path = node_dir / ROOT_CONFIG_FILE_NAME
        if root_path.exists():
            return load_config(root_path)
    config_path = node_dir / CONFIG_FILE_NAME
    return load_config(config_path) if config_path.exists() else None


def parse_config(data: Dict[str, Any]) -> ComfyEnvConfig:
    """Parse TOML data into ComfyEnvConfig."""
    data = copy.deepcopy(data)

    python_version = data.pop("python", None)
    python_version = str(python_version) if python_version else None

    cuda_packages = _ensure_list(data.pop("cuda", {}).get("packages", []))
    apt_packages = _ensure_list(data.pop("apt", {}).get("packages", []))
    env_vars = {str(k): str(v) for k, v in data.pop("env_vars", {}).items()}
    node_reqs = _parse_node_reqs(data.pop("node_reqs", {}))

    return ComfyEnvConfig(
        python=python_version,
        cuda_packages=cuda_packages,
        apt_packages=apt_packages,
        env_vars=env_vars,
        node_reqs=node_reqs,
        pixi_passthrough=data,
    )


def _parse_node_reqs(data: Dict[str, Any]) -> List[NodeDependency]:
    return [
        NodeDependency(name=name, repo=value if isinstance(value, str) else value.get("repo", ""))
        for name, value in data.items()
    ]


def _ensure_list(value) -> List:
    return value if isinstance(value, list) else ([value] if value else [])
