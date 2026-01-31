"""
Configuration parsing for comfy-env.

Loads comfy-env.toml (a superset of pixi.toml) and provides typed config objects.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomli

from .types import ComfyEnvConfig, NodeDependency

CONFIG_FILE_NAME = "comfy-env.toml"


def load_config(path: Path) -> ComfyEnvConfig:
    """
    Load config from a TOML file.

    Args:
        path: Path to comfy-env.toml file.

    Returns:
        Parsed ComfyEnvConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        data = tomli.load(f)
    return parse_config(data)


def discover_config(node_dir: Path) -> Optional[ComfyEnvConfig]:
    """
    Find and load comfy-env.toml from a directory.

    Args:
        node_dir: Directory to search for config file.

    Returns:
        ComfyEnvConfig if found, None otherwise.
    """
    config_path = Path(node_dir) / CONFIG_FILE_NAME
    if config_path.exists():
        return load_config(config_path)
    return None


def parse_config(data: Dict[str, Any]) -> ComfyEnvConfig:
    """
    Parse TOML data into ComfyEnvConfig.

    Args:
        data: Parsed TOML dictionary.

    Returns:
        ComfyEnvConfig with extracted sections.
    """
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


def _parse_node_reqs(data: Dict[str, Any]) -> List[NodeDependency]:
    """Parse [node_reqs] section into NodeDependency list."""
    node_reqs = []
    for name, value in data.items():
        if isinstance(value, str):
            node_reqs.append(NodeDependency(name=name, repo=value))
        elif isinstance(value, dict):
            node_reqs.append(NodeDependency(name=name, repo=value.get("repo", "")))
    return node_reqs


def _ensure_list(value) -> List:
    """Ensure value is a list."""
    if isinstance(value, list):
        return value
    if value:
        return [value]
    return []
