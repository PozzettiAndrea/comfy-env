"""Load configuration from comfy-env.toml.

comfy-env.toml is a superset of pixi.toml. Custom sections we handle:
- python = "3.11" - Python version for isolated envs
- [cuda] packages = [...] - CUDA packages (triggers find-links + PyTorch detection)
- [node_reqs] - Other ComfyUI nodes to clone

Everything else passes through to pixi.toml directly.

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

import copy
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Use built-in tomllib (Python 3.11+) or tomli fallback
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

from .types import ComfyEnvConfig, NodeReq


CONFIG_FILE_NAME = "comfy-env.toml"

# Sections we handle specially (not passed through to pixi.toml)
CUSTOM_SECTIONS = {"python", "cuda", "node_reqs", "apt", "env_vars"}


def load_config(path: Path) -> ComfyEnvConfig:
    """
    Load configuration from a TOML file.

    Args:
        path: Path to comfy-env.toml

    Returns:
        ComfyEnvConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If tomli not installed (Python < 3.11)
    """
    if tomllib is None:
        raise ImportError(
            "TOML parsing requires tomli for Python < 3.11. "
            "Install with: pip install tomli"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return _parse_config(data)


def discover_config(node_dir: Path) -> Optional[ComfyEnvConfig]:
    """
    Find and load comfy-env.toml from a directory.

    Args:
        node_dir: Directory to search

    Returns:
        ComfyEnvConfig if found, None otherwise
    """
    if tomllib is None:
        return None

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
