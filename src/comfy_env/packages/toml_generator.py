"""
TOML generation for pixi.

Converts ComfyEnvConfig to pixi.toml format.
"""

import copy
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import ComfyEnvConfig
from ..detection import get_recommended_cuda_version, get_pixi_platform
from .cuda_wheels import CUDA_TORCH_MAP


def generate_pixi_toml(
    cfg: ComfyEnvConfig,
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> str:
    """
    Generate pixi.toml content from ComfyEnvConfig.

    Args:
        cfg: Configuration to convert.
        node_dir: Node directory (for workspace name).
        log: Logging callback.

    Returns:
        TOML string content.
    """
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "tomli-w required for writing TOML. Install with: pip install tomli-w"
        )

    pixi_data = config_to_pixi_dict(cfg, node_dir, log)
    return tomli_w.dumps(pixi_data)


def write_pixi_toml(
    cfg: ComfyEnvConfig,
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """
    Write pixi.toml file from ComfyEnvConfig.

    Args:
        cfg: Configuration to convert.
        node_dir: Directory to write pixi.toml.
        log: Logging callback.

    Returns:
        Path to written pixi.toml.
    """
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "tomli-w required for writing TOML. Install with: pip install tomli-w"
        )

    pixi_data = config_to_pixi_dict(cfg, node_dir, log)
    pixi_toml = node_dir / "pixi.toml"

    with open(pixi_toml, "wb") as f:
        tomli_w.dump(pixi_data, f)

    log(f"Generated {pixi_toml}")
    return pixi_toml


def config_to_pixi_dict(
    cfg: ComfyEnvConfig,
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Convert ComfyEnvConfig to pixi.toml dictionary.

    This function:
    1. Starts with passthrough sections from comfy-env.toml
    2. Adds workspace metadata (name, version, channels, platforms)
    3. Adds system-requirements if needed (CUDA detection)
    4. Adds PyTorch index URL if CUDA packages present
    5. Enforces torch version constraint for CUDA compatibility

    Args:
        cfg: Configuration to convert.
        node_dir: Node directory (for workspace name).
        log: Logging callback.

    Returns:
        Dictionary ready to be written as pixi.toml.
    """
    # Start with passthrough data
    pixi_data = copy.deepcopy(cfg.pixi_passthrough)

    # Detect CUDA version if needed
    cuda_version = None
    torch_version = None
    if cfg.has_cuda and sys.platform != "darwin":
        cuda_version = get_recommended_cuda_version()
        if cuda_version:
            cuda_mm = ".".join(cuda_version.split(".")[:2])
            torch_version = CUDA_TORCH_MAP.get(cuda_mm, "2.8")
            log(f"Detected CUDA {cuda_version} -> PyTorch {torch_version}")
        else:
            log("Warning: CUDA packages requested but no GPU detected")

    # Build workspace section
    workspace = pixi_data.get("workspace", {})
    workspace.setdefault("name", node_dir.name)
    workspace.setdefault("version", "0.1.0")
    workspace.setdefault("channels", ["conda-forge"])
    workspace.setdefault("platforms", [get_pixi_platform()])
    pixi_data["workspace"] = workspace

    # Build system-requirements section
    system_reqs = pixi_data.get("system-requirements", {})
    if sys.platform.startswith("linux"):
        system_reqs.setdefault("libc", {"family": "glibc", "version": "2.35"})
    if cuda_version:
        cuda_major = cuda_version.split(".")[0]
        system_reqs["cuda"] = cuda_major
    if system_reqs:
        pixi_data["system-requirements"] = system_reqs

    # Build dependencies section (conda packages + python + pip)
    dependencies = pixi_data.get("dependencies", {})
    if cfg.python:
        py_version = cfg.python
        log(f"Using specified Python {py_version}")
    else:
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dependencies.setdefault("python", f"{py_version}.*")
    dependencies.setdefault("pip", "*")
    pixi_data["dependencies"] = dependencies

    # Add PyTorch CUDA index if needed
    if cfg.has_cuda and cuda_version:
        pypi_options = pixi_data.get("pypi-options", {})
        cuda_short = cuda_version.replace(".", "")[:3]
        pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_short}"
        extra_urls = pypi_options.get("extra-index-urls", [])
        if pytorch_index not in extra_urls:
            extra_urls.append(pytorch_index)
        pypi_options["extra-index-urls"] = extra_urls
        pixi_data["pypi-options"] = pypi_options

    # Build pypi-dependencies section
    pypi_deps = pixi_data.get("pypi-dependencies", {})

    # Enforce torch version if CUDA packages present
    if cfg.has_cuda and torch_version:
        torch_major = torch_version.split(".")[0]
        torch_minor = int(torch_version.split(".")[1])
        required_torch = f">={torch_version},<{torch_major}.{torch_minor + 1}"
        if "torch" in pypi_deps and pypi_deps["torch"] != required_torch:
            log(f"Overriding torch={pypi_deps['torch']} with {required_torch} (required for cuda_packages)")
        pypi_deps["torch"] = required_torch

    if pypi_deps:
        pixi_data["pypi-dependencies"] = pypi_deps

    return pixi_data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, override wins for conflicts.

    Args:
        base: Base dictionary.
        override: Dictionary to merge in (takes precedence).

    Returns:
        Merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
