"""Built-in registry of CUDA packages and their wheel sources.

This module loads package configurations from wheel_sources.yml and provides
lookup functions for the install module.

Install method types:
- "index": Use pip --extra-index-url (PEP 503 simple repository)
- "github_index": GitHub Pages index (--find-links)
- "find_links": Use pip --find-links (for PyG, etc.)
- "pypi_variant": Package name varies by CUDA version (e.g., spconv-cu124)
- "github_release": Direct wheel URL from GitHub releases with fallback sources
- "pypi": Standard PyPI install
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def get_cuda_short2(cuda_version: str) -> str:
    """Convert CUDA version to 2-3 digit format for spconv.

    spconv uses "cu124" not "cu1240" for CUDA 12.4.

    Args:
        cuda_version: CUDA version string (e.g., "12.4", "12.8")

    Returns:
        Short format string (e.g., "124", "128")

    Examples:
        >>> get_cuda_short2("12.4")
        '124'
        >>> get_cuda_short2("12.8")
        '128'
        >>> get_cuda_short2("11.8")
        '118'
    """
    parts = cuda_version.split(".")
    major = parts[0]
    minor = parts[1] if len(parts) > 1 else "0"
    return f"{major}{minor}"


def _load_wheel_sources() -> Dict[str, Dict[str, Any]]:
    """Load package registry from wheel_sources.yml."""
    yml_path = Path(__file__).parent / "wheel_sources.yml"
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("packages", {})


# Load registry at module import time
PACKAGE_REGISTRY: Dict[str, Dict[str, Any]] = _load_wheel_sources()


def get_package_info(package: str) -> Optional[Dict[str, Any]]:
    """Get registry info for a package.

    Args:
        package: Package name (case-insensitive)

    Returns:
        Registry entry dict or None if not found
    """
    return PACKAGE_REGISTRY.get(package.lower())


def list_packages() -> Dict[str, str]:
    """List all registered packages with their descriptions.

    Returns:
        Dict mapping package name to description
    """
    return {
        name: info.get("description", "No description")
        for name, info in PACKAGE_REGISTRY.items()
    }


def is_registered(package: str) -> bool:
    """Check if a package is in the registry.

    Args:
        package: Package name (case-insensitive)

    Returns:
        True if package is registered
    """
    return package.lower() in PACKAGE_REGISTRY
