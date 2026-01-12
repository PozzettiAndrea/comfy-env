"""Built-in registry of CUDA packages and their wheel sources.

This module provides a mapping of well-known CUDA packages to their
installation sources, eliminating the need for users to specify
wheel_sources in their comfyui_env.toml.

Install method types:
- "index": Use pip --extra-index-url (PEP 503 simple repository)
- "github_index": GitHub Pages index (--find-links)
- "pypi_variant": Package name varies by CUDA version (e.g., spconv-cu124)
"""

from typing import Dict, Any, Optional


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


# =============================================================================
# Package Registry
# =============================================================================
# Maps package names to their installation configuration.
#
# Template variables available:
#   {cuda_version}  - Full CUDA version (e.g., "12.8")
#   {cuda_short}    - CUDA without dot (e.g., "128")
#   {cuda_short2}   - CUDA short for spconv (e.g., "124" not "1240")
#   {torch_version} - Full PyTorch version (e.g., "2.8.0")
#   {torch_short}   - PyTorch without dots (e.g., "280")
#   {torch_mm}      - PyTorch major.minor (e.g., "28")
#   {py_version}    - Python version (e.g., "3.10")
#   {py_short}      - Python without dot (e.g., "310")
#   {py_minor}      - Python minor version only (e.g., "10")
#   {platform}      - Platform tag (e.g., "linux_x86_64")
# =============================================================================

PACKAGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # PyTorch Geometric (PyG) packages - official index
    # https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
    # =========================================================================
    "torch-scatter": {
        "method": "index",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Scatter operations for PyTorch",
    },
    "torch-cluster": {
        "method": "index",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Clustering algorithms for PyTorch",
    },
    "torch-sparse": {
        "method": "index",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Sparse tensor operations for PyTorch",
    },
    "torch-spline-conv": {
        "method": "index",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Spline convolutions for PyTorch",
    },

    # =========================================================================
    # pytorch3d - Facebook's official wheels
    # https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
    # =========================================================================
    "pytorch3d": {
        "method": "index",
        "index_url": "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py3{py_minor}_cu{cuda_short}_pyt{torch_short}/download.html",
        "description": "PyTorch3D - 3D deep learning library",
    },

    # =========================================================================
    # PozzettiAndrea wheel repos (GitHub Pages indexes)
    # =========================================================================
    "nvdiffrast": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/nvdiffrast-full-wheels/cu{cuda_short}-torch{torch_mm}/",
        "description": "NVIDIA differentiable rasterizer",
    },
    "cumesh": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/cumesh-wheels/cu{cuda_short}-torch{torch_mm}/",
        "description": "CUDA-accelerated mesh utilities",
    },
    "o_voxel": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/ovoxel-wheels/cu{cuda_short}-torch{torch_mm}/",
        "description": "O-Voxel CUDA extension for TRELLIS",
    },
    "flex_gemm": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/flexgemm-wheels/cu{cuda_short}-torch{torch_mm}/",
        "description": "Flexible GEMM operations",
    },
    "nvdiffrec_render": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/nvdiffrec_render-wheels/cu{cuda_short}-torch{torch_mm}/",
        "description": "NVDiffRec rendering utilities",
    },

    # =========================================================================
    # spconv - PyPI with CUDA-versioned package names
    # Package names: spconv-cu118, spconv-cu121, spconv-cu124
    # =========================================================================
    "spconv": {
        "method": "pypi_variant",
        "package_template": "spconv-cu{cuda_short2}",
        "description": "Sparse convolution library",
    },
}


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
