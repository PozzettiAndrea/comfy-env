"""Built-in registry of CUDA packages and their wheel sources.

This module provides a mapping of well-known CUDA packages to their
installation sources, eliminating the need for users to specify
wheel_sources in their comfyui_env.toml.

Install method types:
- "index": Use pip --extra-index-url (PEP 503 simple repository)
- "github_index": GitHub Pages index (--find-links)
- "find_links": Use pip --find-links (for PyG, etc.)
- "pypi_variant": Package name varies by CUDA version (e.g., spconv-cu124)
- "github_release": Direct wheel URL from GitHub releases with fallback sources
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
    # Uses --find-links (not --extra-index-url) for proper wheel discovery
    # =========================================================================
    "torch-scatter": {
        "method": "find_links",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Scatter operations for PyTorch",
    },
    "torch-cluster": {
        "method": "find_links",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Clustering algorithms for PyTorch",
    },
    "torch-sparse": {
        "method": "find_links",
        "index_url": "https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_short}.html",
        "description": "Sparse tensor operations for PyTorch",
    },
    "torch-spline-conv": {
        "method": "find_links",
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
    # nvdiffrast - wheels are now at cu{cuda}-torch{torch_short} releases
    "nvdiffrast": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/nvdiffrast-full-wheels/cu{cuda_short}-torch{torch_short}/",
        "description": "NVIDIA differentiable rasterizer",
    },
    # cumesh, o_voxel, flex_gemm, nvdiffrec_render use torch_short (3 digits: 280)
    "cumesh": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/cumesh-wheels/cu{cuda_short}-torch{torch_short}/",
        "description": "CUDA-accelerated mesh utilities",
    },
    "o_voxel": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/ovoxel-wheels/cu{cuda_short}-torch{torch_short}/",
        "description": "O-Voxel CUDA extension for TRELLIS",
    },
    "flex_gemm": {
        "method": "github_index",
        "index_url": "https://pozzettiandrea.github.io/flexgemm-wheels/cu{cuda_short}-torch{torch_short}/",
        "description": "Flexible GEMM operations",
    },
    "nvdiffrec_render": {
        "method": "github_release",
        "sources": [
            {
                "name": "PozzettiAndrea",
                "url_template": "https://github.com/PozzettiAndrea/nvdiffrec_render-wheels/releases/download/cu{cuda_short}-torch{torch_short}/nvdiffrec_render-{version}%2Bcu{cuda_short}torch{torch_mm}-{py_tag}-{py_tag}-linux_x86_64.whl",
                "platforms": ["linux_x86_64"],
            },
            {
                "name": "PozzettiAndrea-windows",
                "url_template": "https://github.com/PozzettiAndrea/nvdiffrec_render-wheels/releases/download/cu{cuda_short}-torch{torch_short}/nvdiffrec_render-{version}%2Bcu{cuda_short}torch{torch_mm}-{py_tag}-{py_tag}-win_amd64.whl",
                "platforms": ["win_amd64", "windows_amd64"],
            },
        ],
        "description": "NVDiffRec rendering utilities",
    },

    # =========================================================================
    # spconv - PyPI with CUDA-versioned package names
    # Package names: spconv-cu118, spconv-cu120, spconv-cu121, spconv-cu124, spconv-cu126
    # Note: Max available is cu126 as of Jan 2026, use explicit spconv-cu126 in config
    # =========================================================================
    "spconv": {
        "method": "pypi_variant",
        "package_template": "spconv-cu{cuda_short2}",
        "description": "Sparse convolution library (use spconv-cu126 for CUDA 12.6+)",
    },

    # =========================================================================
    # sageattention - Fast quantized attention (2-5x faster than FlashAttention)
    # Linux: Prebuilt wheels from Kijai/PrecompiledWheels (v2.2.0, cp312)
    # Windows: Prebuilt wheels from woct0rdho (v2.2.0, cp39-abi3)
    # =========================================================================
    "sageattention": {
        "method": "github_release",
        "sources": [
            # Linux: Kijai's precompiled wheels on HuggingFace (Python 3.12)
            {
                "name": "kijai-hf",
                "url_template": "https://huggingface.co/Kijai/PrecompiledWheels/resolve/main/sageattention-{version}-cp312-cp312-linux_x86_64.whl",
                "platforms": ["linux_x86_64"],
            },
            # Windows: woct0rdho prebuilt wheels (ABI3: Python >= 3.9)
            # Format: sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl
            {
                "name": "woct0rdho",
                "url_template": "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0%2Bcu{cuda_short}torch{torch_version}.post3-cp39-abi3-win_amd64.whl",
                "platforms": ["win_amd64"],
            },
        ],
        "description": "SageAttention - 2-5x faster than FlashAttention with quantized kernels",
    },

    # =========================================================================
    # triton - Required for sageattention on Linux (usually bundled with PyTorch)
    # =========================================================================
    "triton": {
        "method": "pypi",
        "description": "Triton compiler for custom CUDA kernels (required by sageattention)",
    },

    # =========================================================================
    # flash-attn - Multi-source prebuilt wheels
    # Required for UniRig and other transformer-based models
    # Sources: Dao-AILab (official), mjun0812 (Linux), bdashore3 (Windows)
    # =========================================================================
    "flash-attn": {
        "method": "github_release",
        "sources": [
            # Linux: Dao-AILab official wheels (CUDA 12.x, PyTorch 2.4-2.8)
            # Format: flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
            {
                "name": "Dao-AILab",
                "url_template": "https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/flash_attn-{version}%2Bcu{cuda_major}torch{torch_dotted_mm}cxx11abiTRUE-{py_tag}-{py_tag}-linux_x86_64.whl",
                "platforms": ["linux_x86_64"],
            },
            # Linux: mjun0812 prebuilt wheels (CUDA 12.4-13.0, PyTorch 2.5-2.9)
            # Format: flash_attn-2.8.3+cu128torch2.8-cp310-cp310-linux_x86_64.whl
            # Note: Release v0.7.2 contains multiple flash_attn versions
            {
                "name": "mjun0812",
                "url_template": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-{version}%2Bcu{cuda_short}torch{torch_dotted_mm}-{py_tag}-{py_tag}-linux_x86_64.whl",
                "platforms": ["linux_x86_64"],
            },
            # Windows: bdashore3 prebuilt wheels (CUDA 12.4/12.8, PyTorch 2.6-2.8)
            {
                "name": "bdashore3",
                "url_template": "https://github.com/bdashore3/flash-attention/releases/download/v{version}/flash_attn-{version}%2Bcu{cuda_short}torch{torch_version}cxx11abiFALSE-{py_tag}-{py_tag}-win_amd64.whl",
                "platforms": ["win_amd64"],
            },
        ],
        "description": "Flash Attention for fast transformer inference",
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
