"""
CUDA wheels index integration.

Handles finding and resolving pre-built CUDA wheels from the cuda-wheels index.
See: https://pozzettiandrea.github.io/cuda-wheels/
"""

import re
import sys
import urllib.request
from typing import List, Optional

# CUDA wheels index URL
CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/"

# CUDA version -> PyTorch version mapping
CUDA_TORCH_MAP = {
    "12.8": "2.8",
    "12.4": "2.4",
}


def get_cuda_torch_mapping() -> dict:
    """Get the CUDA to PyTorch version mapping."""
    return CUDA_TORCH_MAP.copy()


def get_torch_version_for_cuda(cuda_version: str) -> Optional[str]:
    """
    Get the PyTorch version that matches a CUDA version.

    Args:
        cuda_version: CUDA version (e.g., "12.8").

    Returns:
        PyTorch version (e.g., "2.8") or None if no mapping.
    """
    cuda_mm = ".".join(cuda_version.split(".")[:2])
    return CUDA_TORCH_MAP.get(cuda_mm)


def get_wheel_url(
    package: str,
    torch_version: str,
    cuda_version: str,
    python_version: str,
) -> Optional[str]:
    """
    Query cuda-wheels index and return the direct URL for a matching wheel.

    This bypasses pip's version validation by providing a direct URL,
    which is necessary for wheels where the filename has a local version
    but the internal METADATA doesn't match.

    Args:
        package: Package name (e.g., "flash-attn").
        torch_version: PyTorch version (e.g., "2.8").
        cuda_version: CUDA version (e.g., "12.8").
        python_version: Python version (e.g., "3.10").

    Returns:
        Direct URL to the wheel file, or None if no match found.
    """
    cuda_short = cuda_version.replace(".", "")[:3]  # "12.8" -> "128"
    torch_short = torch_version.replace(".", "")[:2]  # "2.8" -> "28"
    py_tag = f"cp{python_version.replace('.', '')}"  # "3.10" -> "cp310"

    # Platform tag for current system
    if sys.platform.startswith("linux"):
        platform_tag = "linux_x86_64"
    elif sys.platform == "win32":
        platform_tag = "win_amd64"
    else:
        platform_tag = None  # macOS doesn't typically have CUDA wheels

    # Local version patterns to match:
    # cuda-wheels style: +cu128torch28
    # PyG style: +pt28cu128
    local_patterns = [
        f"+cu{cuda_short}torch{torch_short}",  # cuda-wheels style
        f"+pt{torch_short}cu{cuda_short}",     # PyG style
    ]

    # Try different package name variants (hyphen vs underscore)
    pkg_variants = [package, package.replace("-", "_"), package.replace("_", "-")]

    for pkg_dir in pkg_variants:
        index_url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        try:
            with urllib.request.urlopen(index_url, timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue

        # Parse href and display name from HTML
        link_pattern = re.compile(r'href="([^"]+\.whl)"[^>]*>([^<]+)</a>', re.IGNORECASE)

        for match in link_pattern.finditer(html):
            wheel_url = match.group(1)
            display_name = match.group(2)

            # Match on display name (has normalized torch28 format)
            matches_cuda_torch = any(p in display_name for p in local_patterns)
            matches_python = py_tag in display_name
            matches_platform = platform_tag is None or platform_tag in display_name

            if matches_cuda_torch and matches_python and matches_platform:
                # Return absolute URL
                if wheel_url.startswith("http"):
                    return wheel_url
                # Relative URL - construct absolute
                return f"{CUDA_WHEELS_INDEX}{pkg_dir}/{wheel_url}"

    return None


def find_available_wheels(package: str) -> List[str]:
    """
    List all available wheels for a package from the cuda-wheels index.

    Args:
        package: Package name.

    Returns:
        List of wheel filenames available.
    """
    wheels = []
    pkg_variants = [package, package.replace("-", "_"), package.replace("_", "-")]

    for pkg_dir in pkg_variants:
        index_url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        try:
            with urllib.request.urlopen(index_url, timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue

        # Parse wheel names from HTML
        link_pattern = re.compile(r'href="[^"]*?([^"/]+\.whl)"', re.IGNORECASE)
        for match in link_pattern.finditer(html):
            wheel_name = match.group(1).replace("%2B", "+")
            if wheel_name not in wheels:
                wheels.append(wheel_name)

    return wheels


def find_matching_wheel(
    package: str,
    torch_version: str,
    cuda_version: str,
) -> Optional[str]:
    """
    Find a wheel matching the CUDA/torch version.

    Returns the full version spec (e.g., "flash-attn===2.8.3+cu128torch2.8")
    for use with pip.

    Note: This is used as a fallback for packages with correct wheel metadata.
    For packages with mismatched metadata, use get_wheel_url() instead.

    Args:
        package: Package name.
        torch_version: PyTorch version.
        cuda_version: CUDA version.

    Returns:
        Version spec string or None.
    """
    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = torch_version.replace(".", "")[:2]

    pkg_variants = [package, package.replace("-", "_"), package.replace("_", "-")]

    for pkg_dir in pkg_variants:
        url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue

        wheel_pattern = re.compile(r'href="[^"]*?([^"/]+\.whl)"', re.IGNORECASE)

        local_patterns = [
            f"+cu{cuda_short}torch{torch_short}",
            f"+pt{torch_short}cu{cuda_short}",
        ]

        best_match = None
        best_version = None

        for match in wheel_pattern.finditer(html):
            wheel_name = match.group(1).replace("%2B", "+")

            for local_pattern in local_patterns:
                if local_pattern in wheel_name:
                    parts = wheel_name.split("-")
                    if len(parts) >= 2:
                        version_part = parts[1]
                        if best_version is None or version_part > best_version:
                            best_version = version_part
                            best_match = f"{package}==={version_part}"
                    break

        if best_match:
            return best_match

    return None


def get_find_links_urls(package: str) -> List[str]:
    """
    Get all find-links URLs for a CUDA package.

    Args:
        package: Package name.

    Returns:
        List of URLs to use with pip --find-links.
    """
    pkg_underscore = package.replace("-", "_")
    pkg_hyphen = package.replace("_", "-")

    urls = [f"{CUDA_WHEELS_INDEX}{package}/"]
    if pkg_underscore != package:
        urls.append(f"{CUDA_WHEELS_INDEX}{pkg_underscore}/")
    if pkg_hyphen != package:
        urls.append(f"{CUDA_WHEELS_INDEX}{pkg_hyphen}/")

    return urls
