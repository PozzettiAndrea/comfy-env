"""
Simple index resolver for finding wheel URLs from PEP 503 indexes.

This module fetches and parses simple index HTML to find the exact wheel URL
for a package given the runtime environment (CUDA, torch, python versions).
"""

import re
import urllib.request
from typing import Optional, Dict
from urllib.parse import urljoin


def resolve_wheel_from_index(
    index_url: str,
    package: str,
    vars_dict: Dict[str, str],
    version: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the exact wheel URL from a PEP 503 simple index or find-links page.

    Args:
        index_url: Base URL of the simple index (e.g., https://pozzettiandrea.github.io/cuda-wheels)
                   or find-links page (e.g., https://data.pyg.org/whl/torch-2.8.0+cu128.html)
        package: Package name (e.g., "cumesh")
        vars_dict: Environment variables dict with cuda_short, torch_mm, py_tag, platform
        version: Specific version to match, or None for latest

    Returns:
        Full wheel URL if found, None otherwise
    """
    # PEP 503: normalize package name (lowercase, replace _ with -)
    normalized = package.lower().replace("_", "-")

    # Try two URL patterns:
    # 1. PEP 503 index: {index_url}/{package}/
    # 2. Find-links page: {index_url} directly (e.g., .html file)
    urls_to_try = []

    if index_url.endswith('.html'):
        # Direct HTML page (find-links style)
        urls_to_try = [index_url]
    else:
        # PEP 503 style - try package subdirectory first, then root
        urls_to_try = [
            f"{index_url.rstrip('/')}/{normalized}/",
            index_url,
        ]

    html = None
    base_url = None
    for url in urls_to_try:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                html = response.read().decode('utf-8')
                base_url = url
                break
        except Exception:
            continue

    if not html:
        return None

    # Extract wheel links from HTML
    # Simple index format: <a href="...">wheel_filename</a>
    wheel_pattern = re.compile(r'<a[^>]+href="([^"]+\.whl)"[^>]*>([^<]+)</a>', re.IGNORECASE)
    wheels = wheel_pattern.findall(html)

    if not wheels:
        return None

    # Build matching criteria from vars_dict
    cuda_short = vars_dict.get("cuda_short", "")
    torch_mm = vars_dict.get("torch_mm", "")
    py_tag = vars_dict.get("py_tag", "")
    platform = vars_dict.get("platform", "")

    # Find matching wheel
    # Wheel filename format: {package}-{version}+cu{cuda}torch{torch}-{pytag}-{pytag}-{platform}.whl
    best_match = None
    best_version = None

    for href, filename in wheels:
        # Check if wheel matches our environment
        if cuda_short and f"cu{cuda_short}" not in filename.lower():
            continue
        if torch_mm and f"torch{torch_mm}" not in filename.lower():
            continue
        if py_tag and py_tag not in filename:
            continue
        if platform and platform not in filename:
            continue

        # Extract version from filename
        # Format: package-version+... or package-version-...
        # Package name in wheel can use underscore or hyphen interchangeably
        pkg_pattern = re.escape(package).replace(r'\_', '[-_]').replace(r'\-', '[-_]')
        norm_pattern = re.escape(normalized).replace(r'\_', '[-_]').replace(r'\-', '[-_]')
        match = re.match(rf'{pkg_pattern}[-_]([^+\-]+)', filename, re.IGNORECASE)
        if not match:
            # Try with normalized name
            match = re.match(rf'{norm_pattern}[-_]([^+\-]+)', filename, re.IGNORECASE)

        if match:
            wheel_version = match.group(1)

            # If specific version requested, must match
            if version and version != "*" and wheel_version != version:
                continue

            # Track best (highest) version
            if best_version is None or _version_gt(wheel_version, best_version):
                best_version = wheel_version
                # Resolve relative URL
                if href.startswith('http'):
                    best_match = href
                else:
                    best_match = urljoin(base_url, href)

    return best_match


def _version_gt(v1: str, v2: str) -> bool:
    """Compare version strings (simple comparison)."""
    try:
        # Split into parts and compare numerically
        parts1 = [int(x) for x in re.split(r'[.\-+]', v1) if x.isdigit()]
        parts2 = [int(x) for x in re.split(r'[.\-+]', v2) if x.isdigit()]
        return parts1 > parts2
    except (ValueError, AttributeError):
        return v1 > v2
