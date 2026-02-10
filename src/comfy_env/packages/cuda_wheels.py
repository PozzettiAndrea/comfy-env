"""CUDA wheels index integration. See: https://pozzettiandrea.github.io/cuda-wheels/"""

import re
import sys
import urllib.request
from typing import List, Optional

CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/"
CUDA_TORCH_MAP = {"12.8": "2.8", "12.4": "2.4"}


def get_cuda_torch_mapping() -> dict:
    return CUDA_TORCH_MAP.copy()


def get_torch_version_for_cuda(cuda_version: str) -> Optional[str]:
    return CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]))


def _pkg_variants(package: str) -> List[str]:
    return [package, package.replace("-", "_"), package.replace("_", "-")]


def _platform_tag() -> Optional[str]:
    if sys.platform.startswith("linux"): return "linux_x86_64"
    if sys.platform == "win32": return "win_amd64"
    return None


def get_wheel_url(package: str, torch_version: str, cuda_version: str, python_version: str) -> Optional[str]:
    """Get direct URL to matching wheel from cuda-wheels index."""
    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = torch_version.replace(".", "")[:2]
    py_tag = f"cp{python_version.replace('.', '')}"
    platform_tag = _platform_tag()

    local_patterns = [f"+cu{cuda_short}torch{torch_short}", f"+pt{torch_short}cu{cuda_short}"]
    link_pattern = re.compile(r'href="([^"]+\.whl)"[^>]*>([^<]+)</a>', re.IGNORECASE)

    for pkg_dir in _pkg_variants(package):
        try:
            with urllib.request.urlopen(f"{CUDA_WHEELS_INDEX}{pkg_dir}/", timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception: continue

        for match in link_pattern.finditer(html):
            wheel_url, display = match.group(1), match.group(2)
            if any(p in display for p in local_patterns) and py_tag in display:
                if platform_tag is None or platform_tag in display:
                    return wheel_url if wheel_url.startswith("http") else f"{CUDA_WHEELS_INDEX}{pkg_dir}/{wheel_url}"
    return None


def find_available_wheels(package: str) -> List[str]:
    """List all available wheels for a package."""
    wheels = []
    link_pattern = re.compile(r'href="[^"]*?([^"/]+\.whl)"', re.IGNORECASE)
    for pkg_dir in _pkg_variants(package):
        try:
            with urllib.request.urlopen(f"{CUDA_WHEELS_INDEX}{pkg_dir}/", timeout=10) as resp:
                html = resp.read().decode("utf-8")
            for match in link_pattern.finditer(html):
                name = match.group(1).replace("%2B", "+")
                if name not in wheels: wheels.append(name)
        except Exception: continue
    return wheels


def find_matching_wheel(package: str, torch_version: str, cuda_version: str) -> Optional[str]:
    """Find wheel matching CUDA/torch version, return version spec."""
    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = torch_version.replace(".", "")[:2]
    local_patterns = [f"+cu{cuda_short}torch{torch_short}", f"+pt{torch_short}cu{cuda_short}"]
    wheel_pattern = re.compile(r'href="[^"]*?([^"/]+\.whl)"', re.IGNORECASE)

    for pkg_dir in _pkg_variants(package):
        try:
            with urllib.request.urlopen(f"{CUDA_WHEELS_INDEX}{pkg_dir}/", timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception: continue

        best_match = best_version = None
        for match in wheel_pattern.finditer(html):
            wheel_name = match.group(1).replace("%2B", "+")
            for local in local_patterns:
                if local in wheel_name:
                    parts = wheel_name.split("-")
                    if len(parts) >= 2 and (best_version is None or parts[1] > best_version):
                        best_version = parts[1]
                        best_match = f"{package}==={parts[1]}"
                    break
        if best_match: return best_match
    return None


def get_find_links_urls(package: str) -> List[str]:
    return [f"{CUDA_WHEELS_INDEX}{p}/" for p in _pkg_variants(package)]
