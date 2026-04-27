"""CUDA wheels index integration. See: https://pozzettiandrea.github.io/cuda-wheels/"""

import logging
import re
import ssl
import sys
import urllib.request
from typing import Callable, List, Optional

logger = logging.getLogger("comfy-env.cuda-wheels")

CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/v2/"


def _ssl_context() -> Optional[ssl.SSLContext]:
    """Build an SSL context using certifi's CA bundle when available.

    Portable/embedded Python distributions (e.g. ComfyUI's python_embeded) often ship
    without a complete CA store, which makes urllib fail with CERTIFICATE_VERIFY_FAILED
    against hosts whose chain isn't in the stripped default store (notably GitHub Pages).
    certifi is effectively always present (pip depends on it), so prefer it when we can.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None

# Tier-2 fallback combo. Only consulted by the install pipeline (see
# install._resolve_wheel_combo) when the host's bootstrap (python, cuda, torch)
# combo doesn't have every required cuda-wheel published. Keep this aligned with
# the cuda-wheels build matrix's known-good baseline.
CUDA_TORCH_MAP = {"12.8": "2.8", "12.4": "2.4"}
FALLBACK_COMBO = ("12.8", "2.8")  # (cuda, torch) -- always paired with bootstrap python


def get_cuda_torch_mapping() -> dict:
    return CUDA_TORCH_MAP.copy()


def get_torch_version_for_cuda(cuda_version: str) -> Optional[str]:
    return CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]))


def check_all_wheels_available(packages: List[str], torch_version: str,
                               cuda_version: str, python_version: str,
                               log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Check if all required cuda-wheels are available for this CUDA+torch combo.

    Returns None if all packages have wheels, or the name of the first missing package.
    If `log` is provided, lookup progress and failure reasons are surfaced to the caller's
    log stream (the cuda_wheels logger is not wired to install logs by default).
    """
    for package in packages:
        url = get_wheel_url(package, torch_version, cuda_version, python_version, log=log)
        if not url:
            return package
    return None


def _pkg_variants(package: str) -> List[str]:
    return [package, package.replace("-", "_"), package.replace("_", "-")]


def _platform_tags() -> List[str]:
    """Return platform tags to match against wheel filenames (most specific first)."""
    if sys.platform.startswith("linux"):
        return ["manylinux", "linux"]
    if sys.platform == "win32":
        return ["win_amd64"]
    return []


def get_wheel_url(package: str, torch_version: str, cuda_version: str, python_version: str,
                  log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """Get direct URL to matching wheel from cuda-wheels index.

    If `log` is provided, every HTTP attempt, the matched wheel, or the per-URL failure
    reason is emitted via that callback in addition to the module logger.
    """
    def _emit(msg: str) -> None:
        logger.info(msg)
        if log is not None:
            log(msg)

    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = ".".join(torch_version.split(".")[:2])
    py_tag = f"cp{python_version.replace('.', '')}"
    platform_tags = _platform_tags()

    local_patterns = [f"+cu{cuda_short}torch{torch_short}", f"+pt{torch_short}cu{cuda_short}"]
    link_pattern = re.compile(r'href="([^"]+\.whl)"[^>]*>([^<]+)</a>', re.IGNORECASE)

    _emit(f"[cuda-wheels] Looking up {package}: cu{cuda_short} torch{torch_short} {py_tag} {' '.join(platform_tags) or 'any'}")

    candidates = []
    attempted = []
    for pkg_dir in _pkg_variants(package):
        index_url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        if index_url in attempted:
            continue
        attempted.append(index_url)
        try:
            with urllib.request.urlopen(index_url, timeout=10, context=_ssl_context()) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:
            _emit(f"[cuda-wheels]   {index_url}: {type(e).__name__}: {e}")
            continue

        for match in link_pattern.finditer(html):
            wheel_url, display = match.group(1), match.group(2)
            if any(p in display for p in local_patterns) and py_tag in display:
                if not platform_tags or any(t in display for t in platform_tags):
                    url = wheel_url if wheel_url.startswith("http") else f"{CUDA_WHEELS_INDEX}{pkg_dir}/{wheel_url}"
                    candidates.append((url, display))

    if candidates:
        # Prefer manylinux wheels over plain linux
        for url, display in candidates:
            if "manylinux" in display:
                _emit(f"[cuda-wheels]   Found: {display}")
                return url
        url, display = candidates[0]
        _emit(f"[cuda-wheels]   Found: {display}")
        return url

    _emit(f"[cuda-wheels]   No matching wheel found (tried {len(attempted)} index URL(s))")
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
    torch_short = ".".join(torch_version.split(".")[:2])
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
