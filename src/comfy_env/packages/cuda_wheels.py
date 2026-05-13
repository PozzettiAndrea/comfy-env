"""CUDA wheels index integration. See: https://pozzettiandrea.github.io/cuda-wheels/"""

import json
import logging
import re
import socket
import ssl
import sys
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

try:
    from importlib.metadata import version as _pkg_version
    _UA = f"comfy-env/{_pkg_version('comfy-env')}"
except Exception:
    _UA = "comfy-env/unknown"

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

# torch.minor -> (torchvision_minor, torchaudio_minor). torchaudio mirrors
# torch's major.minor exactly; torchvision is `0.(torch_minor + 15)` for the
# torch-2.x line. Verified against pytorch.org/get-started/previous-versions
# and torchvision's PyPI release history. Update whenever a new torch lands.
TORCH_FAMILY_COMPAT: dict = {
    "2.4":  ("0.19", "2.4"),
    "2.5":  ("0.20", "2.5"),
    "2.6":  ("0.21", "2.6"),
    "2.7":  ("0.22", "2.7"),
    "2.8":  ("0.23", "2.8"),
    "2.9":  ("0.24", "2.9"),
    "2.10": ("0.25", "2.10"),
    "2.11": ("0.26", "2.11"),
}


def derive_family_pins(torch_pin: str) -> Optional[tuple]:
    """Given a torch pin like '==2.11.0' or '==2.8.*', return
    `(torchvision_pin, torchaudio_pin)` derived from `TORCH_FAMILY_COMPAT`,
    or `None` if torch's minor isn't in the table.

    Returned pins are major.minor `.*` specs (e.g. `'==0.26.*'`, `'==2.11.*'`),
    so the resolver picks the latest matching patch from the cu-tagged index.
    """
    m = re.match(r"==\s*(\d+)\.(\d+)", torch_pin)
    if not m:
        return None
    minor_key = f"{m.group(1)}.{m.group(2)}"
    pair = TORCH_FAMILY_COMPAT.get(minor_key)
    if not pair:
        return None
    vision_minor, audio_minor = pair
    return (f"=={vision_minor}.*", f"=={audio_minor}.*")


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


_TRANSIENT_NET_ERRORS = (ConnectionResetError, socket.timeout, TimeoutError)


def _fetch_with_retries(url: str, timeout: int = 10, max_retries: int = 3,
                        log: Optional[Callable[[str], None]] = None) -> str:
    """Fetch `url` with a real User-Agent and exponential-backoff retries on
    transient transport errors. Non-transient HTTP errors (4xx/5xx) are raised
    immediately. Default Python urllib UA gets RST by some corporate proxies
    and AV middleboxes, so we always send `comfy-env/<version>`.
    """
    backoff = (1, 2, 4)
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as e:
            if not isinstance(e.reason, _TRANSIENT_NET_ERRORS):
                raise
            last_err = e
        except _TRANSIENT_NET_ERRORS as e:
            last_err = e
        if attempt < max_retries - 1:
            sleep_s = backoff[attempt]
            if log is not None:
                log(f"[cuda-wheels]   retry {attempt+1}/{max_retries} after {sleep_s}s ({type(last_err).__name__})")
            time.sleep(sleep_s)
    raise last_err


def _fetch_from_github_api(package: str, torch_version: str, cuda_version: str,
                           python_version: str,
                           log: Optional[Callable[[str], None]] = None) -> Optional[tuple]:
    """Fallback when the GH Pages index is unreachable: list release assets
    via `api.github.com/repos/PozzettiAndrea/cuda-wheels/releases` and match
    by the same filename pattern. Different routing edge than Pages, so often
    works when Fastly is RST-ing the Pages host. Returns `(url, name)` or None.
    """
    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = ".".join(torch_version.split(".")[:2])
    py_tag = f"cp{python_version.replace('.', '')}"
    platform_tags = _platform_tags()
    local_patterns = [f"+cu{cuda_short}torch{torch_short}", f"+pt{torch_short}cu{cuda_short}"]
    pkg_variants_set = set(_pkg_variants(package))

    api_url = "https://api.github.com/repos/PozzettiAndrea/cuda-wheels/releases?per_page=100"
    try:
        body = _fetch_with_retries(api_url, timeout=15, log=log)
    except Exception as e:
        if log is not None:
            log(f"[cuda-wheels]   GitHub Releases API: {type(e).__name__}: {e}")
        return None
    try:
        releases = json.loads(body)
    except Exception:
        return None

    candidates = []
    for release in releases:
        for asset in release.get("assets", ()):
            name = asset.get("name", "")
            wheel_pkg = name.split("-", 1)[0] if "-" in name else ""
            if wheel_pkg not in pkg_variants_set:
                continue
            if not any(p in name for p in local_patterns):
                continue
            if py_tag not in name:
                continue
            if platform_tags and not any(t in name for t in platform_tags):
                continue
            url = asset.get("browser_download_url")
            if url:
                candidates.append((url, name))
    if not candidates:
        return None
    for url, name in candidates:
        if "manylinux" in name:
            return (url, name)
    return candidates[0]


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
    deferred_errors = []
    for pkg_dir in _pkg_variants(package):
        index_url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        if index_url in attempted:
            continue
        attempted.append(index_url)
        try:
            html = _fetch_with_retries(index_url, timeout=10, log=_emit)
        except urllib.error.HTTPError as e:
            deferred_errors.append(f"[cuda-wheels]   {index_url}: HTTPError: {e}")
            continue
        except Exception as e:
            deferred_errors.append(f"[cuda-wheels]   {index_url}: {type(e).__name__}: {e}")
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

    # Index path failed for every variant -- try the different-transport fallback.
    _emit(f"[cuda-wheels]   GH Pages index unreachable, falling back to GitHub Releases API...")
    api_result = _fetch_from_github_api(package, torch_version, cuda_version, python_version, log=_emit)
    if api_result is not None:
        url, display = api_result
        _emit(f"[cuda-wheels]   Found via API: {display}")
        return url

    # Both paths failed: surface buffered per-URL errors and an actionable hint.
    for line in deferred_errors:
        _emit(line)
    _emit(f"[cuda-wheels]   No wheel found via index or API. If your network blocks")
    _emit(f"[cuda-wheels]   *.github.io / fastly, set HTTPS_PROXY to a working proxy.")
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
