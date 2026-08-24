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

from ..detection.arch import cpu_arch

try:
    from importlib.metadata import version as _pkg_version
    _UA = f"comfy-env/{_pkg_version('comfy-env')}"
except Exception:
    _UA = "comfy-env/unknown"

logger = logging.getLogger("comfy-env.cuda-wheels")

CUDA_WHEELS_INDEX_DEFAULT = "https://pozzettiandrea.github.io/cuda-wheels/v2/"


def cuda_wheels_index() -> str:
    """The cuda-wheels index base URL, with a trailing slash.

    Override with ``COMFY_ENV_CUDA_WHEELS_INDEX`` (env var, or a line in
    ``~/.comfy-env/settings.env`` -- that file is loaded into ``os.environ``
    with ``setdefault``, so one lookup covers both tiers). The point is
    mirrors: an air-gapped or bandwidth-limited site can serve the same
    directory listing from its own host without patching comfy-env.

    Resolved per call rather than frozen at import so a test or a caller can
    change it without reloading the module.

    !! This URL is a TRUST boundary. Wheels from it are installed with
    ``uv pip install --no-deps`` against direct links and are NOT hash-checked
    here, so whatever it serves executes at import time inside the isolated
    env. Point it only at an index you control or trust as much as the
    default (ADR-0026).
    """
    import os
    # Importing settings has the side effect of loading ~/.comfy-env/settings.env
    # into os.environ; without it a file-only override would be invisible here.
    from .. import settings  # noqa: F401
    raw = (os.environ.get("COMFY_ENV_CUDA_WHEELS_INDEX") or "").strip()
    if not raw:
        return CUDA_WHEELS_INDEX_DEFAULT
    # Every call site does f"{index}{pkg}/", so a missing trailing slash would
    # silently build ".../v2flash_attn/" instead of failing.
    return raw if raw.endswith("/") else raw + "/"


# Back-compat alias for readers that want the default without calling.
CUDA_WHEELS_INDEX = CUDA_WHEELS_INDEX_DEFAULT


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

# ...except on linux aarch64, which is a different world and gets its own cell.
# Three things rule out simply reusing the x86 combo, or nudging one axis of it:
#
#   * (12.8, 2.8) does not exist on ARM. PyTorch shipped no linux-aarch64 wheel
#     for the whole 2.8 line on cu128 -- torch 2.8.0, torchvision 0.23.0 and
#     torchaudio 2.8.0 are x86_64/win_amd64 only there. The aarch64 cu128 build
#     broke during the 2.8 cycle (pytorch#157548) and came back for 2.9.
#   * (13.0, 2.8) does not exist anywhere: PyTorch's CUDA 13 line starts at
#     torch 2.9, so 2.8 cannot reach it on any platform.
#   * Staying on 12.8/12.9 leaves Thor DEAD. Their ARM arch list is
#     `8.0;9.0+PTX;10.0;12.0+PTX` -- sm_110 has no cubin at or below it (the
#     10.0 cubin cannot cross a major, the 12.0 PTX is above), so a Thor gets
#     cudaErrorNoKernelImageForDevice at first kernel launch. 13.0's ARM list
#     adds 11.0 natively.
#
# So ARM takes (13.0, 2.10): every current ARM CUDA product is covered (Grace
# sm_90, GB200 sm_100, Thor sm_110, Orin sm_87 via the 8.0 cubin), and from
# torchvision 0.25 / torchaudio 2.10 the ARM wheels are CUDA-tagged (+cu130)
# rather than the plain CPU-only builds cu128/cu129 carry at that torch level.
# Cost, stated plainly: CUDA 13 needs driver r580+.
FALLBACK_COMBO_AARCH64 = ("13.0", "2.10")

# --- Backend -> wheel-index registry -------------------------------------------
# The single seam for adding a non-CUDA accelerator's prebuilt wheels: register
# its index base URL + tier-2 fallback combo here (data, not an `if backend ==`),
# and ship a resolver alongside. Only `cuda` is populated today -- this module IS
# the cuda resolver. A backend-dispatching caller uses `resolve_index_url(backend)`
# instead of hardcoding an index; adding `rocm` is one dict entry + a rocm resolver.
WHEEL_INDEX_REGISTRY: dict[str, dict] = {
    "cuda": {
        # Resolved lazily via resolve_index_url() -- see cuda_wheels_index().
        "index": None,
        # Keyed by CPU arch: the fallback is a claim about published wheels, and
        # what upstream publishes differs per architecture.
        "fallback_combo": {
            "x86_64": FALLBACK_COMBO,
            "aarch64": FALLBACK_COMBO_AARCH64,
        },
    },
    # "rocm": {"index": ROCM_WHEELS_INDEX, "fallback_combo": (...)},  # additive later
}


def resolve_index_url(backend: str = "cuda") -> str:
    """Wheel-index base URL for a backend. Raises for an unregistered backend."""
    try:
        WHEEL_INDEX_REGISTRY[backend]          # membership check
        if backend == "cuda":
            return cuda_wheels_index()
        return WHEEL_INDEX_REGISTRY[backend]["index"]
    except KeyError:
        raise ValueError(
            f"no wheel index registered for backend {backend!r}; "
            f"known: {sorted(WHEEL_INDEX_REGISTRY)}"
        ) from None


def resolve_fallback_combo(backend: str = "cuda", arch: Optional[str] = None) -> tuple:
    """Tier-2 (toolkit, torch) fallback combo for a backend on this machine's CPU.

    `arch` overrides the detected architecture (for tests); it must be one of
    the keys registered for the backend.
    """
    try:
        combos = WHEEL_INDEX_REGISTRY[backend]["fallback_combo"]
    except KeyError:
        raise ValueError(f"no fallback combo registered for backend {backend!r}") from None
    arch = arch or cpu_arch()
    try:
        return combos[arch]
    except KeyError:
        raise ValueError(
            f"no {backend} fallback combo registered for CPU arch {arch!r}; "
            f"known: {sorted(combos)}"
        ) from None

# torch.minor -> (torchvision_minor, torchaudio_minor). Used as a fallback
# when the bootstrap venv doesn't have torchvision/torchaudio installed
# (bare-env cases). For Desktop / any bootstrap that already has the family
# installed, `derive_family_pins` reads the actual installed versions instead
# — auto-tracks decouplings like torch 2.12 shipping without a matching
# torchaudio (latest torchaudio for that torch line is 2.11).
TORCH_FAMILY_COMPAT: dict = {
    "2.4":  ("0.19", "2.4"),
    "2.5":  ("0.20", "2.5"),
    "2.6":  ("0.21", "2.6"),
    "2.7":  ("0.22", "2.7"),
    "2.8":  ("0.23", "2.8"),
    "2.9":  ("0.24", "2.9"),
    "2.10": ("0.25", "2.10"),
    "2.11": ("0.26", "2.11"),
    "2.12": ("0.27", "2.11"),   # torchaudio never shipped 2.12; stuck at 2.11
}


def derive_family_pins(torch_pin: str) -> Optional[tuple]:
    """Given a torch pin like '==2.11.0' or '==2.8.*', return
    `(torchvision_pin, torchaudio_pin)` as `==X.Y.*` specs.

    Prefers the bootstrap venv's actually-installed versions when torch_pin
    matches bootstrap torch — auto-tracks releases like the missing
    torchaudio 2.12 without needing a table update. Falls back to
    `TORCH_FAMILY_COMPAT` otherwise (bare envs, cuda-wheel-resolver
    fallback combos that differ from bootstrap, etc).
    """
    m = re.match(r"==\s*(\d+)\.(\d+)", torch_pin)
    if not m:
        return None
    minor_key = f"{m.group(1)}.{m.group(2)}"

    # Bootstrap-derived: authoritative when torch_pin matches bootstrap.
    try:
        from ..detection.cuda import (
            get_bootstrap_torch_version,
            get_bootstrap_torchvision_version,
            get_bootstrap_torchaudio_version,
        )
        bt_torch = get_bootstrap_torch_version()
        if bt_torch and bt_torch.startswith(f"{minor_key}."):
            bt_vision = get_bootstrap_torchvision_version()
            bt_audio = get_bootstrap_torchaudio_version()
            if bt_vision and bt_audio:
                v_mm = ".".join(bt_vision.split(".")[:2])
                a_mm = ".".join(bt_audio.split(".")[:2])
                return (f"=={v_mm}.*", f"=={a_mm}.*")
    except Exception:
        pass

    pair = TORCH_FAMILY_COMPAT.get(minor_key)
    if not pair:
        return None
    vision_minor, audio_minor = pair
    return (f"=={vision_minor}.*", f"=={audio_minor}.*")


def get_cuda_torch_mapping() -> dict:
    return CUDA_TORCH_MAP.copy()




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
    """Substrings a wheel filename must contain to be installable on THIS machine.

    The CPU architecture is load-bearing here, not a detail. Matching on bare
    "manylinux" accepts `...-manylinux_2_34_x86_64.whl` on an ARM host, so the
    probe reports the package as published, the combo resolves, and pip only
    refuses it later ("is not a supported wheel on this platform") -- a quiet
    failure a long way from its cause. Every wheel platform tag ends in the
    arch, for both the manylinux and the plain linux spellings, so matching the
    `_<arch>` suffix covers both and excludes the other architecture.
    """
    if sys.platform.startswith("linux"):
        return [f"_{cpu_arch()}"]
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
        index_url = f"{cuda_wheels_index()}{pkg_dir}/"
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
                    url = wheel_url if wheel_url.startswith("http") else f"{cuda_wheels_index()}{pkg_dir}/{wheel_url}"
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
            with urllib.request.urlopen(f"{cuda_wheels_index()}{pkg_dir}/", timeout=10) as resp:
                html = resp.read().decode("utf-8")
            for match in link_pattern.finditer(html):
                name = match.group(1).replace("%2B", "+")
                if name not in wheels: wheels.append(name)
        except Exception: continue
    return wheels


def _version_key(version: str):
    """Sortable key for wheel version strings: numeric segments compare as
    numbers ('1.10' > '1.9', '0.0.1' < '1.0' -- string comparison gets both
    wrong) and non-numeric segments sort BELOW numeric ones, so pre-release
    suffixes order under the release ('2.0rc1' < '2.0', PEP-440-like)."""
    main = version.split("+", 1)[0]
    key = []
    for piece in re.split(r"[._-]", main):
        if piece.isdigit():
            key.append((1, int(piece), ""))
        else:
            key.append((0, 0, piece))
    return key


def find_matching_wheel(package: str, torch_version: str, cuda_version: str) -> Optional[str]:
    """Find wheel matching CUDA/torch version, return version spec."""
    cuda_short = cuda_version.replace(".", "")[:3]
    torch_short = ".".join(torch_version.split(".")[:2])
    local_patterns = [f"+cu{cuda_short}torch{torch_short}", f"+pt{torch_short}cu{cuda_short}"]
    wheel_pattern = re.compile(r'href="[^"]*?([^"/]+\.whl)"', re.IGNORECASE)

    for pkg_dir in _pkg_variants(package):
        try:
            with urllib.request.urlopen(f"{cuda_wheels_index()}{pkg_dir}/", timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception: continue

        best_match = best_version = None
        for match in wheel_pattern.finditer(html):
            wheel_name = match.group(1).replace("%2B", "+")
            for local in local_patterns:
                if local in wheel_name:
                    parts = wheel_name.split("-")
                    if len(parts) >= 2 and (
                        best_version is None
                        or _version_key(parts[1]) > _version_key(best_version)
                    ):
                        best_version = parts[1]
                        best_match = f"{package}==={parts[1]}"
                    break
        if best_match: return best_match
    return None


def get_find_links_urls(package: str) -> List[str]:
    return [f"{cuda_wheels_index()}{p}/" for p in _pkg_variants(package)]
