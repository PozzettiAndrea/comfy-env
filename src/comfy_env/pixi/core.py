"""
Pixi integration for comfy-env.

Pixi is a fast package manager that supports both conda and pip packages.
All dependencies go through pixi for unified management.

See: https://pixi.sh/
"""

import copy
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config.types import ComfyEnvConfig


# Pixi download URLs by platform
PIXI_URLS = {
    ("Linux", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}

# CUDA wheels index (includes flash-attn, PyG packages, and custom wheels)
CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/"

# CUDA version -> PyTorch version mapping
CUDA_TORCH_MAP = {
    "12.8": "2.8",
    "12.4": "2.4",
    "12.1": "2.4",
}

def find_wheel_url(
    package: str,
    torch_version: str,
    cuda_version: str,
    python_version: str,
) -> Optional[str]:
    """
    Query cuda-wheels index and return the direct URL for the matching wheel.

    This bypasses pip's version validation by providing a direct URL,
    which is necessary for wheels where the filename has a local version
    but the internal METADATA doesn't (e.g., flash-attn from mjun0812).

    Args:
        package: Package name (e.g., "flash-attn")
        torch_version: PyTorch version (e.g., "2.8")
        cuda_version: CUDA version (e.g., "12.8")
        python_version: Python version (e.g., "3.10")

    Returns:
        Direct URL to the wheel file, or None if no match found.
    """
    cuda_short = cuda_version.replace(".", "")[:3]  # "12.8" -> "128"
    torch_short = torch_version.replace(".", "")[:2]  # "2.8" -> "28"
    py_tag = f"cp{python_version.replace('.', '')}"  # "3.10" -> "cp310"

    # Platform tag for current system
    if sys.platform == "linux":
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

    pkg_variants = [package, package.replace("-", "_"), package.replace("_", "-")]

    for pkg_dir in pkg_variants:
        index_url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        try:
            with urllib.request.urlopen(index_url, timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue

        # Parse href and display name from HTML: <a href="URL">DISPLAY_NAME</a>
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


def find_matching_wheel(package: str, torch_version: str, cuda_version: str) -> Optional[str]:
    """
    Query cuda-wheels index to find a wheel matching the CUDA/torch version.
    Returns the full version spec (e.g., "flash-attn===2.8.3+cu128torch2.8") or None.

    Note: This is used as a fallback for packages with correct wheel metadata.
    For packages with mismatched metadata (like flash-attn), use find_wheel_url() instead.
    """
    cuda_short = cuda_version.replace(".", "")[:3]  # "12.8" -> "128"
    torch_short = torch_version.replace(".", "")[:2]  # "2.8" -> "28"

    # Try different directory name variants
    pkg_variants = [package, package.replace("-", "_"), package.replace("_", "-")]

    for pkg_dir in pkg_variants:
        url = f"{CUDA_WHEELS_INDEX}{pkg_dir}/"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                html = resp.read().decode("utf-8")
        except Exception:
            continue

        # Parse wheel filenames from href attributes
        # Pattern: package_name-version+localversion-cpXX-cpXX-platform.whl
        wheel_pattern = re.compile(
            r'href="[^"]*?([^"/]+\.whl)"',
            re.IGNORECASE
        )

        # Local version patterns to match:
        # cuda-wheels style: +cu128torch28
        # PyG style: +pt28cu128
        local_patterns = [
            f"+cu{cuda_short}torch{torch_short}",  # cuda-wheels style
            f"+pt{torch_short}cu{cuda_short}",     # PyG style
        ]

        best_match = None
        best_version = None

        for match in wheel_pattern.finditer(html):
            wheel_name = match.group(1)
            # URL decode
            wheel_name = wheel_name.replace("%2B", "+")

            # Check if wheel matches our CUDA/torch version
            for local_pattern in local_patterns:
                if local_pattern in wheel_name:
                    # Extract version from wheel name
                    # Format: name-version+local-cpXX-cpXX-platform.whl
                    parts = wheel_name.split("-")
                    if len(parts) >= 2:
                        version_part = parts[1]  # e.g., "2.8.3+cu128torch2.8"
                        if best_version is None or version_part > best_version:
                            best_version = version_part
                            best_match = f"{package}==={version_part}"
                    break

        if best_match:
            return best_match

    return None


def get_package_spec(package: str, torch_version: str, cuda_version: str) -> str:
    """
    Get package spec with local version for CUDA/torch compatibility.
    Queries the index to find matching wheels dynamically.
    """
    spec = find_matching_wheel(package, torch_version, cuda_version)
    return spec if spec else package


def get_all_find_links(package: str, torch_version: str, cuda_version: str) -> list:
    """Get all find-links URLs for a CUDA package."""
    # Try both underscore and hyphen variants since directory naming is inconsistent
    pkg_underscore = package.replace("-", "_")
    pkg_hyphen = package.replace("_", "-")
    urls = [f"{CUDA_WHEELS_INDEX}{package}/"]
    if pkg_underscore != package:
        urls.append(f"{CUDA_WHEELS_INDEX}{pkg_underscore}/")
    if pkg_hyphen != package:
        urls.append(f"{CUDA_WHEELS_INDEX}{pkg_hyphen}/")
    return urls


def get_current_platform() -> str:
    """Get the current platform string for pixi."""
    if sys.platform == "linux":
        return "linux-64"
    elif sys.platform == "darwin":
        return "osx-arm64" if platform.machine() == "arm64" else "osx-64"
    elif sys.platform == "win32":
        return "win-64"
    return "linux-64"


def get_pixi_path() -> Optional[Path]:
    """Find the pixi executable."""
    pixi_cmd = shutil.which("pixi")
    if pixi_cmd:
        return Path(pixi_cmd)

    home = Path.home()
    candidates = [
        home / ".pixi" / "bin" / "pixi",
        home / ".local" / "bin" / "pixi",
    ]

    if sys.platform == "win32":
        candidates = [p.with_suffix(".exe") for p in candidates]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def ensure_pixi(
    install_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Ensure pixi is installed, downloading if necessary."""
    existing = get_pixi_path()
    if existing:
        return existing

    log("Pixi not found, downloading...")

    if install_dir is None:
        install_dir = Path.home() / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    system = platform.system()
    machine = platform.machine()

    if machine in ("x86_64", "AMD64"):
        machine = "x86_64" if system != "Windows" else "AMD64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64" if system == "Darwin" else "aarch64"

    url_key = (system, machine)
    if url_key not in PIXI_URLS:
        raise RuntimeError(f"No pixi download for {system}/{machine}")

    url = PIXI_URLS[url_key]
    pixi_path = install_dir / ("pixi.exe" if system == "Windows" else "pixi")

    try:
        import urllib.request
        urllib.request.urlretrieve(url, pixi_path)
    except Exception as e:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(pixi_path), url],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download pixi: {result.stderr}") from e

    if system != "Windows":
        pixi_path.chmod(pixi_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log(f"Installed pixi to: {pixi_path}")
    return pixi_path


def get_env_name(dir_name: str) -> str:
    """Convert directory name to env name: ComfyUI-UniRig -> _env_unirig"""
    name = dir_name.lower().replace("-", "_").lstrip("comfyui_")
    return f"_env_{name}"


def clean_pixi_artifacts(node_dir: Path, log: Callable[[str], None] = print) -> None:
    """Remove previous pixi installation artifacts."""
    for path in [node_dir / "pixi.toml", node_dir / "pixi.lock"]:
        if path.exists():
            path.unlink()
    pixi_dir = node_dir / ".pixi"
    if pixi_dir.exists():
        shutil.rmtree(pixi_dir)
    # Also clean old _env_* directories
    env_name = get_env_name(node_dir.name)
    env_dir = node_dir / env_name
    if env_dir.exists():
        shutil.rmtree(env_dir)


def get_pixi_python(node_dir: Path) -> Optional[Path]:
    """Get path to Python in the pixi environment."""
    # Check new _env_<name> location first
    env_name = get_env_name(node_dir.name)
    env_dir = node_dir / env_name
    if not env_dir.exists():
        # Fallback to old .pixi path
        env_dir = node_dir / ".pixi" / "envs" / "default"
    if sys.platform == "win32":
        python_path = env_dir / "python.exe"
    else:
        python_path = env_dir / "bin" / "python"
    return python_path if python_path.exists() else None


def pixi_run(
    command: List[str],
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> subprocess.CompletedProcess:
    """Run a command in the pixi environment."""
    pixi_path = get_pixi_path()
    if not pixi_path:
        raise RuntimeError("Pixi not found")
    return subprocess.run(
        [str(pixi_path), "run"] + command,
        cwd=node_dir,
        capture_output=True,
        text=True,
    )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts, override wins for conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def pixi_install(
    cfg: ComfyEnvConfig,
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> bool:
    """
    Install all packages via pixi.

    comfy-env.toml is a superset of pixi.toml. This function:
    1. Starts with passthrough sections from comfy-env.toml
    2. Adds workspace metadata (name, version, channels, platforms)
    3. Adds system-requirements if needed (CUDA detection)
    4. Adds CUDA find-links and PyTorch if [cuda] packages present
    5. Writes combined data as pixi.toml

    Args:
        cfg: ComfyEnvConfig with packages to install.
        node_dir: Directory to install in.
        log: Logging callback.

    Returns:
        True if installation succeeded.
    """
    try:
        import tomli_w
    except ImportError:
        raise ImportError(
            "tomli-w required for writing TOML. Install with: pip install tomli-w"
        )

    from .cuda_detection import get_recommended_cuda_version

    # Start with passthrough data from comfy-env.toml
    pixi_data = copy.deepcopy(cfg.pixi_passthrough)

    # Detect CUDA version if CUDA packages requested
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

    # Install system dependencies on Linux via apt
    if sys.platform == "linux" and cfg.apt_packages:
        log(f"Installing apt packages: {cfg.apt_packages}")
        subprocess.run(["sudo", "apt-get", "update"], capture_output=True)
        subprocess.run(
            ["sudo", "apt-get", "install", "-y"] + cfg.apt_packages,
            capture_output=True,
        )

    # Clean previous artifacts
    clean_pixi_artifacts(node_dir, log)

    # Create .pixi/config.toml to ensure inline (non-detached) environments
    pixi_config_dir = node_dir / ".pixi"
    pixi_config_dir.mkdir(parents=True, exist_ok=True)
    pixi_config_file = pixi_config_dir / "config.toml"
    pixi_config_file.write_text("detached-environments = false\n")

    # Ensure pixi is installed
    pixi_path = ensure_pixi(log=log)

    # Build workspace section
    workspace = pixi_data.get("workspace", {})
    workspace.setdefault("name", node_dir.name)
    workspace.setdefault("version", "0.1.0")
    workspace.setdefault("channels", ["conda-forge"])
    workspace.setdefault("platforms", [get_current_platform()])
    pixi_data["workspace"] = workspace

    # Build system-requirements section
    system_reqs = pixi_data.get("system-requirements", {})
    if sys.platform == "linux":
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
    dependencies.setdefault("pip", "*")  # Always include pip
    pixi_data["dependencies"] = dependencies

    # Add pypi-options for PyTorch index (CUDA packages installed separately via pip)
    if cfg.has_cuda and cuda_version:
        pypi_options = pixi_data.get("pypi-options", {})
        # Add PyTorch CUDA index for torch installation
        cuda_short = cuda_version.replace(".", "")[:3]
        pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_short}"
        extra_urls = pypi_options.get("extra-index-urls", [])
        if pytorch_index not in extra_urls:
            extra_urls.append(pytorch_index)
        pypi_options["extra-index-urls"] = extra_urls
        pixi_data["pypi-options"] = pypi_options

    # Build pypi-dependencies section (CUDA packages excluded - installed separately)
    pypi_deps = pixi_data.get("pypi-dependencies", {})

    # Enforce torch version if we have CUDA packages (must match cuda_packages wheels)
    if cfg.has_cuda and torch_version:
        torch_major = torch_version.split(".")[0]
        torch_minor = int(torch_version.split(".")[1])
        required_torch = f">={torch_version},<{torch_major}.{torch_minor + 1}"
        if "torch" in pypi_deps and pypi_deps["torch"] != required_torch:
            log(f"Overriding torch={pypi_deps['torch']} with {required_torch} (required for cuda_packages)")
        pypi_deps["torch"] = required_torch

    # NOTE: CUDA packages are NOT added here - they're installed with --no-deps after pixi

    if pypi_deps:
        pixi_data["pypi-dependencies"] = pypi_deps

    # Write pixi.toml
    pixi_toml = node_dir / "pixi.toml"
    with open(pixi_toml, "wb") as f:
        tomli_w.dump(pixi_data, f)
    log(f"Generated {pixi_toml}")

    # Run pixi install
    log("Running pixi install...")
    result = subprocess.run(
        [str(pixi_path), "install"],
        cwd=node_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"pixi install failed:\n{result.stderr}")
        raise RuntimeError(f"pixi install failed: {result.stderr}")

    # Install CUDA packages via direct URL or find-links fallback
    if cfg.cuda_packages and cuda_version:
        log(f"Installing CUDA packages: {cfg.cuda_packages}")
        python_path = get_pixi_python(node_dir)
        if not python_path:
            raise RuntimeError("Could not find Python in pixi environment")

        # Get Python version from the pixi environment (not host Python)
        result = subprocess.run(
            [str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            py_version = result.stdout.strip()
        else:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            log(f"Warning: Could not detect pixi Python version, using host: {py_version}")

        for package in cfg.cuda_packages:
            # Find direct wheel URL (bypasses metadata validation)
            wheel_url = find_wheel_url(package, torch_version, cuda_version, py_version)

            if not wheel_url:
                raise RuntimeError(
                    f"No wheel found for {package} with CUDA {cuda_version}, "
                    f"torch {torch_version}, Python {py_version}. "
                    f"Check cuda-wheels index."
                )

            log(f"  Installing {package} from {wheel_url}")
            pip_cmd = [
                str(python_path), "-m", "pip", "install",
                "--no-deps",
                "--no-cache-dir",
                wheel_url,
            ]

            result = subprocess.run(pip_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                log(f"CUDA package install failed for {package}:\n{result.stderr}")
                raise RuntimeError(f"CUDA package install failed: {result.stderr}")

        log("CUDA packages installed")

    # Move environment from .pixi/envs/default to _env_<name>
    old_env = node_dir / ".pixi" / "envs" / "default"
    env_name = get_env_name(node_dir.name)
    new_env = node_dir / env_name

    if old_env.exists():
        if new_env.exists():
            shutil.rmtree(new_env)  # Clean old env
        shutil.move(str(old_env), str(new_env))
        # Clean up .pixi directory (keep pixi.toml and pixi.lock)
        pixi_dir = node_dir / ".pixi"
        if pixi_dir.exists():
            shutil.rmtree(pixi_dir)
        log(f"Moved environment to {new_env}")

    log("Installation complete!")
    return True
