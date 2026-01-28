"""
Pixi integration for comfy-env.

Pixi is a fast package manager that supports both conda and pip packages.
All dependencies go through pixi for unified management.

See: https://pixi.sh/
"""

import copy
import platform
import shutil
import stat
import subprocess
import sys
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

# CUDA wheels index
CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels/"

# CUDA version -> PyTorch version mapping
CUDA_TORCH_MAP = {
    "12.8": "2.8",
    "12.4": "2.4",
    "12.1": "2.4",
}


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


def clean_pixi_artifacts(node_dir: Path, log: Callable[[str], None] = print) -> None:
    """Remove previous pixi installation artifacts."""
    for path in [node_dir / "pixi.toml", node_dir / "pixi.lock"]:
        if path.exists():
            path.unlink()
    pixi_dir = node_dir / ".pixi"
    if pixi_dir.exists():
        shutil.rmtree(pixi_dir)


def get_pixi_python(node_dir: Path) -> Optional[Path]:
    """Get path to Python in the pixi environment."""
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
    create_env_link: bool = False,
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
        create_env_link: If True, create _env_<name> symlink for isolation.

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
            log(f"Detected CUDA {cuda_version} → PyTorch {torch_version}")
        else:
            log("Warning: CUDA packages requested but no GPU detected")

    # Clean previous artifacts
    clean_pixi_artifacts(node_dir, log)

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

    # Add pypi-options for CUDA wheels
    if cfg.has_cuda and cuda_version:
        pypi_options = pixi_data.get("pypi-options", {})
        # Merge find-links (pixi expects [{url: "..."}, ...] format)
        find_links = pypi_options.get("find-links", [])
        existing_urls = {
            entry.get("url") if isinstance(entry, dict) else entry
            for entry in find_links
        }
        if CUDA_WHEELS_INDEX not in existing_urls:
            find_links.append({"url": CUDA_WHEELS_INDEX})
        # Normalize any plain strings to {url: ...} format
        find_links = [
            {"url": entry} if isinstance(entry, str) else entry
            for entry in find_links
        ]
        pypi_options["find-links"] = find_links
        # Merge extra-index-urls
        cuda_short = cuda_version.replace(".", "")[:3]
        pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_short}"
        extra_urls = pypi_options.get("extra-index-urls", [])
        if pytorch_index not in extra_urls:
            extra_urls.append(pytorch_index)
        pypi_options["extra-index-urls"] = extra_urls
        pixi_data["pypi-options"] = pypi_options

    # Build pypi-dependencies section (CUDA packages excluded - installed separately)
    pypi_deps = pixi_data.get("pypi-dependencies", {})

    # Add torch if we have CUDA packages
    if cfg.has_cuda and torch_version:
        torch_major = torch_version.split(".")[0]
        torch_minor = int(torch_version.split(".")[1])
        pypi_deps.setdefault("torch", f">={torch_version},<{torch_major}.{torch_minor + 1}")

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

    # Install CUDA packages with --no-deps (avoids PyPI version conflicts)
    if cfg.cuda_packages and cuda_version:
        log(f"Installing CUDA packages with --no-deps: {cfg.cuda_packages}")
        python_path = get_pixi_python(node_dir)
        if not python_path:
            raise RuntimeError("Could not find Python in pixi environment")

        pip_cmd = [
            str(python_path), "-m", "pip", "install",
            "--no-deps",
            "--index-url", CUDA_WHEELS_INDEX,
        ] + cfg.cuda_packages

        result = subprocess.run(pip_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"CUDA package install failed:\n{result.stderr}")
            raise RuntimeError(f"CUDA package install failed: {result.stderr}")
        log("CUDA packages installed")

    # Create symlink/junction to _env_<name> for discovery (only for isolated subdirs)
    if create_env_link:
        env_dir = node_dir / ".pixi" / "envs" / "default"
        env_link = node_dir / f"_env_{node_dir.name}"
        if env_dir.exists():
            # Remove existing link/dir if present
            if env_link.is_symlink() or env_link.exists():
                if env_link.is_symlink():
                    env_link.unlink()
                else:
                    shutil.rmtree(env_link)
            # Create symlink (Linux/Mac) or junction (Windows)
            if sys.platform == "win32":
                # Use junction on Windows
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(env_link), str(env_dir)],
                    capture_output=True,
                )
            else:
                env_link.symlink_to(env_dir)
            log(f"Linked: {env_link.name} -> .pixi/envs/default")

    log("Installation complete!")
    return True
