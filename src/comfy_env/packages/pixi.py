"""
Pixi package manager integration.

Pixi is a fast package manager that supports both conda and pip packages.
See: https://pixi.sh/
"""

import platform as platform_mod
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

# Pixi version requirement (for future use)
PIXI_VERSION = "0.35.0"

# Pixi download URLs by platform
PIXI_URLS = {
    ("Linux", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}


def get_pixi_path() -> Optional[Path]:
    """
    Find the pixi executable.

    Searches:
    1. System PATH
    2. ~/.pixi/bin/pixi
    3. ~/.local/bin/pixi

    Returns:
        Path to pixi executable, or None if not found.
    """
    # Check system PATH first
    pixi_cmd = shutil.which("pixi")
    if pixi_cmd:
        return Path(pixi_cmd)

    # Check common install locations
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
    """
    Ensure pixi is installed, downloading if necessary.

    Args:
        install_dir: Directory to install pixi into. Defaults to ~/.local/bin.
        log: Logging callback.

    Returns:
        Path to pixi executable.

    Raises:
        RuntimeError: If pixi cannot be downloaded for this platform.
    """
    existing = get_pixi_path()
    if existing:
        return existing

    log("Pixi not found, downloading...")

    if install_dir is None:
        install_dir = Path.home() / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    system = platform_mod.system()
    machine = platform_mod.machine()

    # Normalize machine name
    if machine in ("x86_64", "AMD64"):
        machine = "x86_64" if system != "Windows" else "AMD64"
    elif machine in ("arm64", "aarch64"):
        machine = "arm64" if system == "Darwin" else "aarch64"

    url_key = (system, machine)
    if url_key not in PIXI_URLS:
        raise RuntimeError(f"No pixi download available for {system}/{machine}")

    url = PIXI_URLS[url_key]
    pixi_path = install_dir / ("pixi.exe" if system == "Windows" else "pixi")

    # Try urllib first, fall back to curl
    try:
        urllib.request.urlretrieve(url, pixi_path)
    except Exception as e:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(pixi_path), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download pixi: {result.stderr}") from e

    # Make executable on Unix
    if system != "Windows":
        pixi_path.chmod(pixi_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log(f"Installed pixi to: {pixi_path}")
    return pixi_path


def get_pixi_python(node_dir: Path) -> Optional[Path]:
    """
    Get path to Python in the pixi environment.

    Checks multiple locations:
    1. Central cache via marker file
    2. Local _env_<name> directory
    3. .pixi/envs/default (legacy)

    Args:
        node_dir: Node directory to check.

    Returns:
        Path to Python executable, or None if not found.
    """
    from ..environment.cache import resolve_env_path

    env_path, _, _ = resolve_env_path(node_dir)
    if not env_path:
        return None

    if sys.platform == "win32":
        python_path = env_path / "python.exe"
    else:
        python_path = env_path / "bin" / "python"

    return python_path if python_path.exists() else None


def pixi_install(
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> subprocess.CompletedProcess:
    """
    Run pixi install in the given directory.

    Args:
        node_dir: Directory containing pixi.toml.
        log: Logging callback.

    Returns:
        CompletedProcess with result.

    Raises:
        RuntimeError: If pixi is not installed.
    """
    pixi_path = get_pixi_path()
    if not pixi_path:
        raise RuntimeError("Pixi not found. Run ensure_pixi() first.")

    return subprocess.run(
        [str(pixi_path), "install"],
        cwd=node_dir,
        capture_output=True,
        text=True,
    )


def pixi_run(
    command: List[str],
    node_dir: Path,
    log: Callable[[str], None] = print,
) -> subprocess.CompletedProcess:
    """
    Run a command in the pixi environment.

    Args:
        command: Command and arguments to run.
        node_dir: Directory containing pixi.toml.
        log: Logging callback.

    Returns:
        CompletedProcess with result.

    Raises:
        RuntimeError: If pixi is not installed.
    """
    pixi_path = get_pixi_path()
    if not pixi_path:
        raise RuntimeError("Pixi not found")

    return subprocess.run(
        [str(pixi_path), "run"] + command,
        cwd=node_dir,
        capture_output=True,
        text=True,
    )


def pixi_clean(node_dir: Path, log: Callable[[str], None] = print) -> None:
    """
    Remove pixi installation artifacts from a directory.

    Removes:
    - pixi.toml
    - pixi.lock
    - .pixi directory

    Args:
        node_dir: Directory to clean.
        log: Logging callback.
    """
    for path in [node_dir / "pixi.toml", node_dir / "pixi.lock"]:
        if path.exists():
            path.unlink()

    pixi_dir = node_dir / ".pixi"
    if pixi_dir.exists():
        shutil.rmtree(pixi_dir)
