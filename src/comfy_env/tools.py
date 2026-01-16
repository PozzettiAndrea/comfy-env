"""
Tool installers for external dependencies like Blender.

Usage in comfy-env.toml:
    [tools]
    blender = "4.2"
"""

import os
import platform
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Callable, Optional
from urllib.request import urlretrieve

from .env.config import ToolConfig

# Default install location
DEFAULT_TOOLS_DIR = Path.home() / ".comfy-env" / "tools"

# Blender download URLs by platform and version
BLENDER_DOWNLOADS = {
    "4.2": {
        "linux": "https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz",
        "windows": "https://download.blender.org/release/Blender4.2/blender-4.2.0-windows-x64.zip",
        "darwin": "https://download.blender.org/release/Blender4.2/blender-4.2.0-macos-arm64.dmg",
    },
    "4.3": {
        "linux": "https://download.blender.org/release/Blender4.3/blender-4.3.0-linux-x64.tar.xz",
        "windows": "https://download.blender.org/release/Blender4.3/blender-4.3.0-windows-x64.zip",
        "darwin": "https://download.blender.org/release/Blender4.3/blender-4.3.0-macos-arm64.dmg",
    },
}


def get_platform() -> str:
    """Get current platform name."""
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    elif system == "darwin":
        return "darwin"
    return system


def install_tool(
    config: ToolConfig,
    log: Callable[[str], None] = print,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Install a tool based on its config.

    Args:
        config: Tool configuration
        log: Logging callback
        base_dir: Base directory for tools. Tools install to base_dir/tools/<name>/
    """
    if config.name.lower() == "blender":
        return install_blender(config.version, log, config.install_dir or base_dir)
    else:
        log(f"Unknown tool: {config.name}")
        return None


def install_blender(
    version: str = "4.2",
    log: Callable[[str], None] = print,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Install Blender to the specified directory.

    Args:
        version: Blender version to install (e.g., "4.2")
        log: Logging callback
        base_dir: Base directory. Blender installs to base_dir/tools/blender/
                  If None, uses ~/.comfy-env/tools/blender/

    Returns path to blender executable if successful.
    """
    plat = get_platform()
    if base_dir:
        install_dir = base_dir / "tools" / "blender"
    else:
        install_dir = DEFAULT_TOOLS_DIR / "blender"

    # Check if already installed
    exe = find_blender(install_dir)
    if exe:
        log(f"Blender already installed: {exe}")
        return exe

    # Get download URL
    if version not in BLENDER_DOWNLOADS:
        log(f"Unknown Blender version: {version}. Available: {list(BLENDER_DOWNLOADS.keys())}")
        return None

    urls = BLENDER_DOWNLOADS[version]
    if plat not in urls:
        log(f"Blender {version} not available for {plat}")
        return None

    url = urls[plat]
    log(f"Downloading Blender {version} for {plat}...")

    install_dir.mkdir(parents=True, exist_ok=True)
    archive_name = url.split("/")[-1]
    archive_path = install_dir / archive_name

    try:
        # Download
        urlretrieve(url, archive_path)
        log(f"Downloaded to {archive_path}")

        # Extract
        log("Extracting...")
        if archive_name.endswith(".tar.xz"):
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(install_dir)
        elif archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(install_dir)
        elif archive_name.endswith(".dmg"):
            # macOS DMG requires special handling
            log("macOS DMG installation not yet automated. Please install manually.")
            archive_path.unlink()
            return None

        # Clean up archive
        archive_path.unlink()

        # Find the executable
        exe = find_blender(install_dir)
        if exe:
            log(f"Blender installed: {exe}")
            return exe
        else:
            log("Blender extracted but executable not found")
            return None

    except Exception as e:
        log(f"Failed to install Blender: {e}")
        if archive_path.exists():
            archive_path.unlink()
        return None


def find_blender(search_dir: Optional[Path] = None) -> Optional[Path]:
    """Find Blender executable."""
    # Check PATH first
    blender_in_path = shutil.which("blender")
    if blender_in_path:
        return Path(blender_in_path)

    # Check common locations
    plat = get_platform()
    search_paths = []

    if search_dir:
        search_paths.append(search_dir)

    search_paths.append(DEFAULT_TOOLS_DIR / "blender")

    if plat == "windows":
        search_paths.extend([
            Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Blender Foundation",
            Path.home() / "AppData" / "Local" / "Blender Foundation",
        ])
    elif plat == "linux":
        search_paths.extend([
            Path("/opt/blender"),
            Path.home() / "blender",
        ])
    elif plat == "darwin":
        search_paths.append(Path("/Applications/Blender.app/Contents/MacOS"))

    exe_name = "blender.exe" if plat == "windows" else "blender"

    for base in search_paths:
        if not base.exists():
            continue
        # Direct check
        exe = base / exe_name
        if exe.exists():
            return exe
        # Search subdirectories (for extracted archives like blender-4.2.0-linux-x64/)
        for subdir in base.iterdir():
            if subdir.is_dir():
                exe = subdir / exe_name
                if exe.exists():
                    return exe

    return None


def ensure_blender(
    version: str = "4.2",
    log: Callable[[str], None] = print,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Ensure Blender is installed, installing if necessary.

    Args:
        version: Blender version to install
        log: Logging callback
        base_dir: Base directory. Searches/installs in base_dir/tools/blender/
    """
    if base_dir:
        search_dir = base_dir / "tools" / "blender"
    else:
        search_dir = DEFAULT_TOOLS_DIR / "blender"

    exe = find_blender(search_dir)
    if exe:
        return exe
    return install_blender(version, log, base_dir)
