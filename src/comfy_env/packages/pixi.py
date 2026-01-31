"""Pixi package manager integration. See: https://pixi.sh/"""

import platform as platform_mod
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

PIXI_URLS = {
    ("Linux", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}


def get_pixi_path() -> Optional[Path]:
    """Find pixi in PATH or common locations."""
    if cmd := shutil.which("pixi"): return Path(cmd)
    home = Path.home()
    for p in [home / ".pixi/bin/pixi", home / ".local/bin/pixi"]:
        candidate = p.with_suffix(".exe") if sys.platform == "win32" else p
        if candidate.exists(): return candidate
    return None


def ensure_pixi(install_dir: Optional[Path] = None, log: Callable[[str], None] = print) -> Path:
    """Ensure pixi is installed, downloading if necessary."""
    if existing := get_pixi_path(): return existing

    log("Pixi not found, downloading...")
    install_dir = install_dir or Path.home() / ".local/bin"
    install_dir.mkdir(parents=True, exist_ok=True)

    system, machine = platform_mod.system(), platform_mod.machine()
    if machine in ("x86_64", "AMD64"): machine = "x86_64" if system != "Windows" else "AMD64"
    elif machine in ("arm64", "aarch64"): machine = "arm64" if system == "Darwin" else "aarch64"

    if (system, machine) not in PIXI_URLS:
        raise RuntimeError(f"No pixi for {system}/{machine}")

    pixi_path = install_dir / ("pixi.exe" if system == "Windows" else "pixi")
    try:
        urllib.request.urlretrieve(PIXI_URLS[(system, machine)], pixi_path)
    except Exception as e:
        result = subprocess.run(["curl", "-fsSL", "-o", str(pixi_path), PIXI_URLS[(system, machine)]], capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(f"Failed to download pixi") from e

    if system != "Windows":
        pixi_path.chmod(pixi_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    log(f"Installed pixi: {pixi_path}")
    return pixi_path


def get_pixi_python(node_dir: Path) -> Optional[Path]:
    """Get Python path from pixi environment."""
    from ..environment.cache import resolve_env_path
    env_path, _, _ = resolve_env_path(node_dir)
    if not env_path: return None
    python_path = env_path / ("python.exe" if sys.platform == "win32" else "bin/python")
    return python_path if python_path.exists() else None


def pixi_install(node_dir: Path, log: Callable[[str], None] = print) -> subprocess.CompletedProcess:
    pixi_path = get_pixi_path()
    if not pixi_path: raise RuntimeError("Pixi not found")
    return subprocess.run([str(pixi_path), "install"], cwd=node_dir, capture_output=True, text=True)


def pixi_run(command: List[str], node_dir: Path, log: Callable[[str], None] = print) -> subprocess.CompletedProcess:
    pixi_path = get_pixi_path()
    if not pixi_path: raise RuntimeError("Pixi not found")
    return subprocess.run([str(pixi_path), "run"] + command, cwd=node_dir, capture_output=True, text=True)


def pixi_clean(node_dir: Path, log: Callable[[str], None] = print) -> None:
    """Remove pixi artifacts (pixi.toml, pixi.lock, .pixi/)."""
    for path in [node_dir / "pixi.toml", node_dir / "pixi.lock"]:
        if path.exists(): path.unlink()
    if (node_dir / ".pixi").exists(): shutil.rmtree(node_dir / ".pixi")
