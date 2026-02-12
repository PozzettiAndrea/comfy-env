"""Pixi package manager integration. See: https://pixi.sh/"""

import platform as platform_mod
import shutil
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

_PIXI_MANIFEST_XML = """\
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1"
    xmlns:asmv3="urn:schemas-microsoft-com:asm.v3" manifestVersion="1.0">
    <asmv3:application>
        <asmv3:windowsSettings>
            <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
        </asmv3:windowsSettings>
    </asmv3:application>
</assembly>
"""


def _ensure_longpath_manifest(pixi_path: Path, log: Callable[[str], None] = print) -> None:
    """Write an external Windows manifest enabling long path support next to pixi.exe."""
    if sys.platform != "win32":
        return
    manifest_path = pixi_path.parent / (pixi_path.name + ".manifest")
    if manifest_path.exists():
        return
    try:
        manifest_path.write_text(_PIXI_MANIFEST_XML, encoding="utf-8")
        log(f"[comfy-env] Wrote long path manifest: {manifest_path}")
    except OSError:
        log(
            f"[comfy-env] ERROR: Cannot write {manifest_path} -- "
            f"pixi will hit the 260-char path limit on Windows. "
            f"Move pixi to a user-writable directory or fix permissions."
        )


PIXI_URLS = {
    ("Linux", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl",
    ("Linux", "aarch64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-unknown-linux-musl",
    ("Darwin", "x86_64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-apple-darwin",
    ("Darwin", "arm64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-aarch64-apple-darwin",
    ("Windows", "AMD64"): "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe",
}


def get_pixi_path() -> Optional[Path]:
    """Find pixi in PATH, venv, or common user locations."""
    # 1. PATH
    if cmd := shutil.which("pixi"):
        return Path(cmd)

    # 2. Active venv (pip-installed pixi)
    prefix = Path(sys.prefix)
    candidates = []
    if sys.platform == "win32":
        candidates.append(prefix / "Scripts" / "pixi.exe")
    else:
        candidates.append(prefix / "bin" / "pixi")

    # 3. User locations
    home = Path.home()
    candidates.extend([
        home / ".pixi" / "bin" / ("pixi.exe" if sys.platform == "win32" else "pixi"),
        home / ".local" / "bin" / ("pixi.exe" if sys.platform == "win32" else "pixi"),
    ])

    for c in candidates:
        if c.exists():
            return c

    return None



def ensure_pixi(install_dir: Optional[Path] = None, log: Callable[[str], None] = print) -> Path:
    """Ensure pixi is installed, downloading if necessary."""
    if existing := get_pixi_path():
        _ensure_longpath_manifest(existing, log)
        return existing

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

    if system == "Windows":
        _ensure_longpath_manifest(pixi_path, log)
    else:
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
    """Remove pixi artifacts (pixi.toml, pixi.lock, .pixi/) and global cache."""
    for path in [node_dir / "pixi.toml", node_dir / "pixi.lock"]:
        if path.exists(): path.unlink()
    if (node_dir / ".pixi").exists(): shutil.rmtree(node_dir / ".pixi")
    # Also nuke global pixi cache to avoid migration issues
    global_pixi = Path.home() / ".pixi"
    if global_pixi.exists(): shutil.rmtree(global_pixi, ignore_errors=True)
