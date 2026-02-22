"""Environment path utilities."""

import shutil
from pathlib import Path
from typing import Optional, Tuple

from .cache import resolve_env_path as _resolve_env_path


def get_comfyui_dir(node_dir: Optional[Path] = None) -> Optional[Path]:
    """Find the ComfyUI base directory.

    Resolution order:
    1. folder_paths.base_path (canonical, available at prestartup time)
    2. Walk up from node_dir looking for ComfyUI markers (main.py + comfy/)
    """
    # 1. Use folder_paths (always available during prestartup)
    try:
        import folder_paths
        return Path(folder_paths.base_path)
    except ImportError:
        pass

    # 2. Walk up from node_dir
    if node_dir is not None:
        current = Path(node_dir).resolve()
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            current = current.parent

    return None


def get_site_packages_path(node_dir: Path) -> Optional[Path]:
    _, site_packages, _ = _resolve_env_path(node_dir)
    return site_packages


def get_lib_path(node_dir: Path) -> Optional[Path]:
    _, _, lib_dir = _resolve_env_path(node_dir)
    return lib_dir


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    return _resolve_env_path(node_dir)


def copy_files(src, dst, pattern: str = "*", overwrite: bool = False) -> int:
    """Copy files matching pattern from src to dst."""
    src, dst = Path(src), Path(dst)
    if not src.exists(): return 0

    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src.glob(pattern):
        if f.is_file():
            target = dst / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not target.exists():
                shutil.copy2(f, target)
                copied += 1
    return copied
