"""
Environment path resolution utilities.

Helpers for finding site-packages, lib directories, and copying files.
"""

import glob
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

from .cache import resolve_env_path as _resolve_env_path


def get_site_packages_path(node_dir: Path) -> Optional[Path]:
    """
    Get site-packages path for a node's environment.

    Args:
        node_dir: Node directory.

    Returns:
        Path to site-packages, or None if not found.
    """
    _, site_packages, _ = _resolve_env_path(node_dir)
    return site_packages


def get_lib_path(node_dir: Path) -> Optional[Path]:
    """
    Get lib directory path for a node's environment.

    Args:
        node_dir: Node directory.

    Returns:
        Path to lib directory, or None if not found.
    """
    _, _, lib_dir = _resolve_env_path(node_dir)
    return lib_dir


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Resolve environment path with fallback chain.

    Re-exported from cache module for convenience.

    Args:
        node_dir: Node directory to check.

    Returns:
        Tuple of (env_path, site_packages, lib_dir). All None if not found.
    """
    return _resolve_env_path(node_dir)


def copy_files(src, dst, pattern: str = "*", overwrite: bool = False) -> int:
    """
    Copy files matching pattern from src to dst.

    Args:
        src: Source directory.
        dst: Destination directory.
        pattern: Glob pattern to match files (default "*").
        overwrite: Whether to overwrite existing files (default False).

    Returns:
        Number of files copied.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return 0

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
