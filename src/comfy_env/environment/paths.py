"""Environment path utilities."""

import shutil
from pathlib import Path
from typing import Optional, Tuple

from .cache import resolve_env_path as _resolve_env_path


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
