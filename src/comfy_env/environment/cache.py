"""Environment cache and path utilities for comfy-env."""

import glob
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


def _get_default_cache_dir() -> Path:
    """Get platform-specific default cache directory."""
    if sys.platform == "win32":
        return Path("C:/comfy-envs")
    else:
        return Path.home() / ".comfy-envs"


CACHE_DIR = _get_default_cache_dir()


def get_cache_dir() -> Path:
    """Get cache dir, checking COMFY_ENV_CACHE_DIR env var each time."""
    cache_dir = Path(os.environ.get("COMFY_ENV_CACHE_DIR", _get_default_cache_dir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:8]


def sanitize_name(name: str) -> str:
    name = name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix): name = name[len(prefix):]
    return name.replace("-", "_").replace(" ", "_")


def get_env_name(node_dir: Path, config_path: Path) -> str:
    """Generate env name: <nodename>_<subfolder>_<hash>"""
    node_name = sanitize_name(node_dir.name)
    config_parent = config_path.parent
    if config_parent == node_dir:
        subfolder = ""
    else:
        try:
            subfolder = config_parent.relative_to(node_dir).as_posix().replace("/", "_")
        except ValueError:
            subfolder = sanitize_name(config_parent.name)
    return f"{node_name}_{subfolder}_{compute_config_hash(config_path)}"


ROOT_ENV_DIR_NAME = "_root_env"


def get_root_env_path(node_dir: Path) -> Path:
    """Return path for root env: node_dir/_root_env."""
    return node_dir / ROOT_ENV_DIR_NAME


def get_local_env_path(main_node_dir: Path, config_path: Path) -> Path:
    """Return path for _env_* folder directly in config_path.parent (subdirectory envs only)."""
    plugin = sanitize_name(main_node_dir.name)
    subfolder = sanitize_name(config_path.parent.name)
    h = compute_config_hash(config_path)[:6]
    if subfolder == plugin or config_path.parent == main_node_dir:
        name = f"_env_{plugin}_{h}"
    else:
        name = f"_env_{plugin}_{subfolder}_{h}"
    return config_path.parent / name


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find _env_* dir in node_dir."""
    try:
        for item in node_dir.iterdir():
            if item.name.startswith("_env_") and item.is_dir():
                return _get_env_paths(item)
    except OSError:
        pass
    return None, None, None


def _get_env_paths(env_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    if sys.platform == "win32":
        return env_path, env_path / "Lib" / "site-packages", env_path / "Library" / "bin"
    matches = glob.glob(str(env_path / "lib" / "python*" / "site-packages"))
    return env_path, Path(matches[0]) if matches else None, env_path / "lib"
