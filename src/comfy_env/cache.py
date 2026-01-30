"""
Central environment cache management for comfy-env.

Stores environments in ~/.comfy-env/envs/<nodename>_<subfolder>_<hash>/
to avoid Windows MAX_PATH (260 char) issues.

Marker files in node folders link to central envs and enable orphan cleanup.
"""

import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Callable

# Import version
try:
    from . import __version__
except ImportError:
    __version__ = "0.0.0-dev"

# Lazy import tomli/tomllib
def _get_tomli():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    else:
        try:
            import tomli
            return tomli
        except ImportError:
            return None

def _get_tomli_w():
    try:
        import tomli_w
        return tomli_w
    except ImportError:
        return None


# Constants
CACHE_DIR = Path.home() / ".comfy-env" / "envs"
MARKER_FILE = ".comfy-env-marker.toml"
METADATA_FILE = ".comfy-env-metadata.toml"


def get_cache_dir() -> Path:
    """Get central cache directory, create if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def compute_config_hash(config_path: Path) -> str:
    """Compute hash of comfy-env.toml content (first 8 chars of SHA256)."""
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:8]


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in filesystem paths."""
    # Lowercase and replace problematic chars
    name = name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("-", "_").replace(" ", "_")


def get_env_name(node_dir: Path, config_path: Path) -> str:
    """
    Generate env name: <nodename>_<subfolder>_<hash>

    Args:
        node_dir: The custom node directory (e.g., custom_nodes/ComfyUI-UniRig)
        config_path: Path to comfy-env.toml

    Examples:
        - ComfyUI-UniRig/nodes/comfy-env.toml -> unirig_nodes_a1b2c3d4
        - ComfyUI-Pack/comfy-env.toml -> pack__f5e6d7c8 (double underscore = no subfolder)
    """
    # Get node name
    node_name = sanitize_name(node_dir.name)

    # Get subfolder (relative path from node_dir to config parent)
    config_parent = config_path.parent
    if config_parent == node_dir:
        subfolder = ""
    else:
        try:
            rel_path = config_parent.relative_to(node_dir)
            subfolder = rel_path.as_posix().replace("/", "_")
        except ValueError:
            # config_path not under node_dir - use parent folder name
            subfolder = sanitize_name(config_parent.name)

    # Compute hash
    config_hash = compute_config_hash(config_path)

    return f"{node_name}_{subfolder}_{config_hash}"


def get_central_env_path(node_dir: Path, config_path: Path) -> Path:
    """Get path to central environment for this config."""
    env_name = get_env_name(node_dir, config_path)
    return get_cache_dir() / env_name


def write_marker(config_path: Path, env_path: Path) -> None:
    """
    Write marker file linking node to central env.

    Args:
        config_path: Path to comfy-env.toml
        env_path: Path to central environment
    """
    tomli_w = _get_tomli_w()
    if not tomli_w:
        # Fallback to manual TOML writing
        marker_path = config_path.parent / MARKER_FILE
        content = f'''[env]
name = "{env_path.name}"
path = "{env_path}"
config_hash = "{compute_config_hash(config_path)}"
created = "{datetime.now().isoformat()}"
comfy_env_version = "{__version__}"
'''
        marker_path.write_text(content)
        return

    marker_path = config_path.parent / MARKER_FILE
    marker_data = {
        "env": {
            "name": env_path.name,
            "path": str(env_path),
            "config_hash": compute_config_hash(config_path),
            "created": datetime.now().isoformat(),
            "comfy_env_version": __version__,
        }
    }
    marker_path.write_text(tomli_w.dumps(marker_data))


def write_env_metadata(env_path: Path, marker_path: Path) -> None:
    """
    Write metadata file in env for orphan detection.

    Args:
        env_path: Path to central environment
        marker_path: Path to marker file in node folder
    """
    tomli_w = _get_tomli_w()
    metadata_path = env_path / METADATA_FILE

    if not tomli_w:
        # Fallback to manual TOML writing
        content = f'''marker_path = "{marker_path}"
created = "{datetime.now().isoformat()}"
'''
        metadata_path.write_text(content)
        return

    metadata = {
        "marker_path": str(marker_path),
        "created": datetime.now().isoformat(),
    }
    metadata_path.write_text(tomli_w.dumps(metadata))


def read_marker(marker_path: Path) -> Optional[dict]:
    """
    Read marker file, return None if invalid/missing.

    Args:
        marker_path: Path to .comfy-env-marker.toml

    Returns:
        Parsed marker data or None
    """
    if not marker_path.exists():
        return None

    tomli = _get_tomli()
    if not tomli:
        return None

    try:
        with open(marker_path, "rb") as f:
            return tomli.load(f)
    except Exception:
        return None


def read_env_metadata(env_path: Path) -> Optional[dict]:
    """
    Read metadata file from env, return None if invalid/missing.

    Args:
        env_path: Path to central environment

    Returns:
        Parsed metadata or None
    """
    metadata_path = env_path / METADATA_FILE
    if not metadata_path.exists():
        return None

    tomli = _get_tomli()
    if not tomli:
        return None

    try:
        with open(metadata_path, "rb") as f:
            return tomli.load(f)
    except Exception:
        return None


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Resolve environment path with fallback chain.

    Args:
        node_dir: Directory containing comfy-env.toml or marker file

    Returns:
        (env_path, site_packages, lib_dir) or (None, None, None)

    Fallback order:
        1. Marker file -> central cache
        2. _env_<name> (current location)
        3. .pixi/envs/default (old pixi)
        4. .venv (venv support)
    """
    # 1. Check marker file -> central cache
    marker_path = node_dir / MARKER_FILE
    marker = read_marker(marker_path)
    if marker and "env" in marker:
        env_path = Path(marker["env"]["path"])
        if env_path.exists():
            return _get_env_paths(env_path)

    # 2. Check _env_<name>
    node_name = sanitize_name(node_dir.name)
    env_name = f"_env_{node_name}"
    local_env = node_dir / env_name
    if local_env.exists():
        return _get_env_paths(local_env)

    # 3. Check .pixi/envs/default
    pixi_env = node_dir / ".pixi" / "envs" / "default"
    if pixi_env.exists():
        return _get_env_paths(pixi_env)

    # 4. Check .venv
    venv_dir = node_dir / ".venv"
    if venv_dir.exists():
        return _get_env_paths(venv_dir)

    return None, None, None


def _get_env_paths(env_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """
    Get site-packages and lib paths from an environment.

    Args:
        env_path: Path to environment root

    Returns:
        (env_path, site_packages, lib_dir)
    """
    import glob

    if sys.platform == "win32":
        site_packages = env_path / "Lib" / "site-packages"
        lib_dir = env_path / "Library" / "bin"
    else:
        # Linux/Mac: lib/python*/site-packages
        matches = glob.glob(str(env_path / "lib" / "python*" / "site-packages"))
        site_packages = Path(matches[0]) if matches else None
        lib_dir = env_path / "lib"

    return env_path, site_packages, lib_dir


def cleanup_orphaned_envs(log: Callable[[str], None] = print) -> int:
    """
    Scan central cache and remove orphaned environments.

    An env is orphaned if its marker file no longer exists
    (meaning the node was deleted).

    Args:
        log: Logging function

    Returns:
        Number of envs cleaned up
    """
    cache_dir = get_cache_dir()
    if not cache_dir.exists():
        return 0

    cleaned = 0
    for env_dir in cache_dir.iterdir():
        if not env_dir.is_dir():
            continue

        # Skip if no metadata (might be manually created or old format)
        metadata = read_env_metadata(env_dir)
        if not metadata:
            continue

        # Check if marker file still exists
        marker_path_str = metadata.get("marker_path", "")
        if not marker_path_str:
            continue

        marker_path = Path(marker_path_str)
        if not marker_path.exists():
            # Marker gone = node was deleted = orphan
            log(f"[comfy-env] Cleaning orphaned env: {env_dir.name}")
            try:
                shutil.rmtree(env_dir)
                cleaned += 1
            except Exception as e:
                log(f"[comfy-env] Failed to cleanup {env_dir.name}: {e}")

    return cleaned
