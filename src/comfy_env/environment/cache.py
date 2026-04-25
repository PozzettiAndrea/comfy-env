"""Environment cache and path utilities for comfy-env.

In the workspace model, comfy-env runs **one pixi workspace per ComfyUI install**
at `<comfyui_dir>/.ce/`, with one environment per custom-node config. Env paths
resolve to `<comfyui_dir>/.ce/.pixi/envs/<env_name>/` directly — no per-config
hash directories, no `_env_*` symlinks.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


CE_WORKSPACE_DIR = ".ce"
PIXI_ENVS_SUBPATH = Path(".pixi") / "envs"


def _get_default_cache_dir() -> Path:
    """Legacy cache dir (kept for backwards compat with code that reads CACHE_DIR)."""
    if sys.platform == "win32":
        return Path("C:/comfy-envs")
    return Path.home() / ".comfy-envs"


CACHE_DIR = _get_default_cache_dir()


def get_cache_dir() -> Path:
    """Legacy: return the (rarely used) external cache dir from COMFY_ENV_CACHE_DIR.

    The pixi package cache lives at the platform-default rattler cache; this is
    only here for code that still reads COMFY_ENV_CACHE_DIR.
    """
    cache_dir = Path(os.environ.get("COMFY_ENV_CACHE_DIR", _get_default_cache_dir()))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def sanitize_name(name: str) -> str:
    """Lowercase + drop comfyui prefix + replace separators."""
    name = name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("-", "").replace("_", "").replace(" ", "")


def get_env_name(plugin_dir: Path, config_path: Path) -> str:
    """Compute the env name for a node's pixi environment.

    Format: `<plugin>` for root-level configs, `<plugin>-<subdir>` otherwise.
    Plugin name has the ComfyUI prefix stripped and is lowercased; underscores
    and dashes are dropped to keep names short (path-length matters on Windows).

    Examples:
        ComfyUI-SAM3 + ComfyUI-SAM3/nodes/comfy-env.toml      -> "sam3-nodes"
        ComfyUI-GeometryPack + nodes/main/comfy-env.toml      -> "geometrypack-main"
        ComfyUI-Foo + comfy-env.toml (root)                   -> "foo"
    """
    plugin = sanitize_name(plugin_dir.name)
    config_parent = config_path.parent.resolve()
    plugin_resolved = plugin_dir.resolve()
    if config_parent == plugin_resolved:
        return plugin
    try:
        rel = config_parent.relative_to(plugin_resolved)
    except ValueError:
        return f"{plugin}-{sanitize_name(config_parent.name)}"
    # rel may be "nodes/main" -> "main"; or just "nodes" -> "nodes"
    parts = list(rel.parts)
    suffix = parts[-1] if parts else ""
    return f"{plugin}-{sanitize_name(suffix)}" if suffix else plugin


def get_workspace_dir(comfyui_dir: Path) -> Path:
    """Return the comfy-env pixi workspace dir for this ComfyUI install."""
    return Path(comfyui_dir) / CE_WORKSPACE_DIR


def get_workspace_env_dir(comfyui_dir: Path, env_name: str) -> Path:
    """Path to one environment inside the workspace."""
    return get_workspace_dir(comfyui_dir) / PIXI_ENVS_SUBPATH / env_name


def find_comfyui_dir_from_node(node_dir: Path) -> Optional[Path]:
    """Walk up from a node dir to find the ComfyUI base (has main.py + comfy/)."""
    current = Path(node_dir).resolve()
    for _ in range(10):
        if (current / "main.py").exists() and (current / "comfy").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def get_local_env_path(plugin_dir: Path, config_path: Path) -> Optional[Path]:
    """Return the env directory for a given node config.

    Resolves to `<comfyui_dir>/.ce/.pixi/envs/<env_name>`. Returns None if the
    ComfyUI base can't be located.
    """
    comfyui_dir = find_comfyui_dir_from_node(plugin_dir)
    if comfyui_dir is None:
        return None
    return get_workspace_env_dir(comfyui_dir, get_env_name(plugin_dir, config_path))


def _get_env_paths(env_path: Path) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Return (env_path, site_packages, lib_dir) for a pixi env directory."""
    if sys.platform == "win32":
        return env_path, env_path / "Lib" / "site-packages", env_path / "Library" / "bin"
    matches = glob.glob(str(env_path / "lib" / "python*" / "site-packages"))
    return env_path, Path(matches[0]) if matches else None, env_path / "lib"


def resolve_env_path(node_dir: Path) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find the pixi env for a node and return (env_path, site_packages, lib_dir).

    `node_dir` is the directory containing a `comfy-env.toml`. We walk up to the
    plugin root (parent of the ComfyUI custom_nodes dir's child), then map to the
    workspace env via `get_local_env_path`.
    """
    node_dir = Path(node_dir).resolve()

    # Find the plugin root: walk up until parent is `custom_nodes/`.
    plugin_dir = node_dir
    for parent in node_dir.parents:
        if parent.parent and parent.parent.name == "custom_nodes":
            plugin_dir = parent
            break

    # Locate the config file inside node_dir
    config_path = None
    for cand in ("comfy-env.toml", "comfy-env-root.toml"):
        if (node_dir / cand).exists():
            config_path = node_dir / cand
            break
    if config_path is None:
        return None, None, None

    env_path = get_local_env_path(plugin_dir, config_path)
    if env_path is None or not env_path.exists():
        return None, None, None
    return _get_env_paths(env_path)
