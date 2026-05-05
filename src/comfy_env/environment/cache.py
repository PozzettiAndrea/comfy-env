"""Environment path resolution for comfy-env."""

import shutil
from pathlib import Path

CE_WORKSPACE_DIR = ".ce"


def get_env_name(plugin_dir, config_path):
    """Compute the pixi env name for a node's isolated environment.

    Format: `<plugin>` for root-level configs, `<plugin>-<subdir>` otherwise.
    Strips ComfyUI prefix, lowercases, drops separators.
    """
    plugin_dir, config_path = Path(plugin_dir), Path(config_path)

    # Sanitize plugin name
    name = plugin_dir.name.lower()
    for prefix in ("comfyui-", "comfyui_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    name = name.replace("-", "").replace("_", "").replace(" ", "")

    # If config is at plugin root, just use the plugin name
    config_parent = config_path.parent.resolve()
    plugin_resolved = plugin_dir.resolve()
    if config_parent == plugin_resolved:
        return name

    # Otherwise append the subdirectory name
    try:
        rel = config_parent.relative_to(plugin_resolved)
        suffix = rel.parts[-1] if rel.parts else ""
    except ValueError:
        suffix = config_parent.name
    suffix = suffix.lower().replace("-", "").replace("_", "")
    return f"{name}-{suffix}" if suffix else name


def get_workspace_dir(comfyui_dir):
    """Return the comfy-env pixi workspace dir: <comfyui>/.ce/"""
    return Path(comfyui_dir) / CE_WORKSPACE_DIR


def get_workspace_env_dir(comfyui_dir, env_name):
    """Path to one environment inside the workspace."""
    return get_workspace_dir(comfyui_dir) / ".pixi" / "envs" / env_name


def find_comfyui_dir_from_node(node_dir):
    """Walk up from a node dir to find the ComfyUI base (has main.py + comfy/)."""
    current = Path(node_dir).resolve()
    for _ in range(10):
        if (current / "main.py").exists() and (current / "comfy").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return None


def copy_files(src, dst, pattern="*", overwrite=False):
    """Copy files matching pattern from src to dst."""
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
