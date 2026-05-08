"""Environment path resolution for comfy-env."""

import os
import shutil
import sys
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


def _find_desktop_source_dir():
    """Find ComfyUI source dir from the Desktop app's extra_models_config.yaml.

    The Electron desktop app stores config in its userData dir:
      macOS:   ~/Library/Application Support/ComfyUI/
      Windows: %APPDATA%/ComfyUI/
      Linux:   ~/.config/ComfyUI/

    The file extra_models_config.yaml contains a desktop_extensions section
    with a custom_nodes path pointing into the app bundle. The parent of
    that path is the ComfyUI source dir (where main.py, comfy/, requirements.txt live).
    """
    if sys.platform == "darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "ComfyUI"
    elif sys.platform == "win32":
        config_dir = Path(os.environ.get("APPDATA", "")) / "ComfyUI"
    else:
        config_dir = Path.home() / ".config" / "ComfyUI"

    yaml_path = config_dir / "extra_models_config.yaml"
    if not yaml_path.exists():
        return None
    try:
        content = yaml_path.read_text(encoding="utf-8")
        # Look for desktop_extensions.custom_nodes line
        in_desktop = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("desktop_extensions"):
                in_desktop = True
                continue
            if in_desktop and stripped.startswith("custom_nodes:"):
                path_str = stripped.split(":", 1)[1].strip()
                candidate = Path(path_str).parent
                if (candidate / "main.py").exists():
                    return candidate
            if not line.startswith(" ") and not line.startswith("\t") and ":" in line:
                in_desktop = False
    except Exception:
        pass
    return None


def find_comfyui_dir_from_node(node_dir=None):
    """Find the ComfyUI source directory.

    Priority:
    1. folder_paths module location (when running inside ComfyUI server)
    2. Walk up from node_dir looking for main.py + comfy/ (standard layout)
    3. Walk up looking for custom_nodes/ + user/ (Desktop app user data dir)
       → then resolve actual source dir from Desktop app config
    4. Desktop app config directly (no node_dir needed)
    """
    # 1. Running inside ComfyUI server
    try:
        import folder_paths
        return Path(folder_paths.__file__).parent
    except ImportError:
        pass

    # 2+3. Walk up from node_dir
    if node_dir is not None:
        current = Path(node_dir).resolve()
        for _ in range(10):
            # Standard layout: has main.py + comfy/
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            # Desktop app user data dir: has custom_nodes/ but no main.py
            if (current / "custom_nodes").is_dir() and (current / "user").is_dir():
                if not (current / "main.py").exists():
                    # This is the user data dir, not the source dir.
                    # Try to find the real source dir from Desktop app config.
                    source = _find_desktop_source_dir()
                    if source:
                        return source
                # If we can't find the source dir, return the user data dir
                # as fallback (better than None)
                return current
            if current.parent == current:
                break
            current = current.parent

    # 4. Last resort: check Desktop app config directly
    return _find_desktop_source_dir()


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
