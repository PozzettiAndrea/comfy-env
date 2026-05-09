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


def _candidate_config_dirs():
    """Yield candidate ComfyUI Desktop config dirs, ordered most-specific
    first. On Windows, robust against SYSTEM-context APPDATA inherited from
    agent harnesses, scheduled tasks, or service-spawned shells: when
    APPDATA points at the systemprofile subtree (where ComfyUI never
    writes), we fall through to USERPROFILE-, USERNAME-, then a glob across
    C:\\Users\\* picking any user that has the userData dir."""
    if sys.platform == "darwin":
        return [Path.home() / "Library" / "Application Support" / "ComfyUI"]
    if sys.platform != "win32":
        return [Path.home() / ".config" / "ComfyUI"]

    seen = set()
    out = []
    def add(p):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    appdata = os.environ.get("APPDATA", "")
    if appdata and "systemprofile" not in appdata.lower():
        add(Path(appdata) / "ComfyUI")
    userprofile = os.environ.get("USERPROFILE", "")
    if userprofile and "systemprofile" not in userprofile.lower():
        add(Path(userprofile) / "AppData" / "Roaming" / "ComfyUI")
    username = os.environ.get("USERNAME", "")
    if username and username.upper() != "SYSTEM":
        add(Path("C:/Users") / username / "AppData" / "Roaming" / "ComfyUI")
    try:
        from glob import glob as _glob
        skip = ("default", "default user", "public", "all users")
        for p in _glob(r"C:\Users\*\AppData\Roaming\ComfyUI"):
            parts = Path(p).parts
            user_seg = parts[2].lower() if len(parts) > 2 else ""
            if user_seg in skip:
                continue
            add(Path(p))
    except Exception:
        pass
    return out


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
    for config_dir in _candidate_config_dirs():
        yaml_path = config_dir / "extra_models_config.yaml"
        if not yaml_path.exists():
            continue
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
                if (not line.startswith(" ")
                        and not line.startswith("\t")
                        and ":" in line):
                    in_desktop = False
        except Exception:
            continue
    return None


def find_comfyui_dir_from_node(node_dir=None):
    """Find the ComfyUI user data directory (where .ce/, custom_nodes/, user/ live).

    This is where the workspace and pixi envs are created.
    On standard installs this is the same as the source dir.
    On Desktop app this is ~/Documents/ComfyUI/ (NOT the app bundle).
    """
    # Running inside ComfyUI server — base_path is the user data dir
    try:
        import folder_paths
        return Path(folder_paths.base_path)
    except ImportError:
        pass

    # Walk up from node_dir
    if node_dir is not None:
        current = Path(node_dir).resolve()
        for _ in range(10):
            # Standard: has main.py + comfy/ (source dir IS the data dir)
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            # Desktop app: has custom_nodes/ + user/ but no main.py
            if (current / "custom_nodes").is_dir() and (current / "user").is_dir():
                return current
            if current.parent == current:
                break
            current = current.parent
    return None


def find_comfyui_source_dir(node_dir=None):
    """Find the ComfyUI source directory (where main.py, comfy/, requirements.txt live).

    On standard installs this is the same as the data dir.
    On Desktop app this is inside the app bundle (e.g. ComfyUI.app/.../ComfyUI/).
    """
    # Running inside ComfyUI server — folder_paths module is in the source dir
    try:
        import folder_paths
        return Path(folder_paths.__file__).parent
    except ImportError:
        pass

    # Walk up from node_dir — if we find main.py + comfy/, that's it
    if node_dir is not None:
        current = Path(node_dir).resolve()
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            if current.parent == current:
                break
            current = current.parent

    # Desktop app: source dir is in the app bundle, read from config
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
