"""Environment path resolution for comfy-env.

Layout (per-env manifests, v0.4+):

    <workspace_root>/
      envs/
        <env_name>/
          pixi.toml                  # one env per file
          pixi.lock
          .pixi/envs/default/        # materialized env

Each env's manifest is isolated -- a parse error in one cannot poison
another. `<workspace_root>` is shared machine-wide so two ComfyUI installs
that declare the same node reuse the same materialized env dir (cross-install
sharing).

No backward compatibility with the v0.3.x single-file layout. Workspaces
created by v0.3.x (``<workspace>/pixi.toml`` + ``<workspace>/.pixi/envs/<name>/``)
are invisible to v0.4+; they need to be re-materialized via
``comfy-env install`` or auto-install at startup
(``COMFY_ENV_AUTO_INSTALL=1``). User is expected to ``rm -rf`` the legacy
``<workspace>/.pixi/`` and ``<workspace>/pixi.toml`` once they've moved over.
"""

import os
import re
import shutil
import sys
from pathlib import Path

# Legacy workspace dir name (kept as a constant for the one orphan-detect
# log line in `install_workspace`).
CE_WORKSPACE_DIR = ".ce"


_ANNOUNCED_WS = False


def _sanitize_pixi_name(s: str) -> str:
    """Pixi environment names must match `[a-z0-9-]+`. Collapse any other
    char run to a single dash and trim, so a folder rename like `Foo._disabled`
    or a name containing dots / parens / spaces cannot poison a manifest.
    """
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower())
    return re.sub(r"-+", "-", s).strip("-")


def _short_global_root():
    """Resolve workspace root. Defaults to %LOCALAPPDATA%\\Programs\\comfy-env
    on Windows (sits next to the ComfyUI Desktop install at
    %LOCALAPPDATA%\\Programs\\ComfyUI) so fresh installs never need admin --
    the old default `C:\\ce` required admin to create at drive root.
    Override via COMFY_ENV_ROOT.
    """
    global _ANNOUNCED_WS

    override = os.environ.get("COMFY_ENV_ROOT")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        root = base / "Programs" / "comfy-env"
    else:
        root = Path.home() / ".ce"

    root.mkdir(parents=True, exist_ok=True)

    if not _ANNOUNCED_WS:
        print(f"[comfy-env] Workspace: {root}", file=sys.stderr, flush=True)
        legacy = Path(r"C:\ce\.pixi\envs")
        if sys.platform == "win32" and root != Path(r"C:\ce") and legacy.is_dir() and any(legacy.iterdir()):
            print(
                "[comfy-env] Legacy workspace detected at C:\\ce. "
                "Workspace has moved to %LOCALAPPDATA%\\Programs\\comfy-env -- "
                "please reinstall the node ('python install.py') and then "
                "delete C:\\ce.",
                file=sys.stderr, flush=True,
            )
        _ANNOUNCED_WS = True

    return root


def get_env_name(plugin_dir, config_path):
    """Compute the pixi env name for a node's isolated environment.

    Format: ``<plugin>`` for root-level configs, ``<plugin>-<subdir>``
    otherwise. Strips ``ComfyUI[-_]`` prefix, lowercases, and collapses any
    char outside ``[a-z0-9-]`` to a single dash (so a folder like
    ``Foo._disabled`` produces ``foo-disabled``, not ``foo.disabled`` which
    pixi rejects).
    """
    plugin_dir, config_path = Path(plugin_dir), Path(config_path)

    plugin_part = plugin_dir.name
    for prefix in ("ComfyUI-", "ComfyUI_", "comfyui-", "comfyui_"):
        if plugin_part.startswith(prefix):
            plugin_part = plugin_part[len(prefix):]
            break
    name = _sanitize_pixi_name(plugin_part)

    config_parent = config_path.parent.resolve()
    plugin_resolved = plugin_dir.resolve()
    if config_parent != plugin_resolved:
        try:
            rel = config_parent.relative_to(plugin_resolved)
            suffix_raw = rel.parts[-1] if rel.parts else ""
        except ValueError:
            suffix_raw = config_parent.name
        suffix = _sanitize_pixi_name(suffix_raw)
        if suffix:
            name = f"{name}-{suffix}" if name else suffix

    return name


def get_workspace_dir(comfyui_dir=None):
    """Return the single global comfy-env pixi workspace root.

    Shared across every ComfyUI install on this machine — env names act as
    the global identifier (conda-style). `comfyui_dir` is accepted for
    signature compatibility but ignored.
    """
    return _short_global_root()


def get_env_manifest_dir(env_name: str, comfyui_dir=None) -> Path:
    """Directory containing one env's `pixi.toml` (new per-env layout).

    `<workspace>/envs/<env_name>/`
    """
    return get_workspace_dir(comfyui_dir) / "envs" / env_name


def get_env_manifest_path(env_name: str, comfyui_dir=None) -> Path:
    """Path to one env's `pixi.toml` (new per-env layout).

    `<workspace>/envs/<env_name>/pixi.toml`
    """
    return get_env_manifest_dir(env_name, comfyui_dir) / "pixi.toml"


def resolve_pixi_manifest(env_root: Path) -> tuple[Path, str]:
    """Given a materialized env directory, return ``(manifest_path, env_pixi_name)``.

    Per-env layout only. ``env_root`` is always
    ``<workspace>/envs/<env_name>/.pixi/envs/default`` -- the per-env pixi
    manifest lives at ``<workspace>/envs/<env_name>/pixi.toml`` and the
    pixi environment inside that manifest is always named ``default``.
    """
    env_root = Path(env_root)
    # <ws>/envs/<name>/.pixi/envs/default -> walk up 3 to reach <ws>/envs/<name>/
    manifest_dir = env_root.parent.parent.parent
    manifest = manifest_dir / "pixi.toml"
    return (manifest, "default")


def get_workspace_env_dir(comfyui_dir, env_name):
    """Path to one environment's materialized site-packages root.

    Always ``<workspace>/envs/<env_name>/.pixi/envs/default/`` -- the v0.4+
    per-env layout. No legacy fallback.
    """
    return get_workspace_dir(comfyui_dir) / "envs" / env_name / ".pixi" / "envs" / "default"


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
