"""Environment setup for ComfyUI prestartup."""

import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .cache import MARKER_FILE, sanitize_name
from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"
ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"


def is_comfy_env_enabled() -> bool:
    return os.environ.get(USE_COMFY_ENV_VAR, "1").lower() not in ("0", "false", "no", "off")


def load_env_vars(config_path: str) -> Dict[str, str]:
    """Load [env_vars] from comfy-env.toml."""
    if not os.path.exists(config_path): return {}
    try:
        import tomli
        with open(config_path, "rb") as f:
            return {str(k): str(v) for k, v in tomli.load(f).get("env_vars", {}).items()}
    except Exception:
        return {}


def inject_site_packages(env_path: str) -> Optional[str]:
    """Add site-packages to sys.path."""
    if sys.platform == "win32":
        site_packages = os.path.join(env_path, "Lib", "site-packages")
    else:
        matches = glob.glob(os.path.join(env_path, "lib", "python*", "site-packages"))
        site_packages = matches[0] if matches else None

    if site_packages and os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        return site_packages
    return None


def setup_env(node_dir: Optional[str] = None) -> None:
    """Set up env for pixi libraries. Call in prestartup_script.py before native imports."""
    if not is_comfy_env_enabled(): return
    dedupe_libomp()

    if node_dir is None:
        import inspect
        node_dir = str(Path(inspect.stack()[1].filename).parent)

    # Apply env vars (check root config first, then regular)
    root_config = os.path.join(node_dir, ROOT_CONFIG_FILE_NAME)
    config = root_config if os.path.exists(root_config) else os.path.join(node_dir, "comfy-env.toml")
    for k, v in load_env_vars(config).items():
        os.environ[k] = v

    # Find env: marker -> _env_<name> -> .pixi
    pixi_env = None
    marker_path = os.path.join(node_dir, MARKER_FILE)
    if os.path.exists(marker_path):
        try:
            import tomli
            with open(marker_path, "rb") as f:
                env_path = tomli.load(f).get("env", {}).get("path")
            if env_path and os.path.exists(env_path):
                pixi_env = env_path
        except Exception: pass

    if not pixi_env:
        local_env = os.path.join(node_dir, f"_env_{sanitize_name(os.path.basename(node_dir))}")
        if os.path.exists(local_env): pixi_env = local_env

    if not pixi_env:
        old_pixi = os.path.join(node_dir, ".pixi", "envs", "default")
        if os.path.exists(old_pixi): pixi_env = old_pixi

    if not pixi_env: return

    # Set library paths
    if sys.platform == "win32":
        lib_dir = os.path.join(pixi_env, "Library", "bin")
        if os.path.exists(lib_dir): os.environ["PATH"] = lib_dir + ";" + os.environ.get("PATH", "")
    else:
        lib_dir = os.path.join(pixi_env, "lib")
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        if os.path.exists(lib_dir): os.environ[var] = lib_dir + ":" + os.environ.get(var, "")

    inject_site_packages(pixi_env)
