"""Environment setup for ComfyUI prestartup."""

import glob
import os
import sys
from pathlib import Path
from typing import Optional

from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"
ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"


def is_comfy_env_enabled() -> bool:
    return os.environ.get(USE_COMFY_ENV_VAR, "1").lower() not in ("0", "false", "no", "off")


def load_env_vars(config_path: str) -> dict:
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


def _set_library_paths(env_path: str) -> None:
    """Add library dirs to PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH."""
    if sys.platform == "win32":
        lib_dir = os.path.join(env_path, "Library", "bin")
        if os.path.exists(lib_dir):
            os.environ["PATH"] = lib_dir + ";" + os.environ.get("PATH", "")
    else:
        lib_dir = os.path.join(env_path, "lib")
        var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        if os.path.exists(lib_dir):
            os.environ[var] = lib_dir + ":" + os.environ.get(var, "")


def _find_env_dirs(node_dir: str) -> list:
    """Recursively find all _env_* directories under node_dir."""
    envs = []
    for root, dirs, _ in os.walk(node_dir):
        for d in dirs:
            if d.startswith("_env_"):
                envs.append(os.path.join(root, d))
        # Don't recurse into _env_* dirs themselves
        dirs[:] = [d for d in dirs if not d.startswith("_env_")]
    return envs


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

    # Find all _env_* dirs
    env_dirs = _find_env_dirs(node_dir)
    print(f"[comfy-env] {os.path.basename(node_dir)}: {len(env_dirs)} env(s) found")
    for env_path in env_dirs:
        print(f"[comfy-env]   {os.path.basename(env_path)} -> {env_path}")

    for env_path in env_dirs:
        sp = inject_site_packages(env_path)
        _set_library_paths(env_path)
