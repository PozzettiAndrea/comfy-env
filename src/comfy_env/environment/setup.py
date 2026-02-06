"""Environment setup for ComfyUI prestartup."""

import glob
import os
import sys
from pathlib import Path
from typing import Optional

from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"
ROOT_CONFIG_FILE_NAME = "comfy-env-root.toml"
ROOT_ENV_DIR_NAME = "_root_env"


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
    """Recursively find all _env_* directories under node_dir (for debug info only)."""
    envs = []
    for root, dirs, _ in os.walk(node_dir):
        for d in dirs:
            if d.startswith("_env_"):
                envs.append(os.path.join(root, d))
        dirs[:] = [d for d in dirs if not d.startswith(("_env_", "_root_env"))]
    return envs


def setup_env(node_dir: Optional[str] = None) -> None:
    """Set up env for pixi libraries. Call in prestartup_script.py before native imports."""
    if node_dir is None:
        import inspect
        node_dir = str(Path(inspect.stack()[1].filename).parent)

    import faulthandler
    faulthandler.enable(file=sys.stderr)

    # Print all detected envs first
    root_env = os.path.join(node_dir, ROOT_ENV_DIR_NAME)
    has_root = os.path.isdir(root_env)
    sub_envs = _find_env_dirs(node_dir)
    if has_root:
        print(f"[comfy-env] {os.path.basename(node_dir)}: _root_env -> {root_env}", file=sys.stderr)
    else:
        print(f"[comfy-env] {os.path.basename(node_dir)}: no _root_env", file=sys.stderr)
    if sub_envs:
        print(f"[comfy-env] {len(sub_envs)} isolation env(s):", file=sys.stderr)
        for env_path in sub_envs:
            print(f"[comfy-env]   {os.path.basename(env_path)} -> {env_path}", file=sys.stderr)

    if not is_comfy_env_enabled(): return
    dedupe_libomp()

    # Apply env vars (check root config first, then regular)
    root_config = os.path.join(node_dir, ROOT_CONFIG_FILE_NAME)
    config = root_config if os.path.exists(root_config) else os.path.join(node_dir, "comfy-env.toml")
    for k, v in load_env_vars(config).items():
        os.environ[k] = v

    # Handle _root_env only -- inject site-packages + set library paths
    if has_root:
        sp = inject_site_packages(root_env)
        _set_library_paths(root_env)
        if sp:
            print(f"[comfy-env]   site-packages: {sp}", file=sys.stderr)

    if os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes"):
        print("[comfy-env] Environment variables:", file=sys.stderr)
        for k, v in sorted(os.environ.items()):
            print(f"[comfy-env]   {k}={v}", file=sys.stderr)
