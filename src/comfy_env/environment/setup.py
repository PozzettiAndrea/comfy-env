"""
Environment setup helpers for ComfyUI prestartup.

Call setup_env() in your prestartup_script.py before any native imports.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .cache import MARKER_FILE, sanitize_name
from .libomp import dedupe_libomp


USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


def is_comfy_env_enabled() -> bool:
    """Check if comfy-env isolation is enabled (default: True)."""
    val = os.environ.get(USE_COMFY_ENV_VAR, "1").lower()
    return val not in ("0", "false", "no", "off")


def load_env_vars(config_path: str) -> Dict[str, str]:
    """
    Load [env_vars] section from comfy-env.toml.

    Args:
        config_path: Path to comfy-env.toml.

    Returns:
        Dict of environment variables, empty if file not found.
    """
    if not os.path.exists(config_path):
        return {}

    try:
        import tomli

        with open(config_path, "rb") as f:
            data = tomli.load(f)

        env_vars_data = data.get("env_vars", {})
        return {str(k): str(v) for k, v in env_vars_data.items()}
    except Exception:
        return {}


def inject_site_packages(env_path: str) -> Optional[str]:
    """
    Add site-packages from environment to sys.path.

    Args:
        env_path: Path to environment.

    Returns:
        Path to site-packages if added, None otherwise.
    """
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
    """
    Set up environment for pixi conda libraries.

    Call this in prestartup_script.py before any native library imports.

    This function:
    1. Checks if comfy-env is enabled (USE_COMFY_ENV env var)
    2. Dedupes libomp on macOS to prevent OpenMP conflicts
    3. Applies [env_vars] from comfy-env.toml
    4. Sets library paths (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, PATH)
    5. Adds site-packages to sys.path

    Args:
        node_dir: Path to the custom node directory. Auto-detected if not provided.

    Example:
        # In prestartup_script.py:
        from comfy_env import setup_env
        setup_env()
    """
    # Skip if isolation is disabled
    if not is_comfy_env_enabled():
        return

    # macOS: Dedupe libomp to prevent OpenMP conflicts
    dedupe_libomp()

    # Auto-detect node_dir from caller
    if node_dir is None:
        import inspect
        frame = inspect.stack()[1]
        node_dir = str(Path(frame.filename).parent)

    # Apply [env_vars] from comfy-env.toml FIRST (before any library loading)
    config_path = os.path.join(node_dir, "comfy-env.toml")
    env_vars = load_env_vars(config_path)
    for key, value in env_vars.items():
        os.environ[key] = value

    # Resolve environment path with fallback chain:
    # 1. Marker file -> central cache
    # 2. _env_<name> (current local)
    # 3. .pixi/envs/default (old pixi)
    pixi_env = None

    # 1. Check marker file -> central cache
    marker_path = os.path.join(node_dir, MARKER_FILE)
    if os.path.exists(marker_path):
        try:
            import tomli
            with open(marker_path, "rb") as f:
                marker = tomli.load(f)
            env_path = marker.get("env", {}).get("path")
            if env_path and os.path.exists(env_path):
                pixi_env = env_path
        except Exception:
            pass  # Fall through to other options

    # 2. Check _env_<name> (local)
    if not pixi_env:
        env_name = f"_env_{sanitize_name(os.path.basename(node_dir))}"
        local_env = os.path.join(node_dir, env_name)
        if os.path.exists(local_env):
            pixi_env = local_env

    # 3. Fallback to old .pixi path
    if not pixi_env:
        old_pixi = os.path.join(node_dir, ".pixi", "envs", "default")
        if os.path.exists(old_pixi):
            pixi_env = old_pixi

    if not pixi_env:
        return  # No environment found

    # Set library paths for native libraries
    if sys.platform == "win32":
        # Windows: add to PATH for DLL loading
        lib_dir = os.path.join(pixi_env, "Library", "bin")
        if os.path.exists(lib_dir):
            os.environ["PATH"] = lib_dir + ";" + os.environ.get("PATH", "")
    elif sys.platform == "darwin":
        # macOS: DYLD_LIBRARY_PATH
        lib_dir = os.path.join(pixi_env, "lib")
        if os.path.exists(lib_dir):
            os.environ["DYLD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
    else:
        # Linux: LD_LIBRARY_PATH
        lib_dir = os.path.join(pixi_env, "lib")
        if os.path.exists(lib_dir):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    # Add site-packages to sys.path
    inject_site_packages(pixi_env)
