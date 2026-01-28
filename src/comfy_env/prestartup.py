"""
Prestartup helpers for ComfyUI custom nodes.

Call setup_env() in your prestartup_script.py before any native imports.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional, Dict


def _load_env_vars(config_path: str) -> Dict[str, str]:
    """
    Load [env_vars] section from comfy-env.toml.

    Uses tomllib (Python 3.11+) or tomli fallback.
    Returns empty dict if file not found or parsing fails.
    """
    if not os.path.exists(config_path):
        return {}

    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            try:
                import tomli as tomllib
            except ImportError:
                return {}

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        env_vars_data = data.get("env_vars", {})
        return {str(k): str(v) for k, v in env_vars_data.items()}
    except Exception:
        return {}


def setup_env(node_dir: Optional[str] = None) -> None:
    """
    Set up environment for pixi conda libraries.

    Call this in prestartup_script.py before any native library imports.
    - Applies [env_vars] from comfy-env.toml first (for OpenMP settings, etc.)
    - Sets LD_LIBRARY_PATH (Linux/Mac) or PATH (Windows) for conda libs
    - Adds pixi site-packages to sys.path

    Args:
        node_dir: Path to the custom node directory. Auto-detected if not provided.

    Example:
        # In prestartup_script.py:
        from comfy_env import setup_env
        setup_env()
    """
    # Auto-detect node_dir from caller
    if node_dir is None:
        import inspect
        frame = inspect.stack()[1]
        node_dir = str(Path(frame.filename).parent)

    # Apply [env_vars] from comfy-env.toml FIRST (before any library loading)
    config_path = os.path.join(node_dir, "comfy-env.toml")
    env_vars = _load_env_vars(config_path)
    for key, value in env_vars.items():
        os.environ[key] = value

    pixi_env = os.path.join(node_dir, ".pixi", "envs", "default")

    if not os.path.exists(pixi_env):
        return  # No pixi environment

    if sys.platform == "win32":
        # Windows: add to PATH for DLL loading
        lib_dir = os.path.join(pixi_env, "Library", "bin")
        if os.path.exists(lib_dir):
            os.environ["PATH"] = lib_dir + ";" + os.environ.get("PATH", "")
    else:
        # Linux/Mac: LD_LIBRARY_PATH
        lib_dir = os.path.join(pixi_env, "lib")
        if os.path.exists(lib_dir):
            os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    # Add site-packages to sys.path for pixi-installed Python packages
    if sys.platform == "win32":
        site_packages = os.path.join(pixi_env, "Lib", "site-packages")
    else:
        matches = glob.glob(os.path.join(pixi_env, "lib", "python*", "site-packages"))
        site_packages = matches[0] if matches else None

    if site_packages and os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
