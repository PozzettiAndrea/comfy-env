"""
Prestartup helpers for ComfyUI custom nodes.

Call setup_env() in your prestartup_script.py before any native imports.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional, Dict


def get_env_name(dir_name: str) -> str:
    """Convert directory name to env name: ComfyUI-UniRig -> _env_unirig"""
    name = dir_name.lower().replace("-", "_").lstrip("comfyui_")
    return f"_env_{name}"


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


def _dedupe_libomp_macos():
    """
    macOS: Dedupe libomp.dylib to prevent OpenMP runtime conflicts.

    Torch and other packages (PyMeshLab, etc.) bundle their own libomp.
    Two OpenMP runtimes in same process = crash.
    Fix: symlink all libomp copies to torch's (canonical source).
    """
    if sys.platform != "darwin":
        return

    # Find torch's libomp (canonical source) via introspection
    try:
        import torch
        torch_libomp = os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib')
        if not os.path.exists(torch_libomp):
            return
    except ImportError:
        return  # No torch, skip

    # Find site-packages for scanning
    site_packages = os.path.dirname(os.path.dirname(torch.__file__))

    # Find other libomp files and symlink to torch's
    patterns = [
        os.path.join(site_packages, '*', 'Frameworks', 'libomp.dylib'),
        os.path.join(site_packages, '*', '.dylibs', 'libomp.dylib'),
        os.path.join(site_packages, '*', 'lib', 'libomp.dylib'),
    ]

    for pattern in patterns:
        for libomp in glob.glob(pattern):
            # Skip torch's own copy
            if 'torch' in libomp:
                continue
            # Skip if already a symlink pointing to torch
            if os.path.islink(libomp):
                if os.path.realpath(libomp) == os.path.realpath(torch_libomp):
                    continue
            # Replace with symlink to torch's
            try:
                if os.path.islink(libomp):
                    os.unlink(libomp)
                else:
                    os.rename(libomp, libomp + '.bak')
                os.symlink(torch_libomp, libomp)
            except OSError:
                pass  # Permission denied, etc.


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
    # macOS: Dedupe libomp to prevent OpenMP conflicts (torch vs pymeshlab, etc.)
    _dedupe_libomp_macos()

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

    # Resolve environment path with fallback chain:
    # 1. Marker file -> central cache
    # 2. _env_<name> (current local)
    # 3. .pixi/envs/default (old pixi)
    pixi_env = None

    # 1. Check marker file -> central cache
    marker_path = os.path.join(node_dir, ".comfy-env-marker.toml")
    if os.path.exists(marker_path):
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                import tomli as tomllib
            with open(marker_path, "rb") as f:
                marker = tomllib.load(f)
            env_path = marker.get("env", {}).get("path")
            if env_path and os.path.exists(env_path):
                pixi_env = env_path
        except Exception:
            pass  # Fall through to other options

    # 2. Check _env_<name> (local)
    if not pixi_env:
        env_name = get_env_name(os.path.basename(node_dir))
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

    # Add site-packages to sys.path for pixi-installed Python packages
    if sys.platform == "win32":
        site_packages = os.path.join(pixi_env, "Lib", "site-packages")
    else:
        matches = glob.glob(os.path.join(pixi_env, "lib", "python*", "site-packages"))
        site_packages = matches[0] if matches else None

    if site_packages and os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
