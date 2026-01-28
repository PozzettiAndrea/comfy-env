"""
Prestartup helpers for ComfyUI custom nodes.

Call setup_env() in your prestartup_script.py before any native imports.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional


def setup_env(node_dir: Optional[str] = None) -> None:
    """
    Set up environment for pixi conda libraries.

    Call this in prestartup_script.py before any native library imports.
    Sets LD_LIBRARY_PATH (Linux/Mac) or PATH (Windows) for conda libs,
    and adds pixi site-packages to sys.path.

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
