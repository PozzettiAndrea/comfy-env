"""
Minimal folder_paths stub for isolated worker processes.

Provides the same interface as ComfyUI's folder_paths module
without importing any ComfyUI dependencies.
"""

import os
from pathlib import Path

_comfyui_base = None

def _find_comfyui_base():
    """Find ComfyUI base from COMFYUI_BASE env var or by walking up."""
    global _comfyui_base
    if _comfyui_base:
        return _comfyui_base

    # Check env var first
    if os.environ.get("COMFYUI_BASE"):
        _comfyui_base = Path(os.environ["COMFYUI_BASE"])
        return _comfyui_base

    # Walk up from cwd looking for ComfyUI
    current = Path.cwd().resolve()
    for _ in range(10):
        if (current / "main.py").exists() and (current / "comfy").exists():
            _comfyui_base = current
            return _comfyui_base
        current = current.parent

    return None

# Models directory
@property
def models_dir():
    base = _find_comfyui_base()
    return str(base / "models") if base else None

# Make models_dir work as both attribute and property
class _ModuleProxy:
    @property
    def models_dir(self):
        base = _find_comfyui_base()
        return str(base / "models") if base else None

    def get_output_directory(self):
        base = _find_comfyui_base()
        return str(base / "output") if base else None

    def get_input_directory(self):
        base = _find_comfyui_base()
        return str(base / "input") if base else None

# Replace module with proxy instance
import sys
sys.modules[__name__] = _ModuleProxy()
