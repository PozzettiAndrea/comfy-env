from __future__ import annotations

import os
import subprocess
import sys


def has_nvidia_gpu() -> bool:
    """Check if NVIDIA GPU is present."""
    return detect_cuda_version() is not None


def detect_cuda_version() -> str | None:
    """Detect system CUDA version. Priority: pixi -> torch metadata."""
    if pixi_cuda := _get_cuda_from_pixi():
        return pixi_cuda
    return get_bootstrap_torch_cuda()


def _get_cuda_from_pixi() -> str | None:
    """Get CUDA version from pixi's virtual package detection."""
    try:
        from ..packages.pixi import PIXI
        import json
        result = subprocess.run([PIXI, "info", "--json"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        info = json.loads(result.stdout)
        for vp in info.get("virtual_packages", []):
            if vp.startswith("__cuda="):
                return vp.split("=")[1]
    except Exception:
        pass
    return None


def get_bootstrap_python_version() -> str:
    """Python version of the interpreter running comfy-env (e.g. '3.10')."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_bootstrap_torch_version() -> str | None:
    """Torch version from package metadata (e.g. '2.11.0'), without importing torch."""
    try:
        from importlib.metadata import version
        v = version("torch")
        return str(v).split("+", 1)[0]
    except Exception:
        return None


def get_bootstrap_torch_cuda() -> str | None:
    """CUDA version the host torch was built against (e.g. '12.8').

    Parsed from the torch package version's local label (e.g. '2.5.0+cu128' -> '12.8').
    """
    try:
        from importlib.metadata import version
        v = version("torch")
        if "+" not in v:
            return None
        local = v.split("+", 1)[1]
        if not local.startswith("cu"):
            return None
        cu_digits = local[2:]  # e.g. "128"
        if len(cu_digits) >= 2:
            return f"{cu_digits[:-1]}.{cu_digits[-1]}"
        return None
    except Exception:
        return None
