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


def get_bootstrap_torch_macos_min() -> str | None:
    """Min macOS version the installed torch wheel targets, e.g. '14.0'.

    Reads the wheel's `WHEEL` metadata (`Tag: cp313-cp313-macosx_14_0_arm64`)
    so we track torch's actual wheel matrix — no hardcoded version→macOS
    table. Returns None if torch isn't installed or wasn't installed from
    a macOS wheel (e.g. sdist, or a Linux/Windows wheel on this box).

    Motivation: pixi defaults osx-arm64 to macOS 13, but torch 2.12+ only
    ships macosx_14_0_arm64 wheels — pixi can't resolve. Emitting
    `[system-requirements] macos = "14.0"` in the generated pixi.toml
    lets pixi see the newer wheel.
    """
    try:
        from importlib.metadata import distribution, PackageNotFoundError
        import re
        try:
            wheel_text = distribution("torch").read_text("WHEEL")
        except PackageNotFoundError:
            return None
        if not wheel_text:
            return None
        for line in wheel_text.splitlines():
            if not line.startswith("Tag:"):
                continue
            m = re.search(r"macosx_(\d+)_(\d+)_", line)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
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
