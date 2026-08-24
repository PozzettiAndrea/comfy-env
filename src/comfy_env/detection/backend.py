"""Accelerator backend detection: which compute backend torch will actually use.

Ground truth is the installed torch build's local version label
(e.g. '2.5.0+cu128' -> cuda, '2.4.0+rocm6.2' -> rocm, '2.5.0+xpu' -> xpu). That
label -- not nvidia-smi or the system driver -- is what determines wheel
compatibility, so it is what names the backend here.

Only 'cuda' and 'cpu' are wired end-to-end today. 'rocm'/'xpu'/'mps' are named so
callers can dispatch on the backend and a new backend slots in additively (its
wheel-index entry, toml section, hardware probe) without a rename. This also
fixes the prior misdetection: a ROCm torch has torch.cuda.is_available() == True
but torch.version.cuda is None, so the cuda-only `+cuXXX` parse read it as CPU
while the torch GPU probe reported a bogus NVIDIA arch.

detection/cuda.py + detection/gpu.py remain the CUDA-specific probes -- they are
the 'cuda' branch callers use when detect_backend() returns 'cuda'.
"""

from __future__ import annotations

import sys

from .cuda import get_bootstrap_torch_cuda

# Backends this project knows how to name. Wired end-to-end: cuda, cpu.
KNOWN_BACKENDS = ("cuda", "rocm", "xpu", "mps", "cpu")


def _torch_local_label() -> str | None:
    """Local part of the installed torch version (e.g. 'cu128', 'rocm6.2', 'xpu'),
    read from package metadata WITHOUT importing torch. None for a pure/CPU build."""
    try:
        from importlib.metadata import version
        v = version("torch")
    except Exception:
        return None
    if "+" not in v:
        return None
    return v.split("+", 1)[1]


def _mps_available() -> bool:
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def detect_backend() -> tuple[str, str | None]:
    """Return (backend_name, version) for the accelerator torch will use.

    backend_name is one of KNOWN_BACKENDS. version is the backend toolkit version
    where meaningful (cuda '12.8', rocm '6.2'), else None.
    """
    label = _torch_local_label()
    if label:
        if label.startswith("rocm"):
            return ("rocm", label[len("rocm"):] or None)
        if label.startswith("cu"):
            # CUDA version parsing lives in detection/cuda.py (the cuda branch).
            return ("cuda", get_bootstrap_torch_cuda())
        if label.startswith("xpu"):
            return ("xpu", None)
    # No accelerator label: mac wheels bundle MPS; everything else is CPU.
    if sys.platform == "darwin" and _mps_available():
        return ("mps", None)
    return ("cpu", None)


