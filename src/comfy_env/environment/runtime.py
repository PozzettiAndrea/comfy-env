"""`RuntimeEnv` -- the machine facts comfy-env resolves against, in one object.

This is the payload behind ``comfy-env info [--json]``, and the JSON form is a
**supported seam**: comfy-test (a separately released repo) needs the workspace
root and the ABI tag, and until this existed it imported
``comfy_env.environment.cache.get_workspace_dir`` and the private ``_abi_tag``
across the repo boundary. Private names are free to move; this schema is not.

Field names are therefore part of the contract. Add fields freely; do not
rename or repurpose one without treating it as a breaking change.

Lives in `environment/` rather than `detection/` because it composes machine
facts (detection) WITH workspace identity (the ABI tag and workspace root).
`detection` is the lower layer and must not import upward into `environment`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class RuntimeEnv:
    """Everything `install()` keys its decisions on, resolved once."""

    os_name: str
    platform_tag: str          # pixi platform, e.g. "linux-64"
    cpu_arch: str              # "x86_64" | "aarch64"
    python_version: Optional[str]
    torch_version: Optional[str]
    cuda_version: Optional[str]
    gpu_name: Optional[str]
    gpu_compute: Optional[str]     # "8.9"
    gpu_vram_mb: Optional[int]
    workspace_dir: Optional[str]   # machine-wide env root
    abi_tag: Optional[str]         # e.g. "py313-torch2.10-cu128"
    comfy_env_version: Optional[str]

    @classmethod
    def detect(cls) -> "RuntimeEnv":
        from ..detection import (
            _get_os_name, get_pixi_platform, get_bootstrap_python_version,
            get_bootstrap_torch_version, detect_cuda_version, detect_gpu,
        )
        from ..detection.arch import cpu_arch

        gpu = detect_gpu()

        # Both of these are best-effort: `comfy-env info` must still print
        # something useful on a machine where the workspace was never created
        # or torch is absent. A diagnostic that dies is worthless.
        workspace_dir = abi_tag = None
        try:
            from .cache import get_workspace_dir, _abi_tag
            workspace_dir = str(get_workspace_dir(None))
            abi_tag = _abi_tag()
        except Exception:
            pass

        version = None
        try:
            from importlib.metadata import version as _v
            version = _v("comfy-env")
        except Exception:
            pass

        return cls(
            os_name=_get_os_name(),
            platform_tag=get_pixi_platform(),
            cpu_arch=cpu_arch(),
            python_version=get_bootstrap_python_version(),
            torch_version=get_bootstrap_torch_version(),
            cuda_version=detect_cuda_version() or None,
            gpu_name=gpu.name if gpu else None,
            gpu_compute=gpu.cc_str() if gpu else None,
            gpu_vram_mb=gpu.vram_total_mb if gpu else None,
            workspace_dir=workspace_dir,
            abi_tag=abi_tag,
            comfy_env_version=version,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
