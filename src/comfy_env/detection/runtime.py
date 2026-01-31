"""
Runtime environment detection - combines all detection into a single snapshot.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional

from .cuda import detect_cuda_version
from .gpu import detect_cuda_environment
from .platform import get_platform_tag, _get_os_name


@dataclass
class RuntimeEnv:
    """
    Detected runtime environment for wheel resolution.

    Contains all variables needed for wheel URL template expansion.
    """
    # OS/Platform
    os_name: str  # linux, windows, darwin
    platform_tag: str  # linux_x86_64, win_amd64, macosx_...

    # Python
    python_version: str  # 3.10, 3.11, 3.12
    python_short: str  # 310, 311, 312

    # CUDA
    cuda_version: Optional[str]  # 12.8, 12.4, None
    cuda_short: Optional[str]  # 128, 124, None

    # PyTorch (detected or configured)
    torch_version: Optional[str]  # 2.8.0, 2.5.1
    torch_short: Optional[str]  # 280, 251
    torch_mm: Optional[str]  # 28, 25 (major.minor without dot)

    # GPU info
    gpu_name: Optional[str] = None
    gpu_compute: Optional[str] = None  # sm_89, sm_100

    @classmethod
    def detect(cls, torch_version: Optional[str] = None) -> "RuntimeEnv":
        """
        Detect runtime environment from current system.

        Args:
            torch_version: Optional PyTorch version override. If not provided,
                          attempts to detect from installed torch.

        Returns:
            RuntimeEnv with detected values.
        """
        # OS detection
        os_name = _get_os_name()

        # Platform tag
        platform_tag = get_platform_tag()

        # Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        py_short = f"{sys.version_info.major}{sys.version_info.minor}"

        # CUDA version
        cuda_version = detect_cuda_version()
        cuda_short = cuda_version.replace(".", "") if cuda_version else None

        # PyTorch version
        if torch_version is None:
            torch_version = _detect_torch_version()

        torch_short = None
        torch_mm = None
        if torch_version:
            torch_short = torch_version.replace(".", "")
            parts = torch_version.split(".")[:2]
            torch_mm = "".join(parts)

        # GPU info
        gpu_name = None
        gpu_compute = None
        try:
            env = detect_cuda_environment()
            if env.gpus:
                gpu = env.gpus[0]
                gpu_name = gpu.name
                gpu_compute = gpu.sm_version()
        except Exception:
            pass

        return cls(
            os_name=os_name,
            platform_tag=platform_tag,
            python_version=py_version,
            python_short=py_short,
            cuda_version=cuda_version,
            cuda_short=cuda_short,
            torch_version=torch_version,
            torch_short=torch_short,
            torch_mm=torch_mm,
            gpu_name=gpu_name,
            gpu_compute=gpu_compute,
        )

    def as_dict(self) -> Dict[str, str]:
        """Convert to dict for template substitution."""
        py_minor = self.python_version.split(".")[-1] if self.python_version else ""

        result = {
            "os": self.os_name,
            "platform": self.platform_tag,
            "python_version": self.python_version,
            "py_version": self.python_version,
            "py_short": self.python_short,
            "py_minor": py_minor,
            "py_tag": f"cp{self.python_short}",
        }

        if self.cuda_version:
            result["cuda_version"] = self.cuda_version
            result["cuda_short"] = self.cuda_short
            result["cuda_major"] = self.cuda_version.split(".")[0]

        if self.torch_version:
            result["torch_version"] = self.torch_version
            result["torch_short"] = self.torch_short
            result["torch_mm"] = self.torch_mm
            parts = self.torch_version.split(".")[:2]
            result["torch_dotted_mm"] = ".".join(parts)

        return result

    def __str__(self) -> str:
        parts = [
            f"Python {self.python_version}",
            f"CUDA {self.cuda_version}" if self.cuda_version else "CPU",
        ]
        if self.torch_version:
            parts.append(f"PyTorch {self.torch_version}")
        if self.gpu_name:
            parts.append(f"GPU: {self.gpu_name}")
        return ", ".join(parts)


def detect_runtime(torch_version: Optional[str] = None) -> RuntimeEnv:
    """Detect runtime environment. Convenience function."""
    return RuntimeEnv.detect(torch_version)


def _detect_torch_version() -> Optional[str]:
    """Detect installed PyTorch version."""
    try:
        import torch
        version = torch.__version__
        if '+' in version:
            version = version.split('+')[0]
        return version
    except ImportError:
        return None


def parse_wheel_requirement(req: str) -> tuple[str, Optional[str]]:
    """
    Parse a wheel requirement string.

    Examples:
        "nvdiffrast==0.4.0" -> ("nvdiffrast", "0.4.0")
        "pytorch3d>=0.7.8" -> ("pytorch3d", "0.7.8")
        "torch-cluster" -> ("torch-cluster", None)

    Returns:
        Tuple of (package_name, version_or_None).
    """
    for op in ['==', '>=', '<=', '~=', '!=', '>', '<']:
        if op in req:
            parts = req.split(op, 1)
            return (parts[0].strip(), parts[1].strip())

    return (req.strip(), None)
