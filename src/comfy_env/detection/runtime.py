"""Runtime environment detection - combines all detection into a single snapshot."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Optional

from .cuda import detect_cuda_version
from .gpu import detect_cuda_environment
from .platform import get_platform_tag, _get_os_name


@dataclass
class RuntimeEnv:
    """Detected runtime environment for wheel resolution."""
    os_name: str
    platform_tag: str
    python_version: str
    python_short: str
    cuda_version: Optional[str]
    cuda_short: Optional[str]
    torch_version: Optional[str]
    torch_short: Optional[str]
    torch_mm: Optional[str]
    gpu_name: Optional[str] = None
    gpu_compute: Optional[str] = None

    @classmethod
    def detect(cls, torch_version: Optional[str] = None) -> "RuntimeEnv":
        """Detect runtime environment from current system."""
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        cuda_version = detect_cuda_version()
        torch_version = torch_version or _detect_torch_version()

        gpu_name, gpu_compute = None, None
        try:
            env = detect_cuda_environment()
            if env.gpus:
                gpu_name, gpu_compute = env.gpus[0].name, env.gpus[0].sm_version()
        except Exception:
            pass

        return cls(
            os_name=_get_os_name(),
            platform_tag=get_platform_tag(),
            python_version=py_version,
            python_short=py_version.replace(".", ""),
            cuda_version=cuda_version,
            cuda_short=cuda_version.replace(".", "") if cuda_version else None,
            torch_version=torch_version,
            torch_short=torch_version.replace(".", "") if torch_version else None,
            torch_mm="".join(torch_version.split(".")[:2]) if torch_version else None,
            gpu_name=gpu_name,
            gpu_compute=gpu_compute,
        )

    def as_dict(self) -> Dict[str, str]:
        """Convert to dict for template substitution."""
        result = {
            "os": self.os_name,
            "platform": self.platform_tag,
            "python_version": self.python_version,
            "py_version": self.python_version,
            "py_short": self.python_short,
            "py_minor": self.python_version.split(".")[-1],
            "py_tag": f"cp{self.python_short}",
        }
        if self.cuda_version:
            result.update(cuda_version=self.cuda_version, cuda_short=self.cuda_short,
                         cuda_major=self.cuda_version.split(".")[0])
        if self.torch_version:
            result.update(torch_version=self.torch_version, torch_short=self.torch_short,
                         torch_mm=self.torch_mm, torch_dotted_mm=".".join(self.torch_version.split(".")[:2]))
        return result

    def __str__(self) -> str:
        parts = [f"Python {self.python_version}",
                 f"CUDA {self.cuda_version}" if self.cuda_version else "CPU"]
        if self.torch_version: parts.append(f"PyTorch {self.torch_version}")
        if self.gpu_name: parts.append(f"GPU: {self.gpu_name}")
        return ", ".join(parts)


def detect_runtime(torch_version: Optional[str] = None) -> RuntimeEnv:
    return RuntimeEnv.detect(torch_version)


def _detect_torch_version() -> Optional[str]:
    try:
        import torch
        return torch.__version__.split('+')[0]
    except ImportError:
        return None


def parse_wheel_requirement(req: str) -> tuple[str, Optional[str]]:
    """Parse 'pkg==1.0' -> ('pkg', '1.0')"""
    for op in ['==', '>=', '<=', '~=', '!=', '>', '<']:
        if op in req:
            parts = req.split(op, 1)
            return (parts[0].strip(), parts[1].strip())
    return (req.strip(), None)
