"""
GPU detection and information gathering.

Detects NVIDIA GPUs using multiple fallback methods: NVML -> PyTorch -> nvidia-smi -> sysfs
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cuda import CUDA_VERSION_ENV_VAR


# Architecture mapping: compute capability -> architecture name
COMPUTE_TO_ARCH = {
    (5, 0): "Maxwell", (5, 2): "Maxwell", (5, 3): "Maxwell",
    (6, 0): "Pascal", (6, 1): "Pascal", (6, 2): "Pascal",
    (7, 0): "Volta", (7, 2): "Volta", (7, 5): "Turing",
    (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere", (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell", (10, 1): "Blackwell", (10, 2): "Blackwell",
}

# Cache for GPU detection results
_cache: tuple[float, "CUDAEnvironment | None"] = (0, None)
CACHE_TTL = 60


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    compute_capability: tuple[int, int]
    architecture: str
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    uuid: str = ""
    pci_bus_id: str = ""
    driver_version: str = ""

    def cc_str(self) -> str:
        """Return compute capability as string (e.g., '8.9')."""
        return f"{self.compute_capability[0]}.{self.compute_capability[1]}"

    def sm_version(self) -> str:
        """Return SM version string (e.g., 'sm_89')."""
        return f"sm_{self.compute_capability[0]}{self.compute_capability[1]}"


@dataclass
class CUDAEnvironment:
    """Complete CUDA environment information."""
    gpus: list[GPUInfo] = field(default_factory=list)
    driver_version: str = ""
    cuda_runtime_version: str = ""
    recommended_cuda: str = ""
    detection_method: str = ""


def compute_capability_to_architecture(major: int, minor: int) -> str:
    """Convert compute capability to architecture name."""
    if arch := COMPUTE_TO_ARCH.get((major, minor)):
        return arch
    # Fallback for unknown compute capabilities
    if major >= 10:
        return "Blackwell"
    if major == 9:
        return "Hopper"
    if major == 8:
        return "Ada" if minor >= 9 else "Ampere"
    if major == 7:
        return "Turing" if minor >= 5 else "Volta"
    if major == 6:
        return "Pascal"
    return "Maxwell" if major == 5 else "Unknown"


def get_compute_capability(gpu_index: int = 0) -> tuple[int, int] | None:
    """Get compute capability of a specific GPU."""
    env = detect_cuda_environment()
    if env.gpus and gpu_index < len(env.gpus):
        return env.gpus[gpu_index].compute_capability
    return None


def get_recommended_cuda_version(gpus: list[GPUInfo] | None = None) -> str:
    """
    Get recommended CUDA version based on GPU architecture.

    - Blackwell (sm_100+): requires CUDA 12.8
    - Modern GPUs (Turing+): CUDA 12.8
    - Legacy GPUs (Pascal): CUDA 12.4
    """
    # Check environment override first
    if override := os.environ.get(CUDA_VERSION_ENV_VAR, "").strip():
        if "." not in override and len(override) >= 2:
            return f"{override[:-1]}.{override[-1]}"
        return override

    if gpus is None:
        gpus = detect_cuda_environment().gpus

    if not gpus:
        return ""

    # Blackwell requires 12.8
    for gpu in gpus:
        if gpu.compute_capability[0] >= 10:
            return "12.8"

    # Pascal and older use 12.4
    for gpu in gpus:
        cc = gpu.compute_capability
        if cc[0] < 7 or (cc[0] == 7 and cc[1] < 5):
            return "12.4"

    # Modern GPUs use 12.8
    return "12.8"


def detect_gpu() -> GPUInfo | None:
    """Detect primary GPU. Returns None if no GPU found."""
    env = detect_cuda_environment()
    return env.gpus[0] if env.gpus else None


def detect_gpus() -> list[GPUInfo]:
    """Detect all GPUs."""
    return detect_cuda_environment().gpus


def detect_cuda_environment(force_refresh: bool = False) -> CUDAEnvironment:
    """
    Detect complete CUDA environment.

    Uses multiple detection methods with fallbacks:
    1. NVML (pynvml) - most reliable
    2. PyTorch - if torch is installed
    3. nvidia-smi - CLI fallback
    4. sysfs - Linux kernel interface
    """
    global _cache
    if not force_refresh and _cache[1] and time.time() - _cache[0] < CACHE_TTL:
        return _cache[1]

    gpus, method = None, "none"
    for name, fn in [
        ("nvml", _detect_nvml),
        ("torch", _detect_torch),
        ("smi", _detect_smi),
        ("sysfs", _detect_sysfs),
    ]:
        if result := fn():
            gpus = result
            method = name
            break

    env = CUDAEnvironment(
        gpus=gpus or [],
        driver_version=_get_driver_version(),
        cuda_runtime_version=_get_cuda_version(),
        recommended_cuda=get_recommended_cuda_version(gpus or []),
        detection_method=method,
    )
    _cache = (time.time(), env)
    return env


def get_gpu_summary() -> str:
    """Human-readable GPU summary."""
    env = detect_cuda_environment()
    if not env.gpus:
        override = os.environ.get(CUDA_VERSION_ENV_VAR)
        if override:
            return f"No NVIDIA GPU detected (using {CUDA_VERSION_ENV_VAR}={override})"
        return f"No NVIDIA GPU detected (set {CUDA_VERSION_ENV_VAR} to override)"

    lines = [f"Detection: {env.detection_method}"]
    if env.driver_version:
        lines.append(f"Driver: {env.driver_version}")
    if env.cuda_runtime_version:
        lines.append(f"CUDA: {env.cuda_runtime_version}")
    lines.append(f"Recommended: CUDA {env.recommended_cuda}")
    lines.append("")
    for gpu in env.gpus:
        vram = f"{gpu.vram_total_mb}MB" if gpu.vram_total_mb else "?"
        lines.append(f"  GPU {gpu.index}: {gpu.name} ({gpu.sm_version()}) [{gpu.architecture}] {vram}")
    return "\n".join(lines)


# Internal detection functions

def _parse_cc(s: str) -> tuple[int, int]:
    """Parse compute capability string."""
    try:
        if "." in s:
            p = s.split(".")
            return (int(p[0]), int(p[1]))
        if len(s) >= 2:
            return (int(s[:-1]), int(s[-1]))
    except (ValueError, IndexError):
        pass
    return (0, 0)


def _detect_nvml() -> list[GPUInfo] | None:
    """Detect GPUs using NVML (pynvml)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if not count:
                return None
            gpus = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append(GPUInfo(
                    index=i,
                    name=name,
                    compute_capability=cc,
                    architecture=compute_capability_to_architecture(*cc),
                    vram_total_mb=mem.total // (1024 * 1024),
                    vram_free_mb=mem.free // (1024 * 1024),
                ))
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def _detect_torch() -> list[GPUInfo] | None:
    """Detect GPUs using PyTorch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(GPUInfo(
                index=i,
                name=p.name,
                compute_capability=(p.major, p.minor),
                architecture=compute_capability_to_architecture(p.major, p.minor),
                vram_total_mb=p.total_memory // (1024 * 1024),
            ))
        return gpus if gpus else None
    except Exception:
        return None


def _detect_smi() -> list[GPUInfo] | None:
    """Detect GPUs using nvidia-smi."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id,compute_cap,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if r.returncode != 0:
            return None
        gpus = []
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            cc = _parse_cc(p[4])
            gpus.append(GPUInfo(
                index=int(p[0]) if p[0].isdigit() else len(gpus),
                name=p[1],
                uuid=p[2] if len(p) > 2 else "",
                pci_bus_id=p[3] if len(p) > 3 else "",
                compute_capability=cc,
                architecture=compute_capability_to_architecture(*cc),
                vram_total_mb=int(p[5]) if len(p) > 5 and p[5].isdigit() else 0,
                vram_free_mb=int(p[6]) if len(p) > 6 and p[6].isdigit() else 0,
                driver_version=p[7] if len(p) > 7 else "",
            ))
        return gpus if gpus else None
    except Exception:
        return None


def _detect_sysfs() -> list[GPUInfo] | None:
    """Detect GPUs using Linux sysfs (minimal info)."""
    try:
        pci_path = Path("/sys/bus/pci/devices")
        if not pci_path.exists():
            return None
        gpus = []
        for d in sorted(pci_path.iterdir()):
            vendor = (d / "vendor").read_text().strip().lower() if (d / "vendor").exists() else ""
            if "10de" not in vendor:  # NVIDIA vendor ID
                continue
            cls = (d / "class").read_text().strip() if (d / "class").exists() else ""
            if not (cls.startswith("0x0300") or cls.startswith("0x0302")):  # VGA/3D controller
                continue
            gpus.append(GPUInfo(
                index=len(gpus),
                name="NVIDIA GPU",
                pci_bus_id=d.name,
                compute_capability=(0, 0),
                architecture="Unknown"
            ))
        return gpus if gpus else None
    except Exception:
        return None


def _get_driver_version() -> str:
    """Get NVIDIA driver version."""
    try:
        import pynvml
        pynvml.nvmlInit()
        v = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        return v.decode() if isinstance(v, bytes) else v
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if r.returncode == 0:
            return r.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return ""


def _get_cuda_version() -> str:
    """Get CUDA runtime version."""
    try:
        import torch
        if torch.cuda.is_available() and torch.version.cuda:
            return torch.version.cuda
    except Exception:
        pass
    return ""
