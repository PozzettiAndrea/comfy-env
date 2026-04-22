"""Generate pixi.toml from ComfyEnvConfig."""

import copy
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..config import ComfyEnvConfig
from ..detection import get_recommended_cuda_version, get_pixi_platform
from .cuda_wheels import CUDA_TORCH_MAP

# Torch bundle packages that can be inherited from the host
_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


def _require_tomli_w():
    try:
        import tomli_w
        return tomli_w
    except ImportError:
        raise ImportError("tomli-w required: pip install tomli-w")


def generate_pixi_toml(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print,
                       cuda_override: Optional[str] = None, torch_override: Optional[str] = None,
                       force_install_torch: bool = False) -> str:
    return _require_tomli_w().dumps(config_to_pixi_dict(cfg, node_dir, log,
                                                         cuda_override=cuda_override, torch_override=torch_override,
                                                         force_install_torch=force_install_torch))


def write_pixi_toml(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print,
                    cuda_override: Optional[str] = None, torch_override: Optional[str] = None,
                    force_install_torch: bool = False) -> Path:
    tomli_w = _require_tomli_w()
    pixi_toml = node_dir / "pixi.toml"
    with open(pixi_toml, "wb") as f:
        tomli_w.dump(config_to_pixi_dict(cfg, node_dir, log,
                                          cuda_override=cuda_override, torch_override=torch_override,
                                          force_install_torch=force_install_torch), f)
    log(f"Generated {pixi_toml}")
    return pixi_toml


def config_to_pixi_dict(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print,
                        cuda_override: Optional[str] = None, torch_override: Optional[str] = None,
                        force_install_torch: bool = True) -> Dict[str, Any]:
    pixi_data = copy.deepcopy(cfg.pixi_passthrough)

    # Detect CUDA/PyTorch versions and compute PyTorch index URL.
    # Overrides allow the install logic to force a specific combo (e.g. fallback to cu128/2.8,
    # or CPU mode matching the main env's torch version via `torch_override` with no cuda).
    cuda_version = torch_version = pytorch_index = None
    if cfg.has_cuda and sys.platform != "darwin":
        cuda_version = cuda_override or get_recommended_cuda_version()
        if cuda_version:
            torch_version = torch_override or CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")
            pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
            log(f"CUDA {cuda_version} -> PyTorch {torch_version}")
        else:
            # CPU mode: use torch_override if the caller detected a main-env torch version,
            # otherwise let the downstream fallback kick in.
            torch_version = torch_override
            pytorch_index = "https://download.pytorch.org/whl/cpu"
            if torch_version:
                log(f"No GPU detected - using PyTorch CPU index, matching torch {torch_version}")
            else:
                log("No GPU detected - using PyTorch CPU index")

    # Pixi always installs its own torch into each isolation env.
    # Add PyTorch packages to pypi-dependencies with per-package index.
    # This lets pixi resolve torch alongside all other deps in a single pass.
    torchvision_map = {
        "2.4": "0.19", "2.5": "0.20", "2.6": "0.21",
        "2.7": "0.22", "2.8": "0.23", "2.9": "0.24", "2.10": "0.25",
        "2.11": "0.26",
    }

    if cfg.cuda_packages and sys.platform != "darwin" and pytorch_index:
        pypi_deps = pixi_data.setdefault("pypi-dependencies", {})
        pin_version = torch_version or "2.8"

        # Explicit torch pin with the CUDA index so pixi doesn't pull in CPU-only torch
        # as a transitive dep (e.g. from timm).
        pypi_deps["torch"] = {"version": f"=={pin_version}.*", "index": pytorch_index}
        tv_ver = torchvision_map.get(pin_version, "0.23")
        pypi_deps["torchvision"] = {"version": f"=={tv_ver}.*", "index": pytorch_index}
        log(f"  torch=={pin_version}.* torchvision=={tv_ver}.* from {pytorch_index}")

        for pkg in cfg.cuda_packages:
            if pkg in _TORCH_PACKAGES:
                if pkg == "torchvision":
                    ver = torchvision_map.get(pin_version, "0.23")
                else:
                    ver = pin_version
                pypi_deps[pkg] = {"version": f"=={ver}.*", "index": pytorch_index}

    # Workspace
    workspace = pixi_data.setdefault("workspace", {})
    workspace.setdefault("name", node_dir.name)
    workspace.setdefault("version", "0.1.0")
    workspace.setdefault("channels", ["conda-forge"])
    current_platform = get_pixi_platform()
    workspace.setdefault("platforms", [current_platform])

    # Strip target sections for other platforms (pixi errors on unmatched targets)
    if "target" in pixi_data:
        non_matching = [k for k in pixi_data["target"] if k != current_platform]
        for k in non_matching:
            del pixi_data["target"][k]
        if not pixi_data["target"]:
            del pixi_data["target"]

    # System requirements
    if sys.platform.startswith("linux") or cuda_version:
        system_reqs = pixi_data.setdefault("system-requirements", {})
        if sys.platform.startswith("linux"):
            system_reqs.setdefault("libc", {"family": "glibc", "version": "2.35"})
        if cuda_version:
            system_reqs["cuda"] = cuda_version.split(".")[0]

    # Dependencies
    dependencies = pixi_data.setdefault("dependencies", {})
    py_version = cfg.python or f"{sys.version_info.major}.{sys.version_info.minor}"
    dependencies.setdefault("python", f"{py_version}.*")
    dependencies.setdefault("pip", "*")

    # Pin setuptools in conda deps (not pypi) to avoid pypi/conda version conflicts.
    # Range >=75.0,<82 satisfies both:
    #   - >=75.0 fixes conda-forge Python version string parsing
    #   - <82 satisfies torch (>=2.10) which requires setuptools<82
    dependencies.setdefault("setuptools", ">=75.0,<82")

    # Windows only: force libblas to the OpenBLAS variant so conda-forge doesn't pull in
    # mkl -> llvm-openmp -> Library\bin\libiomp5md.dll. That DLL shares the filename of
    # PyTorch's bundled Intel-OpenMP libiomp5md.dll but exports a different (smaller) symbol
    # set and causes fbgemm.dll to fail with WinError 127 on 'import torch' inside workers
    # (the _vcomp_* symbols fbgemm imports only exist in the Intel build shipped with torch).
    # Using setdefault so author overrides in comfy-env.toml still win.
    if sys.platform == "win32":
        dependencies.setdefault("libblas", {"version": "*", "build": "*openblas*"})

    # On macOS, strip CUDA-specific pypi deps (e.g. cumm-cu121, spconv-cu121)
    if sys.platform == "darwin":
        pypi_deps = pixi_data.get("pypi-dependencies", {})
        cuda_pkgs = [k for k in pypi_deps if re.search(r"-cu\d+", k)]
        for k in cuda_pkgs:
            del pypi_deps[k]
            log(f"  Skipping {k} (CUDA-only, no macOS wheels)")

    return pixi_data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
