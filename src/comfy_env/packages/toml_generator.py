"""Generate pixi.toml from ComfyEnvConfig."""

import copy
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from ..config import ComfyEnvConfig
from ..detection import get_recommended_cuda_version, get_pixi_platform
from .cuda_wheels import CUDA_TORCH_MAP


def _require_tomli_w():
    try:
        import tomli_w
        return tomli_w
    except ImportError:
        raise ImportError("tomli-w required: pip install tomli-w")


def generate_pixi_toml(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print) -> str:
    return _require_tomli_w().dumps(config_to_pixi_dict(cfg, node_dir, log))


def write_pixi_toml(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print) -> Path:
    tomli_w = _require_tomli_w()
    pixi_toml = node_dir / "pixi.toml"
    with open(pixi_toml, "wb") as f:
        tomli_w.dump(config_to_pixi_dict(cfg, node_dir, log), f)
    log(f"Generated {pixi_toml}")
    return pixi_toml


def config_to_pixi_dict(cfg: ComfyEnvConfig, node_dir: Path, log: Callable[[str], None] = print) -> Dict[str, Any]:
    pixi_data = copy.deepcopy(cfg.pixi_passthrough)

    cuda_version = torch_version = None
    if cfg.has_cuda and sys.platform != "darwin":
        cuda_version = get_recommended_cuda_version()
        if cuda_version:
            torch_version = CUDA_TORCH_MAP.get(".".join(cuda_version.split(".")[:2]), "2.8")
            log(f"CUDA {cuda_version} -> PyTorch {torch_version}")

    # Pin torch/torchvision versions in pypi-dependencies if they're cuda packages
    # This ensures transitive dependencies (pytorch_lightning, timm, etc.) get the correct torch version
    pytorch_packages = {"torch", "torchvision", "torchaudio"}
    torchvision_map = {"2.8": "0.23", "2.4": "0.19"}

    if cfg.cuda_packages and sys.platform != "darwin":
        pypi_deps = pixi_data.setdefault("pypi-dependencies", {})
        # Use detected torch_version for GPU, or default "2.8" for CPU
        pin_version = torch_version or "2.8"
        for pkg in cfg.cuda_packages:
            if pkg in pytorch_packages:
                if pkg == "torch":
                    pypi_deps[pkg] = f"=={pin_version}.*"
                elif pkg == "torchvision":
                    tv_version = torchvision_map.get(pin_version, "0.23")
                    pypi_deps[pkg] = f"=={tv_version}.*"
                elif pkg == "torchaudio":
                    pypi_deps[pkg] = f"=={pin_version}.*"

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

    # Always require modern setuptools (fixes conda-forge Python version string parsing)
    pypi_deps = pixi_data.setdefault("pypi-dependencies", {})
    pypi_deps.setdefault("setuptools", ">=75.0")

    # On macOS, strip CUDA-specific pypi deps (e.g. cumm-cu121, spconv-cu121)
    if sys.platform == "darwin":
        pypi_deps = pixi_data.get("pypi-dependencies", {})
        cuda_pkgs = [k for k in pypi_deps if re.search(r"-cu\d+", k)]
        for k in cuda_pkgs:
            del pypi_deps[k]
            log(f"  Skipping {k} (CUDA-only, no macOS wheels)")

    # PyTorch index (CUDA or CPU) - skip on macOS
    if cfg.has_cuda and sys.platform != "darwin":
        pypi_options = pixi_data.setdefault("pypi-options", {})
        if cuda_version:
            pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
        else:
            # No GPU detected - use CPU index
            pytorch_index = "https://download.pytorch.org/whl/cpu"
            log("No GPU detected - using PyTorch CPU index")
        extra_urls = pypi_options.setdefault("extra-index-urls", [])
        if pytorch_index not in extra_urls: extra_urls.append(pytorch_index)
        pypi_options["index-strategy"] = "unsafe-best-match"

    return pixi_data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
