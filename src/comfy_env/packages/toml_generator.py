"""Generate pixi.toml from ComfyEnvConfig."""

import copy
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

    # Workspace
    workspace = pixi_data.setdefault("workspace", {})
    workspace.setdefault("name", node_dir.name)
    workspace.setdefault("version", "0.1.0")
    workspace.setdefault("channels", ["conda-forge"])
    workspace.setdefault("platforms", [get_pixi_platform()])

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

    # PyTorch CUDA index
    if cfg.has_cuda and cuda_version:
        pypi_options = pixi_data.setdefault("pypi-options", {})
        pytorch_index = f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')[:3]}"
        extra_urls = pypi_options.setdefault("extra-index-urls", [])
        if pytorch_index not in extra_urls: extra_urls.append(pytorch_index)

    # Enforce torch version
    if cfg.has_cuda and torch_version:
        pypi_deps = pixi_data.setdefault("pypi-dependencies", {})
        torch_minor = int(torch_version.split(".")[1])
        pypi_deps["torch"] = f">={torch_version},<{torch_version.split('.')[0]}.{torch_minor + 1}"

    return pixi_data


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
