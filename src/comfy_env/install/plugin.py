"""Per-plugin install: apt/brew/node_reqs and main-env pip install.

Called from `install()` in __init__.py for the plugin whose `install.py` invoked
`from comfy_env import install; install()`. Workspace-level (`pixi install --all`)
is handled separately in `workspace.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from ..config import (
    ComfyEnvConfig,
    NodeDependency,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
)


def _install_apt_packages(packages: List[str], log: Callable[[str], None], dry_run: bool) -> None:
    from ..packages.apt import apt_install
    import platform
    if platform.system() != "Linux":
        return
    log(f"\n[apt] Installing: {', '.join(packages)}")
    if not dry_run:
        success = apt_install(packages, log)
        if not success:
            log("[apt] WARNING: Some apt packages failed to install. This may cause issues.")


def _install_brew_packages(packages: List[str], log: Callable[[str], None], dry_run: bool) -> None:
    from ..packages.brew import brew_install
    import platform
    if platform.system() != "Darwin":
        return
    log(f"\n[brew] Installing: {', '.join(packages)}")
    if not dry_run:
        success = brew_install(packages, log)
        if not success:
            log("[brew] WARNING: Some brew packages failed to install. This may cause issues.")


def _install_node_dependencies(
    node_reqs: List[NodeDependency], node_dir: Path,
    log: Callable[[str], None], dry_run: bool,
) -> None:
    from ..packages.node_dependencies import install_node_dependencies
    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")
    if dry_run:
        for req in node_reqs:
            log(f"  {req.name}: {'exists' if (custom_nodes_dir / req.name).exists() else 'would clone'}")
        return
    install_node_dependencies(node_reqs, custom_nodes_dir, log, {node_dir.name})


def _reinstall_main_requirements(
    node_dir: Path, log: Callable[[str], None], dry_run: bool,
) -> None:
    """Re-install main package's requirements.txt after node_reqs to restore correct versions."""
    from ..packages.node_dependencies import install_requirements
    req_file = node_dir / "requirements.txt"
    if not req_file.exists():
        return
    log(f"\n[requirements] Re-installing main package requirements...")
    if not dry_run:
        install_requirements(node_dir, log)


def _collect_node_req_dirs(
    node_reqs: List[NodeDependency],
    custom_nodes_dir: Path,
    visited: Optional[Set[str]] = None,
) -> List[Path]:
    """Recursively collect all directories of nodes installed via node_reqs."""
    visited = visited or set()
    result = []
    for dep in node_reqs:
        if dep.name in visited:
            continue
        visited.add(dep.name)
        node_path = custom_nodes_dir / dep.name
        if not node_path.exists():
            continue
        result.append(node_path)
        nested_cfg = discover_config(node_path)
        if nested_cfg and nested_cfg.node_reqs:
            result.extend(_collect_node_req_dirs(nested_cfg.node_reqs, custom_nodes_dir, visited))
    return result


def _install_to_main_env(
    node_dir: Path, log: Callable[[str], None], dry_run: bool,
    node_req_dirs: Optional[List[Path]] = None,
) -> None:
    """Install deps from comfy-env.toml files into the main Python env (isolation disabled)."""
    import subprocess

    log("\n[comfy-env] Installing to main env")
    all_pypi: dict = {}
    cuda_packages: List[str] = []
    conda_warnings: List[Tuple[str, List[str]]] = []

    scan_dirs = [node_dir]
    if node_req_dirs:
        scan_dirs.extend(node_req_dirs)
        log(f"  Scanning {len(node_req_dirs)} node_req(s): {', '.join(d.name for d in node_req_dirs)}")

    for scan_dir in scan_dirs:
        for config_file in scan_dir.rglob(CONFIG_FILE_NAME):
            if config_file.parent == scan_dir:
                continue
            cfg = load_config(config_file)
            try:
                rel = config_file.parent.relative_to(scan_dir)
                label = f"{scan_dir.name}/{rel}" if scan_dir != node_dir else str(rel)
            except ValueError:
                label = str(config_file.parent)
            pypi = cfg.pixi_passthrough.get("pypi-dependencies", {})
            conda = cfg.pixi_passthrough.get("dependencies", {})
            if pypi:
                log(f"  [{label}] PyPI: {', '.join(pypi.keys())}")
                all_pypi.update(pypi)
            if cfg.cuda_packages:
                cuda_packages.extend(cfg.cuda_packages)
            if conda:
                conda_warnings.append((label, list(conda.keys())))

    if conda_warnings:
        for rel, pkgs in conda_warnings:
            log(f"  WARNING: [{rel}] has conda deps ({', '.join(pkgs)}) -- cannot install to main env")

    if not all_pypi and not cuda_packages:
        log("  No packages to install")
        return

    pip_args: List[str] = []
    for name, spec in all_pypi.items():
        if isinstance(spec, dict):
            version = spec.get("version", "*")
            extras = spec.get("extras", [])
            pkg = name
            if extras:
                pkg = f"{name}[{','.join(extras)}]"
            if version and version != "*":
                pkg = f"{pkg}{version}"
            pip_args.append(pkg)
        elif isinstance(spec, str) and spec != "*":
            pip_args.append(f"{name}{spec}")
        else:
            pip_args.append(name)

    pytorch_packages = {"torch", "torchvision", "torchaudio"}
    cuda_only = [p for p in cuda_packages if p not in pytorch_packages]
    if cuda_only:
        try:
            from ..packages.cuda_wheels import get_wheel_url
            from ..detection import get_recommended_cuda_version
            import torch
            torch_ver = ".".join(torch.__version__.split(".")[:2])
            cuda_ver = get_recommended_cuda_version()
            py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
            for package in cuda_only:
                url = get_wheel_url(package, torch_ver, cuda_ver, py_ver)
                if url:
                    pip_args.append(url)
                    log(f"  {package} from {url}")
                else:
                    log(f"  WARNING: No cuda-wheel for {package} (cu{cuda_ver}/torch{torch_ver}/py{py_ver})")
        except Exception as e:
            log(f"  WARNING: Could not resolve cuda-wheels: {e}")

    if pip_args:
        log(f"\n  pip install {' '.join(pip_args)}")
        if not dry_run:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install"] + pip_args,
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                log(f"  WARNING: pip install failed:\n{result.stderr}")
            else:
                log("  Installed successfully")
