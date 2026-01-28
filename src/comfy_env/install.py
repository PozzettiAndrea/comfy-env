"""
Installation API for comfy-env.

Example:
    from comfy_env import install
    install()  # Auto-discovers comfy-env.toml and installs
"""

import inspect
from pathlib import Path
from typing import Callable, List, Optional, Set, Union

from .config.types import ComfyEnvConfig, NodeReq
from .config.parser import load_config, discover_config


def install(
    config: Optional[Union[str, Path]] = None,
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """
    Install dependencies from comfy-env.toml.

    Args:
        config: Optional path to comfy-env.toml. Auto-discovered if not provided.
        node_dir: Optional node directory. Auto-discovered from caller if not provided.
        log_callback: Optional callback for logging. Defaults to print.
        dry_run: If True, show what would be installed without installing.

    Returns:
        True if installation succeeded.
    """
    # Auto-discover caller's directory if not provided
    if node_dir is None:
        frame = inspect.stack()[1]
        caller_file = frame.filename
        node_dir = Path(caller_file).parent.resolve()

    log = log_callback or print

    # Load config
    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        cfg = load_config(config_path)
    else:
        cfg = discover_config(node_dir)

    if cfg is None:
        raise FileNotFoundError(
            f"No comfy-env.toml found in {node_dir}. "
            "Create comfy-env.toml to define dependencies."
        )

    # Install apt packages first (Linux only)
    if cfg.apt_packages:
        _install_apt_packages(cfg.apt_packages, log, dry_run)

    # Install node dependencies
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)

    # Install everything via pixi
    _install_via_pixi(cfg, node_dir, log, dry_run)

    # Auto-discover and install isolated subdirectory environments
    _install_isolated_subdirs(node_dir, log, dry_run)

    log("\nInstallation complete!")
    return True


def _install_apt_packages(
    packages: List[str],
    log: Callable[[str], None],
    dry_run: bool,
) -> None:
    """Install apt packages (Linux only)."""
    import os
    import platform
    import shutil
    import subprocess

    if platform.system() != "Linux":
        log(f"[apt] Skipping apt packages (not Linux)")
        return

    log(f"\n[apt] Installing {len(packages)} system package(s):")
    for pkg in packages:
        log(f"  - {pkg}")

    if dry_run:
        log("  (dry run - no changes made)")
        return

    # Determine if we need sudo
    is_root = os.geteuid() == 0
    has_sudo = shutil.which("sudo") is not None
    use_sudo = not is_root and has_sudo
    prefix = ["sudo"] if use_sudo else []

    if not is_root and not has_sudo:
        log(f"[apt] Warning: No root access. Install manually:")
        log(f"  sudo apt-get update && sudo apt-get install -y {' '.join(packages)}")
        return

    # Run apt-get update (suppress output, just show errors)
    log("[apt] Updating package lists...")
    result = subprocess.run(
        prefix + ["apt-get", "update"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"[apt] Warning: apt-get update failed: {result.stderr.strip()}")

    # Run apt-get install
    log(f"[apt] Installing packages...")
    result = subprocess.run(
        prefix + ["apt-get", "install", "-y"] + packages,
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        log("[apt] System packages installed successfully!")
    else:
        log(f"[apt] Warning: Installation failed: {result.stderr.strip()}")
        log(f"[apt] You may need to install manually:")
        log(f"  sudo apt-get install -y {' '.join(packages)}")


def _install_node_dependencies(
    node_reqs: List[NodeReq],
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
) -> None:
    """Install node dependencies (other ComfyUI custom nodes)."""
    from .nodes import install_node_deps

    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")

    if dry_run:
        for req in node_reqs:
            node_path = custom_nodes_dir / req.name
            status = "exists" if node_path.exists() else "would clone"
            log(f"  {req.name}: {status}")
        return

    visited: Set[str] = {node_dir.name}
    install_node_deps(node_reqs, custom_nodes_dir, log, visited)


def _install_via_pixi(
    cfg: ComfyEnvConfig,
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
) -> None:
    """Install all packages via pixi."""
    from .pixi import pixi_install

    # Count what we're installing
    cuda_count = len(cfg.cuda_packages)

    # Count from passthrough (pixi-native format)
    deps = cfg.pixi_passthrough.get("dependencies", {})
    pypi_deps = cfg.pixi_passthrough.get("pypi-dependencies", {})

    if cuda_count == 0 and not deps and not pypi_deps:
        log("No packages to install")
        return

    log(f"\nInstalling via pixi:")
    if cuda_count:
        log(f"  CUDA packages: {', '.join(cfg.cuda_packages)}")
    if deps:
        log(f"  Conda packages: {len(deps)}")
    if pypi_deps:
        log(f"  PyPI packages: {len(pypi_deps)}")

    if dry_run:
        log("\n(dry run - no changes made)")
        return

    pixi_install(cfg, node_dir, log)


def _install_isolated_subdirs(
    node_dir: Path,
    log: Callable[[str], None],
    dry_run: bool,
) -> None:
    """Find and install comfy-env.toml in subdirectories."""
    from .pixi import pixi_install
    from .config.parser import CONFIG_FILE_NAME

    # Find all comfy-env.toml files in subdirectories (not root)
    for config_file in node_dir.rglob(CONFIG_FILE_NAME):
        if config_file.parent == node_dir:
            continue  # Skip root (already installed)

        sub_dir = config_file.parent
        relative = sub_dir.relative_to(node_dir)

        log(f"\n[isolated] Installing: {relative}")
        sub_cfg = load_config(config_file)

        if dry_run:
            log(f"  (dry run)")
            continue

        pixi_install(sub_cfg, sub_dir, log)


def verify_installation(
    packages: List[str],
    log: Callable[[str], None] = print,
) -> bool:
    """Verify that packages are importable."""
    all_ok = True
    for package in packages:
        import_name = package.replace("-", "_").split("[")[0]
        try:
            __import__(import_name)
            log(f"  {package}: OK")
        except ImportError as e:
            log(f"  {package}: FAILED ({e})")
            all_ok = False
    return all_ok
