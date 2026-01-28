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

    # Set persistent env vars (for OpenMP settings, etc.)
    if cfg.env_vars:
        _set_persistent_env_vars(cfg.env_vars, log, dry_run)

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

    # Install each package individually (some may not exist on all distros)
    log("[apt] Installing packages...")
    installed = []
    skipped = []
    for pkg in packages:
        result = subprocess.run(
            prefix + ["apt-get", "install", "-y", pkg],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            installed.append(pkg)
            log(f"  [apt] Installed {pkg}")
        else:
            skipped.append(pkg)
            log(f"  [apt] Skipped {pkg} (not available)")

    if installed:
        log(f"[apt] Installed {len(installed)} package(s)")
    if skipped:
        log(f"[apt] Skipped {len(skipped)} unavailable package(s)")


def _set_persistent_env_vars(
    env_vars: dict,
    log: Callable[[str], None],
    dry_run: bool,
) -> None:
    """Set env vars permanently (survives restarts)."""
    import os
    import platform
    import subprocess
    from pathlib import Path

    if not env_vars:
        return

    system = platform.system()
    log(f"\n[env] Setting {len(env_vars)} persistent environment variable(s)...")

    for key, value in env_vars.items():
        log(f"  - {key}={value}")

    if dry_run:
        log("  (dry run - no changes made)")
        return

    if system == "Windows":
        # Windows: use setx (writes to registry)
        for key, value in env_vars.items():
            result = subprocess.run(
                ["setx", key, value],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                log(f"  [env] Set {key} (Windows registry)")
            else:
                log(f"  [env] Warning: Failed to set {key}: {result.stderr.strip()}")
        log("[env] Restart terminal/ComfyUI for changes to take effect")

    elif system == "Darwin":  # macOS
        # macOS: launchctl for GUI apps + zshrc for terminal
        for key, value in env_vars.items():
            subprocess.run(["launchctl", "setenv", key, value], capture_output=True)
            log(f"  [env] Set {key} (launchctl)")

        # Also add to zshrc for terminal (zsh is default on macOS)
        _add_to_shell_profile(env_vars, log)

    else:  # Linux
        _add_to_shell_profile(env_vars, log)


def _add_to_shell_profile(
    env_vars: dict,
    log: Callable[[str], None],
) -> None:
    """Add env vars to shell profile (Linux/macOS)."""
    import os
    from pathlib import Path

    # Determine shell profile
    shell = os.environ.get("SHELL", "/bin/bash")
    if "zsh" in shell:
        rc_file = Path.home() / ".zshrc"
    else:
        rc_file = Path.home() / ".bashrc"

    profile_file = Path.home() / ".comfy-env-profile"

    # Write env vars to our dedicated file
    with open(profile_file, "w") as f:
        f.write("# Generated by comfy-env - do not edit manually\n")
        for key, value in env_vars.items():
            f.write(f'export {key}="{value}"\n')
    log(f"  [env] Wrote {profile_file}")

    # Add source line to shell rc (only once)
    source_line = f'source "{profile_file}"'
    existing = rc_file.read_text() if rc_file.exists() else ""

    if source_line not in existing and str(profile_file) not in existing:
        with open(rc_file, "a") as f:
            f.write(f'\n# comfy-env environment variables\n')
            f.write(f'{source_line}\n')
        log(f"  [env] Added source line to {rc_file}")
    else:
        log(f"  [env] Already configured in {rc_file}")

    log("[env] Restart terminal/ComfyUI for changes to take effect")


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
