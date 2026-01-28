"""
CLI for comfy-env.

Provides the `comfy-env` command with subcommands:
- init: Create a default comfy-env.toml
- generate: Generate pixi.toml from comfy-env.toml
- install: Install dependencies from config
- info: Show runtime environment information
- doctor: Verify installation

Usage:
    comfy-env init ---> creates template comfy-env.toml
    comfy-env generate nodes/cgal/comfy-env.toml ---> nodes/cgal/pixi.toml
    comfy-env install ---> installs from comfy
    comfy-env install --dry-run

    comfy-env info

    comfy-env doctor
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for comfy-env CLI."""
    parser = argparse.ArgumentParser(
        prog="comfy-env",
        description="Environment management for ComfyUI custom nodes",
    )
    parser.add_argument(
        "--version", action="version", version=f"comfy-env {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser(
        "init",
        help="Create a default comfy-env.toml config file",
        description="Initialize a new comfy-env.toml configuration file in the current directory",
    )
    init_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing config file",
    )

    # generate command
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate pixi.toml from comfy-env.toml",
        description="Parse comfy-env.toml and generate a pixi.toml in the same directory",
    )
    generate_parser.add_argument(
        "config",
        type=str,
        help="Path to comfy-env.toml",
    )
    generate_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing pixi.toml",
    )

    # install command
    install_parser = subparsers.add_parser(
        "install",
        help="Install dependencies from config",
        description="Install CUDA wheels and dependencies from comfy-env.toml",
    )
    install_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file (default: auto-discover)",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without installing",
    )
    install_parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory containing the config (default: current directory)",
    )

    # info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show runtime environment information",
        description="Display detected Python, CUDA, PyTorch, and GPU information",
    )
    info_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    # doctor command
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Verify installation and diagnose issues",
        description="Check if packages are properly installed and importable",
    )
    doctor_parser.add_argument(
        "--package", "-p",
        type=str,
        help="Check specific package",
    )
    doctor_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file",
    )

    # apt-install command
    apt_parser = subparsers.add_parser(
        "apt-install",
        help="Install system packages from [apt] section (Linux only)",
        description="Read [apt] packages from comfy-env.toml and install via apt-get",
    )
    apt_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to comfy-env.toml (default: auto-discover)",
    )
    apt_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be installed without installing",
    )

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        return 0

    try:
        if parsed.command == "init":
            return cmd_init(parsed)
        elif parsed.command == "generate":
            return cmd_generate(parsed)
        elif parsed.command == "install":
            return cmd_install(parsed)
        elif parsed.command == "info":
            return cmd_info(parsed)
        elif parsed.command == "doctor":
            return cmd_doctor(parsed)
        elif parsed.command == "apt-install":
            return cmd_apt_install(parsed)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


DEFAULT_CONFIG = """\
# comfy-env.toml - Environment configuration for ComfyUI custom nodes
# Documentation: https://github.com/PozzettiAndrea/comfy-env

[system]
# System packages required (apt on Linux, brew on macOS)
linux = []

[environment]
python = "3.11"
cuda_version = "auto"
pytorch_version = "auto"

[environment.cuda]
# CUDA packages from https://pozzettiandrea.github.io/cuda-wheels/
# Example: nvdiffrast = "0.4.0"

[environment.packages]
requirements = []
"""


def cmd_init(args) -> int:
    """Handle init command."""
    config_path = Path.cwd() / "comfy-env.toml"

    if config_path.exists() and not args.force:
        print(f"Config file already exists: {config_path}", file=sys.stderr)
        print("Use --force to overwrite", file=sys.stderr)
        return 1

    config_path.write_text(DEFAULT_CONFIG)
    print(f"Created {config_path}")
    return 0


def cmd_generate(args) -> int:
    """Handle generate command - create pixi.toml from comfy-env.toml."""
    from .config.parser import load_config
    from .pixi import create_pixi_toml

    config_path = Path(args.config).resolve()

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    if config_path.name != "comfy-env.toml":
        print(f"Warning: Expected comfy-env.toml, got {config_path.name}", file=sys.stderr)

    node_dir = config_path.parent
    pixi_path = node_dir / "pixi.toml"

    if pixi_path.exists() and not args.force:
        print(f"pixi.toml already exists: {pixi_path}", file=sys.stderr)
        print("Use --force to overwrite", file=sys.stderr)
        return 1

    # Load the config
    config = load_config(config_path)
    if not config or not config.envs:
        print(f"No environments found in {config_path}", file=sys.stderr)
        return 1

    # Use the first environment
    env_name = next(iter(config.envs.keys()))
    env_config = config.envs[env_name]

    print(f"Generating pixi.toml from {config_path}")
    print(f"  Environment: {env_name}")
    print(f"  Python: {env_config.python}")

    # Generate pixi.toml
    result_path = create_pixi_toml(env_config, node_dir)

    print(f"Created {result_path}")
    print()
    print("Next steps:")
    print(f"  cd {node_dir}")
    print("  pixi install")
    return 0


def cmd_install(args) -> int:
    """Handle install command."""
    from .install import install

    node_dir = Path(args.dir) if args.dir else Path.cwd()

    try:
        install(
            config=args.config,
            node_dir=node_dir,
            dry_run=args.dry_run,
        )
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Installation failed: {e}", file=sys.stderr)
        return 1


def cmd_info(args) -> int:
    """Handle info command."""
    from .pixi import RuntimeEnv

    env = RuntimeEnv.detect()

    if args.json:
        import json
        print(json.dumps(env.as_dict(), indent=2))
        return 0

    print("Runtime Environment")
    print("=" * 40)
    print(f"  OS:           {env.os_name}")
    print(f"  Platform:     {env.platform_tag}")
    print(f"  Python:       {env.python_version}")

    if env.cuda_version:
        print(f"  CUDA:         {env.cuda_version}")
    else:
        print("  CUDA:         Not detected")

    if env.torch_version:
        print(f"  PyTorch:      {env.torch_version}")
    else:
        print("  PyTorch:      Not installed")

    if env.gpu_name:
        print(f"  GPU:          {env.gpu_name}")
        if env.gpu_compute:
            print(f"  Compute:      {env.gpu_compute}")

    print()
    return 0


def cmd_doctor(args) -> int:
    """Handle doctor command."""
    from .install import verify_installation
    from .config.parser import discover_env_config, load_env_from_file

    print("Running diagnostics...")
    print("=" * 40)

    # Check environment
    print("\n1. Environment")
    cmd_info(argparse.Namespace(json=False))

    # Check packages
    print("2. Package Verification")

    packages = []
    if args.package:
        packages = [args.package]
    elif args.config:
        config = load_env_from_file(Path(args.config))
        if config:
            packages = (config.requirements or []) + (config.no_deps_requirements or [])
    else:
        config = discover_env_config(Path.cwd())
        if config:
            packages = (config.requirements or []) + (config.no_deps_requirements or [])

    if packages:
        pkg_names = []
        for pkg in packages:
            name = pkg.split("==")[0].split(">=")[0].split("[")[0]
            pkg_names.append(name)

        all_ok = verify_installation(pkg_names)
        if all_ok:
            print("\nAll packages verified!")
            return 0
        else:
            print("\nSome packages failed verification.")
            return 1
    else:
        print("  No packages to verify (no config found)")
        return 0


def cmd_apt_install(args) -> int:
    """Handle apt-install command - install system packages from [apt] section."""
    import os
    import shutil
    import subprocess
    import platform

    if platform.system() != "Linux":
        print("apt-install is only supported on Linux", file=sys.stderr)
        return 1

    # Find config
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = Path.cwd() / "comfy-env.toml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Parse config to get apt packages
    from .config.parser import load_config
    config = load_config(config_path)

    if not config.apt_packages:
        print("No [apt] packages specified in config")
        return 0

    packages = config.apt_packages
    print(f"Found {len(packages)} apt package(s) to install:")
    for pkg in packages:
        print(f"  - {pkg}")

    # Determine if we need sudo
    is_root = os.geteuid() == 0
    has_sudo = shutil.which("sudo") is not None
    use_sudo = not is_root and has_sudo

    prefix = ["sudo"] if use_sudo else []

    if args.dry_run:
        print("\n[Dry run] Would run:")
        prefix_str = "sudo " if use_sudo else ""
        print(f"  {prefix_str}apt-get update && {prefix_str}apt-get install -y {' '.join(packages)}")
        return 0

    if not is_root and not has_sudo:
        print("\nError: Need root privileges to install apt packages.", file=sys.stderr)
        print("Run manually with:", file=sys.stderr)
        print(f"  sudo apt-get update && sudo apt-get install -y {' '.join(packages)}", file=sys.stderr)
        return 1

    # Run apt-get update
    print("\nUpdating package lists...")
    result = subprocess.run(
        prefix + ["apt-get", "update"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("Warning: apt-get update failed, continuing anyway...")

    # Run apt-get install
    print(f"\nInstalling: {' '.join(packages)}")
    result = subprocess.run(
        prefix + ["apt-get", "install", "-y"] + packages,
        capture_output=False,
    )

    if result.returncode == 0:
        print("\nSystem packages installed successfully!")
        return 0
    else:
        print("\nFailed to install some packages", file=sys.stderr)
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
