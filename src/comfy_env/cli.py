"""
CLI for comfy-env.

Provides the `comfy-env` command with subcommands:
- init: Create a default comfy-env.toml
- generate: Generate pixi.toml from comfy-env.toml
- install: Install dependencies from config
- info: Show runtime environment information
- resolve: Show resolved wheel URLs
- doctor: Verify installation
- list-packages: Show all packages in the built-in registry

Usage:
    comfy-env init ---> creates template comfy-env.toml
    comfy-env generate nodes/cgal/comfy-env.toml ---> nodes/cgal/pixi.toml
    comfy-env install ---> installs from comfy
    comfy-env install --dry-run

    comfy-env info

    comfy-env resolve nvdiffrast==0.4.0
    comfy-env resolve --all

    comfy-env doctor

    comfy-env list-packages
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

    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Resolve wheel URLs for packages",
        description="Show resolved wheel URLs without installing",
    )
    resolve_parser.add_argument(
        "packages",
        nargs="*",
        help="Package specs (e.g., nvdiffrast==0.4.0)",
    )
    resolve_parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Resolve all packages from config",
    )
    resolve_parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file",
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

    # list-packages command
    list_parser = subparsers.add_parser(
        "list-packages",
        help="Show all packages in the built-in registry",
        description="List CUDA packages that comfy-env knows how to install",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
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
        elif parsed.command == "resolve":
            return cmd_resolve(parsed)
        elif parsed.command == "doctor":
            return cmd_doctor(parsed)
        elif parsed.command == "list-packages":
            return cmd_list_packages(parsed)
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
# CUDA packages from comfy-env registry
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


def cmd_resolve(args) -> int:
    """Handle resolve command."""
    from .pixi import RuntimeEnv, parse_wheel_requirement
    from .pixi import PACKAGE_REGISTRY
    from .pixi.registry import get_cuda_short2
    from .config.parser import discover_env_config, load_env_from_file

    env = RuntimeEnv.detect()
    packages = []

    # Get packages from args or config
    if args.all or (not args.packages and args.config):
        if args.config:
            config = load_env_from_file(Path(args.config))
        else:
            config = discover_env_config(Path.cwd())

        if config and config.no_deps_requirements:
            packages = config.no_deps_requirements
        else:
            print("No CUDA packages found in config", file=sys.stderr)
            return 1
    elif args.packages:
        packages = args.packages
    else:
        print("Specify packages or use --all with a config file", file=sys.stderr)
        return 1

    print(f"Resolving wheels for: {env}")
    print("=" * 60)

    # Build template variables
    vars_dict = env.as_dict()
    if env.cuda_version:
        vars_dict["cuda_short2"] = get_cuda_short2(env.cuda_version)

    all_ok = True
    for pkg_spec in packages:
        package, version = parse_wheel_requirement(pkg_spec)
        pkg_lower = package.lower()

        try:
            if pkg_lower in PACKAGE_REGISTRY:
                config = PACKAGE_REGISTRY[pkg_lower]

                if "wheel_template" in config:
                    # Direct wheel URL template
                    effective_version = version or config.get("default_version")
                    if not effective_version:
                        print(f"  {package}: No version specified (no default in registry)")
                        all_ok = False
                        continue

                    vars_dict["version"] = effective_version
                    url = _substitute_template(config["wheel_template"], vars_dict)
                    print(f"  {package}=={effective_version}: resolved")
                    print(f"    {url}")

                elif "package_name" in config:
                    # PyPI variant (e.g., spconv-cu124)
                    pkg_name = _substitute_template(config["package_name"], vars_dict)
                    pkg_spec = f"{pkg_name}=={version}" if version else pkg_name
                    print(f"  {package}: installs as {pkg_spec} from PyPI")

                else:
                    print(f"  {package}: no wheel_template or package_name in registry")
                    all_ok = False
            else:
                print(f"  {package}: NOT in registry")
                print(f"    Add to [wheel_sources] in comfy-env.toml:")
                print(f'    {package} = "https://example.com/{package}-{{version}}+cu{{cuda_short}}-{{py_tag}}-{{platform}}.whl"')
                all_ok = False

        except Exception as e:
            print(f"  {package}: FAILED - {e}")
            all_ok = False

    return 0 if all_ok else 1


def _substitute_template(template: str, vars_dict: dict) -> str:
    """Substitute {var} placeholders in template."""
    result = template
    for key, value in vars_dict.items():
        if value is not None:
            result = result.replace(f"{{{key}}}", str(value))
    return result


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


def cmd_list_packages(args) -> int:
    """Handle list-packages command."""
    from .pixi import PACKAGE_REGISTRY

    if args.json:
        import json
        result = {}
        for name, config in PACKAGE_REGISTRY.items():
            result[name] = {
                "description": config.get("description", ""),
            }
            if "wheel_template" in config:
                result[name]["wheel_template"] = config["wheel_template"]
            if "package_name" in config:
                result[name]["package_name"] = config["package_name"]
            if "default_version" in config:
                result[name]["default_version"] = config["default_version"]
        print(json.dumps(result, indent=2))
        return 0

    print("Built-in CUDA Package Registry")
    print("=" * 60)
    print()
    print("These packages can be installed by adding them to comfy-env.toml:")
    print()
    print("  [cuda]")
    print('  nvdiffrast = "0.4.0"')
    print('  torch-scatter = "2.1.2"')
    print()
    print("Or override with custom wheel source:")
    print()
    print("  [wheel_sources]")
    print('  nvdiffrast = "https://my-mirror.com/nvdiffrast-{version}+cu{cuda_short}-{py_tag}-{platform}.whl"')
    print()
    print("-" * 60)

    # Group by type
    wheel_template_packages = []
    package_name_packages = []

    for name, config in PACKAGE_REGISTRY.items():
        desc = config.get("description", "")
        default = config.get("default_version", "")
        if "wheel_template" in config:
            wheel_template_packages.append((name, desc, default))
        elif "package_name" in config:
            package_name_packages.append((name, desc, config["package_name"]))

    if wheel_template_packages:
        print("\nDirect wheel URL packages:")
        for name, desc, default in sorted(wheel_template_packages):
            version_info = f" (default: {default})" if default else ""
            print(f"  {name:20} - {desc}{version_info}")

    if package_name_packages:
        print("\nPyPI variant packages:")
        for name, desc, pkg_template in sorted(package_name_packages):
            print(f"  {name:20} - {desc}")
            print(f"                       installs as: {pkg_template}")

    print()
    print("Template variables: {version}, {cuda_short}, {torch_mm}, {py_tag}, {platform}")
    print("See README for full documentation on writing wheel templates.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
