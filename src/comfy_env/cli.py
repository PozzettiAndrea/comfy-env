"""CLI for comfy-env."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import ROOT_CONFIG_FILE_NAME, CONFIG_FILE_NAME


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="comfy-env", description="Environment management for ComfyUI")
    parser.add_argument("--version", action="version", version=f"comfy-env {__version__}")
    sub = parser.add_subparsers(dest="command", help="Commands")

    # init
    p = sub.add_parser("init", help=f"Create {ROOT_CONFIG_FILE_NAME}")
    p.add_argument("--force", "-f", action="store_true", help="Overwrite existing")
    p.add_argument("--isolated", action="store_true", help=f"Create {CONFIG_FILE_NAME} instead (for isolated folders)")

    # generate
    p = sub.add_parser("generate", help="Generate pixi.toml from config")
    p.add_argument("config", type=str, help="Path to config file")
    p.add_argument("--force", "-f", action="store_true", help="Overwrite existing")

    # install
    p = sub.add_parser("install", help="Install dependencies")
    p.add_argument("--config", "-c", type=str, help="Config path")
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--dir", "-d", type=str, help="Node directory")

    # info
    p = sub.add_parser("info", help="Show runtime info")
    p.add_argument("--json", action="store_true", help="JSON output")

    # doctor
    p = sub.add_parser("doctor", help="Verify installation")
    p.add_argument("--package", "-p", type=str, help="Check specific package")
    p.add_argument("--config", "-c", type=str, help="Config path")

    # apt-install
    p = sub.add_parser("apt-install", help="Install apt packages (Linux)")
    p.add_argument("--config", "-c", type=str, help="Config path")
    p.add_argument("--dry-run", action="store_true", help="Preview only")

    parsed = parser.parse_args(args)
    if not parsed.command:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init, "generate": cmd_generate, "install": cmd_install,
        "info": cmd_info, "doctor": cmd_doctor, "apt-install": cmd_apt_install,
    }

    try:
        return commands.get(parsed.command, lambda _: 1)(parsed)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


ROOT_DEFAULT_CONFIG = """\
# comfy-env-root.toml - Main node config
# PyPI deps go in requirements.txt

[cuda]
packages = []

[apt]
packages = []

[dependencies]
# cgal = "*"

[env_vars]
# KMP_DUPLICATE_LIB_OK = "TRUE"

[node_reqs]
# ComfyUI_essentials = "cubiq/ComfyUI_essentials"
"""

ISOLATED_DEFAULT_CONFIG = """\
# comfy-env.toml - Isolated folder config
python = "3.11"

[dependencies]
# cgal = "*"

[pypi-dependencies]
# trimesh = { version = "*", extras = ["easy"] }

[env_vars]
# SOME_VAR = "value"
"""


def cmd_init(args) -> int:
    if getattr(args, 'isolated', False):
        config_path = Path.cwd() / CONFIG_FILE_NAME
        content = ISOLATED_DEFAULT_CONFIG
    else:
        config_path = Path.cwd() / ROOT_CONFIG_FILE_NAME
        content = ROOT_DEFAULT_CONFIG

    if config_path.exists() and not args.force:
        print(f"Already exists: {config_path}\nUse --force to overwrite", file=sys.stderr)
        return 1
    config_path.write_text(content)
    print(f"Created {config_path}")
    return 0


def cmd_generate(args) -> int:
    from .config import load_config
    from .packages.toml_generator import write_pixi_toml

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"Not found: {config_path}", file=sys.stderr)
        return 1

    node_dir = config_path.parent
    pixi_path = node_dir / "pixi.toml"
    if pixi_path.exists() and not args.force:
        print(f"Already exists: {pixi_path}\nUse --force to overwrite", file=sys.stderr)
        return 1

    config = load_config(config_path)
    if not config:
        print(f"Failed to load: {config_path}", file=sys.stderr)
        return 1

    print(f"Generating pixi.toml from {config_path}")
    write_pixi_toml(config, node_dir)
    print(f"Created {pixi_path}\nNext: cd {node_dir} && pixi install")
    return 0


def cmd_install(args) -> int:
    from .install import install
    node_dir = Path(args.dir) if args.dir else Path.cwd()
    try:
        install(config=args.config, node_dir=node_dir, dry_run=args.dry_run)
        return 0
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1


def cmd_info(args) -> int:
    from .detection import RuntimeEnv
    env = RuntimeEnv.detect()

    if args.json:
        import json
        print(json.dumps(env.as_dict(), indent=2))
        return 0

    print("Runtime Environment\n" + "=" * 40)
    print(f"  OS:       {env.os_name}")
    print(f"  Platform: {env.platform_tag}")
    print(f"  Python:   {env.python_version}")
    print(f"  CUDA:     {env.cuda_version or 'Not detected'}")
    print(f"  PyTorch:  {env.torch_version or 'Not installed'}")
    if env.gpu_name:
        print(f"  GPU:      {env.gpu_name}")
        if env.gpu_compute: print(f"  Compute:  {env.gpu_compute}")
    print()
    return 0


def cmd_doctor(args) -> int:
    from .install import verify_installation
    from .config import load_config, discover_config

    print("Diagnostics\n" + "=" * 40)
    print("\n1. Environment")
    cmd_info(argparse.Namespace(json=False))

    print("2. Packages")
    packages = []
    if args.package:
        packages = [args.package]
    else:
        cfg = load_config(Path(args.config)) if args.config else discover_config(Path.cwd())
        if cfg:
            packages = list(cfg.pixi_passthrough.get("pypi-dependencies", {}).keys()) + cfg.cuda_packages

    if packages:
        return 0 if verify_installation(packages) else 1
    print("  No packages to verify")
    return 0


def cmd_apt_install(args) -> int:
    import os, shutil, subprocess, platform
    if platform.system() != "Linux":
        print("apt-install: Linux only", file=sys.stderr)
        return 1

    # Check root config first, then regular
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        root_path = Path.cwd() / ROOT_CONFIG_FILE_NAME
        config_path = root_path if root_path.exists() else Path.cwd() / CONFIG_FILE_NAME

    if not config_path.exists():
        print(f"Not found: {config_path}", file=sys.stderr)
        return 1

    from .config.parser import load_config
    packages = load_config(config_path).apt_packages
    if not packages:
        print("No apt packages in config")
        return 0

    print(f"Packages: {', '.join(packages)}")
    use_sudo = os.geteuid() != 0 and shutil.which("sudo")
    prefix = ["sudo"] if use_sudo else []

    if args.dry_run:
        print(f"Would run: {'sudo ' if use_sudo else ''}apt-get install -y {' '.join(packages)}")
        return 0

    if os.geteuid() != 0 and not shutil.which("sudo"):
        print("Need root. Run: sudo apt-get install -y " + " ".join(packages), file=sys.stderr)
        return 1

    subprocess.run(prefix + ["apt-get", "update"], capture_output=False)
    result = subprocess.run(prefix + ["apt-get", "install", "-y"] + packages, capture_output=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
