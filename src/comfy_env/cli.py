"""CLI for comfy-env."""

import argparse
import os
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

    # debug
    sub.add_parser("debug", help="Toggle debug logging categories")

    # cleanup
    sub.add_parser("cleanup", help="Remove orphaned environments")

    parsed = parser.parse_args(args)
    if not parsed.command:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init, "generate": cmd_generate, "install": cmd_install,
        "info": cmd_info, "doctor": cmd_doctor, "apt-install": cmd_apt_install,
        "debug": cmd_debug, "cleanup": cmd_cleanup,
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
# CUDA/conda/pypi packages go in comfy-env.toml (isolated subfolders only)

[apt]
packages = []

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


def cmd_debug(args) -> int:
    """Toggle debug logging categories with a curses TUI."""
    from .debug import CATEGORIES, SETTINGS_FILE

    # Read current state from settings file
    enabled = set()
    if SETTINGS_FILE.exists():
        try:
            for line in SETTINGS_FILE.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip().lower() in ("1", "true", "yes"):
                        enabled.add(k.strip())
        except Exception:
            pass

    # Also check live env vars (they override the file)
    for var, _ in CATEGORIES:
        val = os.environ.get(var, "")
        if val.lower() in ("1", "true", "yes"):
            enabled.add(var)

    try:
        import curses
        return _debug_tui(curses, CATEGORIES, enabled, SETTINGS_FILE)
    except ImportError:
        # No curses (Windows without windows-curses) — simple text fallback
        return _debug_text(CATEGORIES, enabled, SETTINGS_FILE)


def _debug_tui(curses, categories, enabled, settings_file):
    """Curses-based checkbox TUI."""
    import os

    selected = [var in enabled for var, _ in categories]
    n_cats = len(categories)
    # cursor positions: 0..n_cats-1 = checkboxes, n_cats = "Apply & Exit", n_cats+1 = "Quit"
    cursor = 0
    status_msg = ""

    def draw(stdscr):
        nonlocal cursor, status_msg
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 2, "comfy-env debug logging", curses.A_BOLD)
            stdscr.addstr(1, 2, "\u2501" * min(40, w - 4))

            for i, (var, label) in enumerate(categories):
                y = i + 3
                if y >= h - 4:
                    break
                check = "x" if selected[i] else " "
                if i > 0 and selected[0] and not selected[i]:
                    check = "*"
                attr = curses.A_REVERSE if cursor == i else 0
                line = f"  [{check}] {label:<40s} {var}"
                stdscr.addstr(y, 0, line[:w-1], attr)

            # Button row
            btn_y = n_cats + 4
            if btn_y < h - 2:
                # "Apply & Exit" button
                apply_attr = curses.A_REVERSE | curses.A_BOLD if cursor == n_cats else curses.A_BOLD
                stdscr.addstr(btn_y, 2, "[ Apply & Exit ]", apply_attr)

                # "Quit" button
                quit_attr = curses.A_REVERSE if cursor == n_cats + 1 else 0
                stdscr.addstr(btn_y, 22, "[ Quit ]", quit_attr)

            # Help
            help_y = btn_y + 2
            if help_y < h:
                stdscr.addstr(help_y, 2, "\u2191\u2193 navigate  Space toggle  Enter select  q quit",
                              curses.A_DIM)
                if help_y + 1 < h:
                    stdscr.addstr(help_y + 1, 2, "* = enabled via master switch",
                                  curses.A_DIM)

            # Status message
            if status_msg and help_y + 2 < h:
                color = curses.color_pair(1) if curses.has_colors() else curses.A_BOLD
                stdscr.addstr(help_y + 2, 2, status_msg, color)

            stdscr.refresh()

            key = stdscr.getch()
            status_msg = ""  # clear on next keypress

            if key in (ord('q'), ord('Q'), 27):  # q or ESC
                return 0
            elif key == curses.KEY_UP and cursor > 0:
                cursor -= 1
            elif key == curses.KEY_DOWN and cursor < n_cats + 1:
                cursor += 1
            elif key == ord(' '):
                if cursor < n_cats:
                    selected[cursor] = not selected[cursor]
                elif cursor == n_cats:
                    _save_debug_settings(categories, selected, settings_file)
                    return 0
                elif cursor == n_cats + 1:
                    return 0
            elif key in (curses.KEY_ENTER, 10, 13):
                if cursor < n_cats:
                    selected[cursor] = not selected[cursor]
                elif cursor == n_cats:
                    _save_debug_settings(categories, selected, settings_file)
                    return 0
                elif cursor == n_cats + 1:
                    return 0
            elif key == ord('a') or key == ord('A'):
                # Shortcut: apply immediately
                _save_debug_settings(categories, selected, settings_file)
                return 0

    return curses.wrapper(draw)


def _debug_text(categories, enabled, settings_file):
    """Simple text fallback for systems without curses."""
    import os

    selected = [var in enabled for var, _ in categories]

    print("comfy-env debug logging")
    print("=" * 40)
    for i, (var, label) in enumerate(categories):
        check = "x" if selected[i] else " "
        if i > 0 and selected[0] and not selected[i]:
            check = "*"
        print(f"  {i}. [{check}] {label:<40s} {var}")
    print()
    print("Enter numbers to toggle (space-separated), or 'save' to save, 'quit' to exit:")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if line.lower() in ("q", "quit", "exit"):
            return 0
        if line.lower() in ("s", "save"):
            _save_debug_settings(categories, selected, settings_file)
            print("Saved.")
            return 0
        for part in line.split():
            try:
                idx = int(part)
                if 0 <= idx < len(categories):
                    selected[idx] = not selected[idx]
            except ValueError:
                pass
        # Redisplay
        for i, (var, label) in enumerate(categories):
            check = "x" if selected[i] else " "
            if i > 0 and selected[0] and not selected[i]:
                check = "*"
            print(f"  {i}. [{check}] {label:<40s} {var}")


def _save_debug_settings(categories, selected, settings_file):
    """Write debug settings to ~/.comfy-env/debug.env."""
    import os
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# comfy-env debug settings (managed by `comfy-env debug`)\n"]
    for i, (var, label) in enumerate(categories):
        if selected[i]:
            lines.append(f"{var}=1\n")
        else:
            lines.append(f"# {var}=1\n")
    settings_file.write_text("".join(lines))
    # Also update current process env so changes take effect immediately
    for i, (var, _) in enumerate(categories):
        if selected[i]:
            os.environ[var] = "1"
        else:
            os.environ.pop(var, None)
    print(f"Saved to {settings_file}")


def cmd_cleanup(args) -> int:
    print("Cleanup is no longer needed -- envs are stored directly in node dirs.")
    print("To remove an env, delete the _env_* folder in the node directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
