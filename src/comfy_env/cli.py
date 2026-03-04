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

    # settings
    sub.add_parser("settings", help="Configure comfy-env settings")

    # debug (alias for settings, opens on Debug tab)
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
        "settings": cmd_settings, "debug": cmd_debug, "cleanup": cmd_cleanup,
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


def _read_env_file(path):
    """Read KEY=VALUE file, return set of enabled keys."""
    enabled = set()
    if path.exists():
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip().lower() in ("1", "true", "yes"):
                        enabled.add(k.strip())
        except Exception:
            pass
    return enabled


def cmd_settings(args) -> int:
    """Configure comfy-env settings (tabbed TUI)."""
    return _open_settings_tui(initial_tab=0)


def cmd_debug(args) -> int:
    """Toggle debug logging categories (alias for settings, opens on Debug tab)."""
    return _open_settings_tui(initial_tab=1)


def _open_settings_tui(initial_tab=0) -> int:
    from .debug import CATEGORIES as DEBUG_CATEGORIES, SETTINGS_FILE as DEBUG_FILE
    from .settings import (GENERAL_SETTINGS, GENERAL_DEFAULTS, SETTINGS_FILE as GENERAL_FILE,
                           PATCH_SETTINGS, PATCH_DEFAULTS)

    # Read current state
    debug_enabled = _read_env_file(DEBUG_FILE)
    general_enabled = _read_env_file(GENERAL_FILE)
    patch_enabled = _read_env_file(GENERAL_FILE)  # patches stored in same file

    # Also check live env vars
    for var, _ in DEBUG_CATEGORIES:
        if os.environ.get(var, "").lower() in ("1", "true", "yes"):
            debug_enabled.add(var)
    for var, _ in GENERAL_SETTINGS:
        val = os.environ.get(var, "")
        if val.lower() in ("1", "true", "yes"):
            general_enabled.add(var)
        elif val == "" and GENERAL_DEFAULTS.get(var, False):
            general_enabled.add(var)  # default on
    for var, _ in PATCH_SETTINGS:
        val = os.environ.get(var, "")
        if val.lower() in ("1", "true", "yes"):
            patch_enabled.add(var)
        elif val == "" and PATCH_DEFAULTS.get(var, False):
            patch_enabled.add(var)

    tabs = [
        ("General", GENERAL_SETTINGS, general_enabled, GENERAL_FILE),
        ("Debug", DEBUG_CATEGORIES, debug_enabled, DEBUG_FILE),
        ("Patches", PATCH_SETTINGS, patch_enabled, GENERAL_FILE),
    ]

    try:
        import curses
        return _settings_tui(curses, tabs, initial_tab)
    except ImportError:
        return _settings_text(tabs, initial_tab)


def _settings_tui(curses, tabs, initial_tab):
    """Curses-based tabbed settings TUI."""

    tab_names = [t[0] for t in tabs]
    tab_items = [t[1] for t in tabs]  # list of [(var, label), ...]
    tab_selected = [[var in t[2] for var, _ in t[1]] for t in tabs]
    tab_files = [t[3] for t in tabs]

    active_tab = initial_tab
    cursor = 0
    status_msg = ""

    def cur_items():
        return tab_items[active_tab]

    def cur_sel():
        return tab_selected[active_tab]

    def n_items():
        return len(cur_items())

    def draw(stdscr):
        nonlocal active_tab, cursor, status_msg
        curses.curs_set(0)
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)

        while True:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            stdscr.addstr(0, 2, "comfy-env settings", curses.A_BOLD)
            stdscr.addstr(1, 2, "\u2501" * min(40, w - 4))

            # Tab bar
            col = 2
            for i, name in enumerate(tab_names):
                if i == active_tab:
                    label = f" \u25b8 {name} "
                    attr = curses.A_BOLD | curses.A_REVERSE
                else:
                    label = f"   {name} "
                    attr = curses.A_DIM
                if col + len(label) < w:
                    stdscr.addstr(2, col, label, attr)
                col += len(label) + 1

            # Checkboxes
            items = cur_items()
            sel = cur_sel()
            ni = n_items()
            for i, (var, label) in enumerate(items):
                y = i + 4
                if y >= h - 4:
                    break
                check = "x" if sel[i] else " "
                # Debug tab: master switch indicator
                if active_tab == 1 and i > 0 and sel[0] and not sel[i]:
                    check = "*"
                attr = curses.A_REVERSE if cursor == i else 0
                line = f"  [{check}] {label:<48s} {var}"
                stdscr.addstr(y, 0, line[:w-1], attr)

            # Button row
            btn_y = ni + 5
            if btn_y < h - 2:
                apply_attr = curses.A_REVERSE | curses.A_BOLD if cursor == ni else curses.A_BOLD
                stdscr.addstr(btn_y, 2, "[ Apply & Exit ]", apply_attr)
                quit_attr = curses.A_REVERSE if cursor == ni + 1 else 0
                stdscr.addstr(btn_y, 22, "[ Quit ]", quit_attr)

            # Help
            help_y = btn_y + 2
            if help_y < h:
                stdscr.addstr(help_y, 2,
                              "Tab/\u2190\u2192 switch tab  \u2191\u2193 navigate  Space toggle  q quit",
                              curses.A_DIM)
                if active_tab == 1 and help_y + 1 < h:
                    stdscr.addstr(help_y + 1, 2, "* = enabled via master switch",
                                  curses.A_DIM)

            # Status message
            if status_msg:
                sy = help_y + 2 if active_tab == 1 else help_y + 1
                if sy < h:
                    color = curses.color_pair(1) if curses.has_colors() else curses.A_BOLD
                    stdscr.addstr(sy, 2, status_msg, color)

            stdscr.refresh()
            key = stdscr.getch()
            status_msg = ""

            ni = n_items()  # refresh after potential tab switch

            if key in (ord('q'), ord('Q'), 27):  # q or ESC
                return 0
            elif key == 9 or key == curses.KEY_RIGHT:  # Tab or Right
                active_tab = (active_tab + 1) % len(tabs)
                cursor = 0
            elif key == curses.KEY_LEFT:
                active_tab = (active_tab - 1) % len(tabs)
                cursor = 0
            elif key == curses.KEY_UP and cursor > 0:
                cursor -= 1
            elif key == curses.KEY_DOWN and cursor < n_items() + 1:
                cursor += 1
            elif key in (ord(' '), curses.KEY_ENTER, 10, 13):
                ni = n_items()
                if cursor < ni:
                    cur_sel()[cursor] = not cur_sel()[cursor]
                elif cursor == ni:
                    _save_all_settings(tab_items, tab_selected, tab_files)
                    return 0
                elif cursor == ni + 1:
                    return 0
            elif key in (ord('a'), ord('A')):
                _save_all_settings(tab_items, tab_selected, tab_files)
                return 0

    return curses.wrapper(draw)


def _settings_text(tabs, initial_tab):
    """Simple text fallback for systems without curses."""
    tab_items = [t[1] for t in tabs]
    tab_selected = [[var in t[2] for var, _ in t[1]] for t in tabs]
    tab_files = [t[3] for t in tabs]

    print("comfy-env settings")
    print("=" * 40)
    offset = 0
    offsets = []
    for ti, (name, items, _, _) in enumerate(tabs):
        print(f"\n  --- {name} ---")
        offsets.append(offset)
        for i, (var, label) in enumerate(items):
            check = "x" if tab_selected[ti][i] else " "
            if ti == 1 and i > 0 and tab_selected[ti][0] and not tab_selected[ti][i]:
                check = "*"
            print(f"  {offset + i}. [{check}] {label:<48s} {var}")
        offset += len(items)
    print()
    print("Enter numbers to toggle (space-separated), 'save' to save, 'quit' to exit:")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if line.lower() in ("q", "quit", "exit"):
            return 0
        if line.lower() in ("s", "save"):
            _save_all_settings(tab_items, tab_selected, tab_files)
            print("Saved.")
            return 0
        for part in line.split():
            try:
                idx = int(part)
                # Find which tab and local index
                for ti in range(len(tabs)):
                    if idx < offsets[ti] + len(tab_items[ti]):
                        local = idx - offsets[ti]
                        if 0 <= local < len(tab_items[ti]):
                            tab_selected[ti][local] = not tab_selected[ti][local]
                        break
            except ValueError:
                pass
        # Redisplay
        for ti, (name, items, _, _) in enumerate(tabs):
            print(f"\n  --- {name} ---")
            for i, (var, label) in enumerate(items):
                check = "x" if tab_selected[ti][i] else " "
                if ti == 1 and i > 0 and tab_selected[ti][0] and not tab_selected[ti][i]:
                    check = "*"
                print(f"  {offsets[ti] + i}. [{check}] {label:<48s} {var}")


def _save_all_settings(tab_items, tab_selected, tab_files):
    """Write settings to their respective files (merges tabs sharing the same file)."""
    # Group by file path (General and Patches may share the same file)
    from collections import defaultdict
    file_entries = defaultdict(list)  # filepath -> [(var, enabled), ...]
    for items, selected, filepath in zip(tab_items, tab_selected, tab_files):
        for i, (var, _) in enumerate(items):
            file_entries[filepath].append((var, selected[i]))
            # Update current process env
            os.environ[var] = "1" if selected[i] else "0"

    for filepath, entries in file_entries.items():
        filepath.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# comfy-env settings (managed by `comfy-env settings`)\n"]
        for var, enabled in entries:
            lines.append(f"{var}={'1' if enabled else '0'}\n")
        filepath.write_text("".join(lines))

    unique_files = list(dict.fromkeys(str(f) for f in tab_files))
    print(f"Saved to {', '.join(unique_files)}")


def cmd_cleanup(args) -> int:
    print("Cleanup is no longer needed -- envs are stored directly in node dirs.")
    print("To remove an env, delete the _env_* folder in the node directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
