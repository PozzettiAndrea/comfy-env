"""Per-plugin install: node_reqs and main-env pip install.

Called from `install()` in __init__.py for the plugin whose `install.py` invoked
`from comfy_env import install; install()`. Workspace-level (`pixi install --all`)
is handled separately in `workspace.py`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

from ..config import (
    ComfyEnvConfig,
    load_config,
    discover_config,
    CONFIG_FILE_NAME,
)


def _install_node_dependencies(
    node_reqs: List[dict], node_dir: Path,
    log: Callable[[str], None], dry_run: bool,
) -> None:
    from ..packages.node_dependencies import install_node_dependencies
    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_reqs)} node dependencies...")
    if dry_run:
        for req in node_reqs:
            log(f"  {req['name']}: {'exists' if (custom_nodes_dir / req['name']).exists() else 'would clone'}")
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


# comfy-env requirement line in a sibling pack's requirements.txt, e.g.
# "comfy-env==0.3.9" or "comfy_env >=0.3, <0.5". Captures the spec part.
_COMFY_ENV_REQ_RE = re.compile(
    r"^comfy[-_]env\s*(?P<spec>[=<>!~][^#;]*)?", re.IGNORECASE)


def check_sibling_comfy_env_pins(
    node_dir: Path, log: Callable[[str], None] = print,
) -> List[Tuple[str, str]]:
    """Warn when a sibling pack's requirements.txt pins an older comfy-env.

    The shared main env has one comfy-env; whichever pack (re)installs its
    requirements last wins. A stale ``comfy-env==0.3.x`` (or an upper bound
    below the installed version) in ANY pack silently downgrades comfy-env
    for every pack on the next reinstall. Warn-only: never fails an install.

    Returns [(pack_name, offending_line), ...] for testability.
    """
    from ..packages.cuda_wheels import _version_key

    try:
        from .. import __version__ as installed
    except Exception:
        return []
    installed_key = _version_key(installed)

    custom_nodes_dir = Path(node_dir).parent
    findings: List[Tuple[str, str]] = []
    try:
        siblings = sorted(p for p in custom_nodes_dir.iterdir() if p.is_dir())
    except OSError:
        return []

    for sib in siblings:
        if sib.name == Path(node_dir).name:
            continue
        req_file = sib / "requirements.txt"
        if not req_file.is_file():
            continue
        try:
            lines = req_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.split("#", 1)[0].strip()
            m = _COMFY_ENV_REQ_RE.match(line)
            if not m or not m.group("spec"):
                continue
            stale = False
            for clause in m.group("spec").split(","):
                clause = clause.strip()
                # ~= is deliberately not flagged: `~=0.4.10` still allows
                # newer patch releases, so it is not a downgrade pin.
                cm = re.match(r"(==|<=|<)\s*([0-9][0-9a-zA-Z.\-_+]*)", clause)
                if not cm:
                    continue
                op, ver = cm.group(1), cm.group(2)
                ver_key = _version_key(ver)
                if op in ("==", "<=") and ver_key < installed_key:
                    stale = True
                elif op == "<" and ver_key <= installed_key:
                    stale = True
            if stale:
                findings.append((sib.name, line))

    for pack, line in findings:
        log(f"[comfy-env] WARNING: {pack}/requirements.txt pins '{line}' but "
            f"comfy-env {installed} is installed. If that pack reinstalls its "
            f"requirements, comfy-env will be DOWNGRADED for every pack -- "
            f"update {pack} (or relax its pin).")
    return findings


def _collect_node_req_dirs(
    node_reqs: List[dict],
    custom_nodes_dir: Path,
    visited: Optional[Set[str]] = None,
) -> List[Path]:
    """Recursively collect all directories of nodes installed via node_reqs."""
    visited = visited or set()
    result = []
    for dep in node_reqs:
        name = dep["name"]
        if name in visited:
            continue
        visited.add(name)
        node_path = custom_nodes_dir / name
        if not node_path.exists():
            continue
        result.append(node_path)
        nested_cfg = discover_config(node_path)
        if nested_cfg and nested_cfg.node_reqs:
            result.extend(_collect_node_req_dirs(nested_cfg.node_reqs, custom_nodes_dir, visited))
    return result


