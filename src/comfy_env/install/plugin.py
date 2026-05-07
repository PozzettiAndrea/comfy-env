"""Per-plugin install: node_reqs and main-env pip install.

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


