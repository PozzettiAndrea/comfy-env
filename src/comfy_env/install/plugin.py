"""Per-plugin install: node_packs and main-env pip install.

Called from `install()` in __init__.py for the plugin whose `install.py` invoked
`from comfy_env import install; install()`. Workspace-level (`pixi install --all`)
is handled separately in `workspace.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List



def _install_node_packs(
    node_packs: List[dict], node_dir: Path,
    log: Callable[[str], None], dry_run: bool,
) -> None:
    from ..packages.node_packs import install_node_packs
    custom_nodes_dir = node_dir.parent
    log(f"\nInstalling {len(node_packs)} node packs...")
    if dry_run:
        for req in node_packs:
            log(f"  {req['name']}: {'exists' if (custom_nodes_dir / req['name']).exists() else 'would clone'}")
        return
    install_node_packs(node_packs, custom_nodes_dir, log, {node_dir.name})


def _reinstall_main_requirements(
    node_dir: Path, log: Callable[[str], None], dry_run: bool,
) -> None:
    """Re-install main package's requirements.txt after node_packs to restore correct versions."""
    from ..packages.node_packs import install_requirements
    req_file = node_dir / "requirements.txt"
    if not req_file.exists():
        return
    log(f"\n[requirements] Re-installing main package requirements...")
    if not dry_run:
        install_requirements(node_dir, log)
