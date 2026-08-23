from __future__ import annotations

import inspect
import os
from pathlib import Path

import sys
os.environ["PYTHONUNBUFFERED"] = "1"
# Unbuffer current process too (takes effect even when stdout is piped)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)
from typing import Callable, List, Optional, Union

from ..config import (
    discover_config,
    load_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)
from .helpers import USE_COMFY_ENV_VAR
from .plugin import (
    _install_node_dependencies,
    _reinstall_main_requirements,
    _collect_node_req_dirs,
    check_sibling_comfy_env_pins,
)
from .workspace import install_workspace
from .verify import verify_installation

__all__ = [
    "install",
    "install_workspace",
    "verify_installation",
    "USE_COMFY_ENV_VAR",
]


def install(
    config: Optional[Union[str, Path]] = None,
    node_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
) -> bool:
    """Install dependencies for the calling plugin and (re)build the workspace.

    Called from a plugin's `install.py` as `from comfy_env import install; install()`.
    Performs per-plugin work (node_reqs/main-env pip), then triggers a
    workspace-wide `pixi install --all` covering every plugin in this ComfyUI install.
    """
    if node_dir is None:
        # abspath, not resolve(): a symlinked/junctioned pack's install.py must
        # keep its custom_nodes-side spelling, or the comfyui-root walk starts
        # from the physical location and finds nothing (#8, third entry point).
        node_dir = Path(os.path.abspath(inspect.stack()[1].filename)).parent

    log = log_callback or print

    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        cfg = load_config(config_path)
    else:
        cfg = discover_config(node_dir, root=True)

    if cfg is None:
        raise FileNotFoundError(f"No {ROOT_CONFIG_FILE_NAME} or {CONFIG_FILE_NAME} found in {node_dir}")

    node_req_dirs: List[Path] = []
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)
        _reinstall_main_requirements(node_dir, log, dry_run)
        node_req_dirs = _collect_node_req_dirs(cfg.node_reqs, node_dir.parent)
        for nr_dir in node_req_dirs:
            nr_cfg = discover_config(nr_dir, root=True)
            if nr_cfg:
                pass  # Future: could process nr_cfg here

    # Surface stale comfy-env pins in sibling packs (warn-only): whichever
    # pack reinstalls its requirements last wins in the shared env, so an old
    # `comfy-env==X` elsewhere can silently downgrade us after this install.
    check_sibling_comfy_env_pins(node_dir, log)

    from ..environment.cache import find_comfyui_dir_from_node as get_comfyui_dir
    comfyui_dir = get_comfyui_dir(node_dir)
    if comfyui_dir is None:
        log("[comfy-env] WARNING: Could not locate ComfyUI base; skipping workspace install")
    else:
        install_workspace(comfyui_dir, log=log, dry_run=dry_run)

    log("\nInstallation complete!")
    return True
