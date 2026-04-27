from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable, List, Optional, Union

from ..config import (
    discover_config,
    load_config,
    CONFIG_FILE_NAME,
    ROOT_CONFIG_FILE_NAME,
)
from .helpers import USE_COMFY_ENV_VAR, _enable_windows_long_paths
from .plugin import (
    _install_apt_packages,
    _install_brew_packages,
    _install_node_dependencies,
    _reinstall_main_requirements,
    _collect_node_req_dirs,
    _install_to_main_env,
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
    Performs per-plugin work (apt/brew/node_reqs/main-env pip), then triggers a
    workspace-wide `pixi install --all` covering every plugin in this ComfyUI install.
    """
    if node_dir is None:
        node_dir = Path(inspect.stack()[1].filename).parent.resolve()

    log = log_callback or print
    _enable_windows_long_paths(log)

    if config is not None:
        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = node_dir / config_path
        cfg = load_config(config_path)
    else:
        cfg = discover_config(node_dir, root=True)

    if cfg is None:
        raise FileNotFoundError(f"No {ROOT_CONFIG_FILE_NAME} or {CONFIG_FILE_NAME} found in {node_dir}")

    if cfg.apt_packages:
        _install_apt_packages(cfg.apt_packages, log, dry_run)
    if cfg.brew_packages:
        _install_brew_packages(cfg.brew_packages, log, dry_run)

    node_req_dirs: List[Path] = []
    if cfg.node_reqs:
        _install_node_dependencies(cfg.node_reqs, node_dir, log, dry_run)
        _reinstall_main_requirements(node_dir, log, dry_run)
        node_req_dirs = _collect_node_req_dirs(cfg.node_reqs, node_dir.parent)
        for nr_dir in node_req_dirs:
            nr_cfg = discover_config(nr_dir, root=True)
            if nr_cfg:
                if nr_cfg.apt_packages:
                    _install_apt_packages(nr_cfg.apt_packages, log, dry_run)
                if nr_cfg.brew_packages:
                    _install_brew_packages(nr_cfg.brew_packages, log, dry_run)

    from ..settings import resolve_bool, GENERAL_DEFAULTS
    node_settings = cfg.settings if cfg.settings else None
    install_isolated = resolve_bool(
        "COMFY_ENV_INSTALL_ISOLATED", node_settings,
        GENERAL_DEFAULTS["COMFY_ENV_INSTALL_ISOLATED"],
    )
    install_main = resolve_bool(
        "COMFY_ENV_INSTALL_MAIN", node_settings,
        GENERAL_DEFAULTS["COMFY_ENV_INSTALL_MAIN"],
    )

    if install_isolated:
        # Workspace install -- picks up every plugin's comfy-env.toml under custom_nodes/
        from ..environment.paths import get_comfyui_dir
        comfyui_dir = get_comfyui_dir(node_dir)
        if comfyui_dir is None:
            log("[comfy-env] WARNING: Could not locate ComfyUI base; skipping workspace install")
        else:
            install_workspace(comfyui_dir, log=log, dry_run=dry_run)

    if install_main:
        _install_to_main_env(node_dir, log, dry_run, node_req_dirs=node_req_dirs)

    if not install_isolated and not install_main:
        log("\n[comfy-env] Both install targets disabled -- nothing to install")

    log("\nInstallation complete!")
    return True
