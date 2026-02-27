"""Environment setup for ComfyUI prestartup."""

import os
import sys
from pathlib import Path
from typing import Optional

from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


def is_comfy_env_enabled() -> bool:
    return os.environ.get(USE_COMFY_ENV_VAR, "1").lower() not in ("0", "false", "no", "off")


def _find_env_dirs(node_dir: str) -> list:
    """Recursively find all _env_* directories under node_dir (for debug info only)."""
    envs = []
    for root, dirs, _ in os.walk(node_dir):
        for d in dirs:
            if d.startswith("_env_"):
                envs.append(os.path.join(root, d))
        dirs[:] = [d for d in dirs if not d.startswith("_env_")]
    return envs


def _ensure_base_directory():
    """Ensure comfy.cli_args.args.base_directory is set.

    Some nodes (e.g. KJNodes) resolve relative paths via args.base_directory.
    If the user didn't pass --base-directory, it defaults to None and relative
    paths break when cwd != ComfyUI root (common on CI / portable builds).
    """
    try:
        from comfy.cli_args import args
        if args.base_directory:
            return
        import folder_paths
        args.base_directory = folder_paths.base_path
    except Exception:
        pass


def setup_env(node_dir: Optional[str] = None) -> None:
    """Set up comfy-env runtime. Call in prestartup_script.py."""
    if node_dir is None:
        import inspect
        node_dir = str(Path(inspect.stack()[1].filename).parent)

    import faulthandler
    faulthandler.enable(file=sys.stderr)

    node_name = os.path.basename(node_dir)

    # Log isolation envs
    sub_envs = _find_env_dirs(node_dir)
    if sub_envs:
        print(f"[comfy-env] {node_name}: {len(sub_envs)} isolation env(s):", file=sys.stderr)
        for env_path in sub_envs:
            print(f"[comfy-env]   {os.path.basename(env_path)} -> {env_path}", file=sys.stderr)
    else:
        print(f"[comfy-env] {node_name}: no isolation envs", file=sys.stderr)

    if not is_comfy_env_enabled(): return
    dedupe_libomp()

    _ensure_base_directory()
    print("[comfy-env] prestartup complete", file=sys.stderr, flush=True)
