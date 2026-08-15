"""Environment setup for ComfyUI prestartup."""

import os
import sys
from pathlib import Path

from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"


def is_comfy_env_enabled():
    from ..settings import ISOLATE
    return ISOLATE


def _find_env_dirs(node_dir):
    """Recursively find comfy-env.toml files under node_dir (for debug info only)."""
    # Same two shapes the binder supports (nodes/ and nodes/<subdir>) --
    # kept consistent with install discovery; no recursive walk.
    node_dir = Path(node_dir)
    out = []
    nodes_dir = node_dir / "nodes"
    if nodes_dir.is_dir():
        if (nodes_dir / "comfy-env.toml").is_file():
            out.append(str(nodes_dir))
        for child in sorted(nodes_dir.iterdir()):
            if (child.is_dir() and not child.name.startswith((".", "_"))
                    and (child / "comfy-env.toml").is_file()):
                out.append(str(child))
    return out


def _ensure_base_directory():
    """Ensure comfy.cli_args.args.base_directory is set.

    Some nodes resolve relative paths via args.base_directory.
    If the user didn't pass --base-directory, it defaults to None.
    """
    try:
        from comfy.cli_args import args
        if args.base_directory:
            return
        import folder_paths
        args.base_directory = folder_paths.base_path
    except Exception:
        pass



def setup_env(node_dir=None):
    """Set up comfy-env runtime. Call in prestartup_script.py."""
    if node_dir is None:
        import inspect
        node_dir = str(Path(inspect.stack()[1].filename).parent)

    import faulthandler
    faulthandler.enable(file=sys.stderr)

    node_name = os.path.basename(node_dir)

    sub_envs = _find_env_dirs(node_dir)
    if sub_envs:
        from .cache import get_env_name, get_workspace_env_dir
        print(f"[comfy-env] {node_name}: {len(sub_envs)} isolation env(s):", file=sys.stderr)
        for env_path in sub_envs:
            print(f"[comfy-env]   {os.path.basename(env_path)} -> {env_path}", file=sys.stderr)
            try:
                config_path = Path(env_path) / "comfy-env.toml"
                env_name = get_env_name(node_dir, config_path)
                target = get_workspace_env_dir(None, env_name)
                status = "OK" if target.is_dir() else "MISSING -- run install.py"
                print(f"[comfy-env]     env: {target}  [{status}]", file=sys.stderr)
            except Exception as e:
                print(f"[comfy-env]     env: <resolution failed: {e}>", file=sys.stderr)
    else:
        print(f"[comfy-env] {node_name}: no isolation envs", file=sys.stderr)

    if not is_comfy_env_enabled():
        print("[comfy-env] prestartup complete (isolation disabled)",
              file=sys.stderr, flush=True)
        return

    dedupe_libomp()

    _ensure_base_directory()
    print("[comfy-env] prestartup complete", file=sys.stderr, flush=True)
