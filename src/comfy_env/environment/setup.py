"""Environment setup for ComfyUI prestartup."""

import os
import sys
from pathlib import Path

from .libomp import dedupe_libomp



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
                if target.is_dir():
                    status = "OK"
                else:
                    # Name the pack on THIS line. The header carrying node_name
                    # is two lines up and scrolls away once several packs report,
                    # and this is the line people grep for. Remedy matches
                    # wrap.py: bare `comfy-env install` resolves the config from
                    # the CURRENT directory, so it fails from the ComfyUI root --
                    # --dir is the spelling that works from anywhere.
                    status = (f"{node_name}: MISSING -- run "
                              f"`comfy-env install --dir {node_dir}`")
                print(f"[comfy-env]     env: {target}  [{status}]", file=sys.stderr)
            except Exception as e:
                print(f"[comfy-env]     env: <resolution failed: {e}>", file=sys.stderr)
    else:
        print(f"[comfy-env] {node_name}: no isolation envs", file=sys.stderr)

    res = dedupe_libomp()
    if res.status != "not-darwin":
        print(f"[comfy-env] libomp: {res.summary()}", file=sys.stderr)
        if res.candidates and not res.touched:
            for sp in res.skipped_paths:
                print(f"[comfy-env] libomp:   skipped {sp}", file=sys.stderr)

    print("[comfy-env] prestartup complete", file=sys.stderr, flush=True)
