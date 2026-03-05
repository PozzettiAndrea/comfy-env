"""Environment setup for ComfyUI prestartup."""

import os
import sys
from pathlib import Path
from typing import Optional

from .libomp import dedupe_libomp

USE_COMFY_ENV_VAR = "USE_COMFY_ENV"  # kept for backwards compat


def is_comfy_env_enabled() -> bool:
    from ..settings import ISOLATE
    return ISOLATE


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


_shareable_pool_applied = False


def _activate_attention(flash=False, sage=False):
    """Set attention flags on comfy.cli_args.args (already imported at prestartup time)."""
    try:
        from comfy.cli_args import args
        patches = []
        if sage and not getattr(args, 'use_sage_attention', False):
            args.use_sage_attention = True
            patches.append("sage")
        if flash and not getattr(args, 'use_flash_attention', False):
            args.use_flash_attention = True
            patches.append("flash")
        if patches:
            print(f"[comfy-env] auto-activated {' + '.join(patches)} attention",
                  file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[comfy-env] attention patch FAILED: {e}",
              file=sys.stderr, flush=True)


def _register_shareable_pool_hook():
    """Register sys.meta_path hook that fires on 'import comfy.model_management'.

    Creates a shareable CUDA memory pool in the parent process and patches
    ComfyUI's memory functions to query it correctly. This enables full
    zero-copy GPU tensor transfer between parent and worker processes.
    """
    global _shareable_pool_applied
    if _shareable_pool_applied:
        return

    class _ShareablePoolHook:

        def find_module(self, fullname, path=None):
            if fullname == "comfy.model_management" and not _shareable_pool_applied:
                return self
            return None

        def load_module(self, fullname):
            sys.meta_path.remove(self)
            import importlib
            mod = importlib.import_module(fullname)
            sys.modules[fullname] = mod
            self._apply(mod)
            return mod

        def _apply(self, mm):
            global _shareable_pool_applied
            if _shareable_pool_applied:
                return
            _shareable_pool_applied = True
            try:
                from ..isolation.workers.subprocess import (
                    _create_shareable_pool, _set_device_pool,
                    _get_pool_mem_stats, _trim_pool,
                )
                from ..isolation.workers import subprocess as sp
                import torch

                pool = _create_shareable_pool(device=0)
                _set_device_pool(0, pool)
                sp._parent_shareable_pool = pool

                # Patch get_free_memory
                _orig_get_free = mm.get_free_memory

                def _patched_get_free(dev=None, torch_free_too=False):
                    if dev is None:
                        dev = mm.get_torch_device()
                    if hasattr(dev, 'type') and dev.type != 'cuda':
                        return _orig_get_free(dev, torch_free_too)
                    reserved, active = _get_pool_mem_stats(pool)
                    mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
                    mem_free_torch = reserved - active
                    mem_free_total = mem_free_cuda + mem_free_torch
                    if torch_free_too:
                        return (mem_free_total, mem_free_torch)
                    return mem_free_total

                mm.get_free_memory = _patched_get_free

                # Patch get_total_memory
                _orig_get_total = mm.get_total_memory

                def _patched_get_total(dev=None, torch_total_too=False):
                    if dev is None:
                        dev = mm.get_torch_device()
                    if hasattr(dev, 'type') and dev.type != 'cuda':
                        return _orig_get_total(dev, torch_total_too)
                    reserved, _ = _get_pool_mem_stats(pool)
                    _, mem_total_cuda = torch.cuda.mem_get_info(dev)
                    if torch_total_too:
                        return (mem_total_cuda, reserved)
                    return mem_total_cuda

                mm.get_total_memory = _patched_get_total

                # Patch empty_cache to also trim our pool
                _orig_empty = torch.cuda.empty_cache

                def _patched_empty():
                    _orig_empty()
                    try:
                        _trim_pool(pool, 0)
                    except Exception:
                        pass

                torch.cuda.empty_cache = _patched_empty

                print("[comfy-env] shareable pool patch applied", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[comfy-env] shareable pool patch FAILED: {e}",
                      file=sys.stderr, flush=True)

    sys.meta_path.insert(0, _ShareablePoolHook())


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

    # Patches (apply regardless of isolation setting)
    from ..settings import _is_on, PATCH_DEFAULTS
    use_flash = _is_on("COMFY_ENV_PATCH_FLASH_ATTENTION",
                        PATCH_DEFAULTS["COMFY_ENV_PATCH_FLASH_ATTENTION"])
    use_sage = _is_on("COMFY_ENV_PATCH_SAGE_ATTENTION",
                       PATCH_DEFAULTS["COMFY_ENV_PATCH_SAGE_ATTENTION"])
    if use_flash or use_sage:
        _activate_attention(flash=use_flash, sage=use_sage)

    if not is_comfy_env_enabled():
        print("[comfy-env] prestartup complete (isolation disabled)",
              file=sys.stderr, flush=True)
        return
    dedupe_libomp()

    if _is_on("COMFY_ENV_PATCH_SHAREABLE_POOL",
              PATCH_DEFAULTS["COMFY_ENV_PATCH_SHAREABLE_POOL"]):
        _register_shareable_pool_hook()
        print("[comfy-env] shareable pool hook registered",
              file=sys.stderr, flush=True)

    _ensure_base_directory()
    print("[comfy-env] prestartup complete", file=sys.stderr, flush=True)
