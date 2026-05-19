"""Auto-materialize a missing pixi env at register_nodes time.

Gated behind the ``COMFY_ENV_AUTO_INSTALL`` setting (default OFF). When ON,
``ensure_env_materialized`` is invoked from ``isolation/wrap.py:_find_env_dir``
whenever the expected env dir doesn't exist on disk. Generates the per-env
pixi.toml from the node's comfy-env.toml (if not already present), runs
``pixi install --manifest-path <env>/pixi.toml`` synchronously, and returns
the materialized env dir. Blocks ComfyUI startup for the duration of the
install -- typically minutes for heavy CUDA nodes.

On failure, returns None so the caller falls back to in-process import
(ComfyUI still boots). On success, the worker subprocess + isolation path
in ``register_nodes`` takes over as if the env had been there all along.

Concurrent register_nodes calls (e.g. multiple custom nodes loading in
parallel during startup) are deduped via a per-env file lock at
``<env_manifest_dir>/.materializing.lock`` so we only spawn one
``pixi install`` per env. pixi has internal locking but our dedup keeps
log output and progress reporting coherent.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

# Stale-lock threshold: any lockfile older than this is treated as abandoned
# (probably a killed ComfyUI startup mid-install). Set high enough that a
# legitimate heavy CUDA install won't trip it.
_STALE_LOCK_AGE_SECONDS = 30 * 60


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = 30 * 60):
    """Cross-platform exclusive file lock with stale-recovery + polling wait.

    Stale-lock recovery: a lock file whose mtime is older than
    ``_STALE_LOCK_AGE_SECONDS`` is considered abandoned and removed before
    the new acquire attempt.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows. The
    lock file persists at ``lock_path`` while held; the *file descriptor*
    holds the OS lock. On context exit the descriptor is closed (releases
    the lock) and the file is unlinked (best-effort).

    Times out by raising RuntimeError if `timeout_seconds` elapses without
    acquiring -- caller handles that as a non-fatal failure (falls back to
    in-process import).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Stale-lock check before opening: if the existing lock file is older
    # than the staleness threshold, nuke it. Race-safe because the OS lock
    # below is what gates concurrent access; this is just opportunistic
    # cleanup of a known-dead lock.
    try:
        if lock_path.is_file():
            age = time.time() - lock_path.stat().st_mtime
            if age > _STALE_LOCK_AGE_SECONDS:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
    except OSError:
        pass

    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.time() + timeout_seconds
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.time() >= deadline:
                        raise RuntimeError(
                            f"timed out acquiring lock at {lock_path}"
                        )
                    time.sleep(0.5)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise RuntimeError(
                            f"timed out acquiring lock at {lock_path}"
                        )
                    time.sleep(0.5)

        # Record holder pid for human debugging. Best-effort.
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass

        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    os.lseek(fd, 0, 0)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                lock_path.unlink()
            except OSError:
                pass


def ensure_env_materialized(
    env_name: str,
    plugin_dir: Path,
    config_path: Path,
    comfyui_dir: Path,
    log: Callable[[str], None],
) -> Optional[Path]:
    """Ensure the materialized env dir for ``env_name`` exists on disk.

    Behavior:
      1. If the env is already materialized at
         ``<workspace>/envs/<env_name>/.pixi/envs/default/``, return it.
      2. Else acquire the per-env file lock.
      3. Re-check materialization (another process may have just finished).
      4. Generate ``<workspace>/envs/<env_name>/pixi.toml`` from the node's
         comfy-env.toml if it doesn't already exist.
      5. Run ``pixi install --manifest-path <env>/pixi.toml`` synchronously.
      6. Verify materialization and return the env dir, or None on failure.

    On any failure: log the reason and return None. The caller
    (``wrap.py:_find_env_dir``) then falls back to in-process import so
    ComfyUI still boots.

    Args:
        env_name: The pixi env name (e.g. ``"lito-nodes"``).
        plugin_dir: The node's plugin root (``custom_nodes/ComfyUI-X/``).
        config_path: Path to the node's ``comfy-env.toml``.
        comfyui_dir: The ComfyUI data dir (where ``custom_nodes/`` lives).
        log: Logger callback (typically ``isolation/wrap.py:_log``).
    """
    from ..environment.cache import (
        get_workspace_env_dir, get_env_manifest_dir,
    )
    from ..config import load_config

    # Already materialized? Nothing to do.
    env_dir = get_workspace_env_dir(comfyui_dir, env_name)
    if env_dir.is_dir():
        return env_dir

    env_manifest_dir = get_env_manifest_dir(env_name, comfyui_dir)
    env_manifest = env_manifest_dir / "pixi.toml"
    lock_path = env_manifest_dir / ".materializing.lock"

    try:
        with _file_lock(lock_path, timeout_seconds=_STALE_LOCK_AGE_SECONDS):
            # Re-check inside the lock: another process may have just finished.
            env_dir = get_workspace_env_dir(comfyui_dir, env_name)
            if env_dir.is_dir():
                return env_dir

            # Generate manifest on first auto-install for this env.
            if not env_manifest.is_file():
                _generate_env_manifest(
                    env_name=env_name,
                    config_path=config_path,
                    env_manifest_dir=env_manifest_dir,
                    log=log,
                )

            log(
                f"[comfy-env] Materializing `{env_name}` via `pixi install` "
                f"(this can take several minutes on first run; CUDA wheels "
                f"are heavy)..."
            )
            ok = _run_pixi_install(env_manifest_dir, env_manifest, log)
            if not ok:
                return None
            log(f"[comfy-env] `{env_name}` materialized.")
    except RuntimeError as e:
        log(f"[comfy-env] auto-install of `{env_name}` aborted: {e}")
        return None

    # Defensive re-check: pixi succeeded but did the dir actually appear?
    env_dir = get_workspace_env_dir(comfyui_dir, env_name)
    return env_dir if env_dir.is_dir() else None


def _generate_env_manifest(
    env_name: str,
    config_path: Path,
    env_manifest_dir: Path,
    log: Callable[[str], None],
) -> None:
    """Generate the per-env pixi.toml from the node's comfy-env.toml.

    Cuda-wheel combo resolution is intentionally skipped here -- that's a
    workspace-wide decision that needs every env's configs together. Per-env
    auto-install pins to bootstrap torch only. Nodes with cuda-only wheel
    deps (flash_attn, sageattention, cumesh) will need explicit
    ``comfy-env install`` to provision them.
    """
    from ..config import load_config
    from ..packages.toml_generator import write_env_pixi_toml
    from ..install.workspace import _resolve_workspace_torch

    log(f"[comfy-env] Generating manifest for `{env_name}` at {env_manifest_dir}/pixi.toml")
    cfg = load_config(config_path)
    (
        torch_index, _cuda_version, _cuda_major, bootstrap_python, bootstrap_torch,
    ) = _resolve_workspace_torch(log)
    torch_pin = f"=={bootstrap_torch}" if bootstrap_torch else None
    write_env_pixi_toml(
        env_manifest_dir=env_manifest_dir,
        env_name=env_name,
        cfg=cfg,
        torch_index=torch_index,
        bootstrap_python=bootstrap_python,
        torch_pin=torch_pin,
        log=log,
    )


def _run_pixi_install(
    env_manifest_dir: Path,
    env_manifest: Path,
    log: Callable[[str], None],
) -> bool:
    """Run ``pixi install --manifest-path <env>/pixi.toml`` synchronously.

    Returns True on success, False on failure (with last 15 lines of stderr
    logged). Captures stdout silently -- pixi's progress output is verbose
    and would flood ComfyUI's startup log. The "Materializing..." line
    logged by the caller is the user-facing progress signal.
    """
    import subprocess
    from ..packages.pixi import PIXI, ensure_pixi

    ensure_pixi()
    pixi_env = dict(os.environ)
    pixi_env["PIXI_NO_PROGRESS"] = "true"
    result = subprocess.run(
        [PIXI, "install", "--manifest-path", str(env_manifest)],
        cwd=str(env_manifest_dir),
        capture_output=True,
        text=True,
        env=pixi_env,
    )
    if result.returncode != 0:
        log(
            f"[comfy-env] auto-install FAILED for "
            f"{env_manifest_dir.name} (exit {result.returncode}):"
        )
        for line in (result.stderr or "").splitlines()[-15:]:
            log(f"  {line}")
        return False
    return True
