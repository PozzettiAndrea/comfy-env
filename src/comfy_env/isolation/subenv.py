"""Isolation subprocess launch-environment construction.

A leaf module: stdlib + comfy_env.debug only, imported DOWNWARD by
subprocess.py and metadata.py (which spawn workers) and by wrap.py (env
paths). It was previously inlined at the top of wrap.py, which forced
subprocess/metadata to reach UP into the orchestrator for it -- a
lazy-import cycle. Nothing here depends on the worker pool, node
registration, or metadata; it is pure launch-env setup.
"""

import glob
import os
import sys
from pathlib import Path
from typing import Optional

from ..debug import INSTALL as _DBG_INSTALL


def _log(msg: str) -> None:
    """Print to stderr with flush -- survives process crashes."""
    print(msg, file=sys.stderr, flush=True)


def _build_isolation_env_win32(env: dict, python: Path) -> dict:
    """Windows: minimal PATH with env + Library/bin + system dirs."""
    env_root = python.parent
    library_bin = env_root / "Library" / "bin"
    windir = os.environ.get("WINDIR", r"C:\Windows")
    minimal_path_parts = [
        str(env_root),
        str(env_root / "Scripts"),
        str(env_root / "Lib" / "site-packages" / "bpy"),
        f"{windir}\\System32",
        f"{windir}",
        f"{windir}\\System32\\Wbem",
    ]
    if library_bin.is_dir():
        minimal_path_parts.insert(1, str(library_bin))
        if _DBG_INSTALL:
            dll_count = len([f for f in library_bin.iterdir() if f.suffix.lower() == ".dll"])
            _log(f"[comfy-env] {env_root.name}: Library/bin has {dll_count} DLLs")
    else:
        if _DBG_INSTALL:
            _log(f"[comfy-env] {env_root.name}: Library/bin NOT FOUND at {library_bin}")
    env["PATH"] = ";".join(minimal_path_parts)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    # Pixi/conda envs on Windows: the Python binary resolves sys.prefix to the
    # base UV/conda Python instead of the env, causing both stdlib version
    # mismatches (SRE module mismatch) and missing site-packages (CGAL).
    # PYTHONHOME forces Python to use the env's own stdlib and site-packages.
    if (env_root / "Lib").is_dir():
        env["PYTHONHOME"] = str(env_root)
    return env


def _build_isolation_env_darwin(env: dict, python: Path) -> dict:
    """macOS: add env's lib dir to DYLD_FALLBACK_LIBRARY_PATH.

    We use FALLBACK rather than DYLD_LIBRARY_PATH so absolute-path-linked
    system libs (e.g. /usr/lib/libiconv.2.dylib used by bpy) keep resolving
    to /usr/lib instead of being shadowed by conda-forge replicas inside the
    pixi env. dyld only consults the fallback path when the explicit lookup
    fails, which is the right behavior for a conda-style env on macOS.

    KMP_DUPLICATE_LIB_OK=TRUE: the parent venv often ships pip-built libs
    with their own bundled libomp.dylib (cv2, scipy, etc.), and the pixi env
    has conda-forge's libomp. When meta-scan or worker imports pull in both,
    libomp aborts (OMP: Error #15). KMP_DUPLICATE_LIB_OK is the official
    Intel-OpenMP escape hatch and is already set on win32.
    """
    lib_dir = python.parent.parent / "lib"
    if lib_dir.is_dir():
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _build_isolation_env_linux(env: dict, python: Path) -> dict:
    """Linux: add env's lib dir + system libs to LD_LIBRARY_PATH."""
    lib_dir = python.parent.parent / "lib"
    if lib_dir.is_dir():
        existing = env.get("LD_LIBRARY_PATH", "")
        system_libs = "/usr/lib/x86_64-linux-gnu:/usr/lib:/lib/x86_64-linux-gnu"
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{system_libs}:{existing}" if existing else f"{lib_dir}:{system_libs}"
    return env


def build_isolation_env(python: Path, env_vars: dict = None) -> dict:
    """Build environment dict for isolation subprocess. Dispatches to platform-specific builder."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    env["COMFYUI_ISOLATION_WORKER"] = "1"

    # Scrub Python-pathing vars inherited from the parent (python_embeded on
    # portable). PYTHONPATH would let the env's python import from the
    # parent's site-packages -- same ABI tag but a DIFFERENT torch wheel -- so
    # the DLL graph straddles two torches: ERROR_PROC_NOT_FOUND on shm.dll.
    # PYTHONHOME would point it at the parent's stdlib outright.
    # PYTHONNOUSERSITE keeps ~/.local and %APPDATA%\Python out.
    # PYTHONSTARTUP can side-load arbitrary code.
    #
    # Platform-independent, though it lived in the win32 builder alone until
    # 0.4.30 -- so `pip install --user torch`, the documented PEP-668
    # workaround, silently shadowed the pinned torch in every posix worker.
    # win32 re-sets PYTHONHOME below for its own sys.prefix reasons.
    for var in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONHOME"):
        env.pop(var, None)
    env["PYTHONNOUSERSITE"] = "1"

    if sys.platform == "win32":
        return _build_isolation_env_win32(env, python)
    elif sys.platform == "darwin":
        return _build_isolation_env_darwin(env, python)
    else:
        return _build_isolation_env_linux(env, python)


def _get_env_paths(env_dir: Path) -> "Optional[Path]":
    """Get site_packages from env.

    Used to return (site_packages, lib_dir) too. The lib_dir was threaded
    through seven functions in three files and then dropped: _create_worker
    accepted it and built SubprocessWorker without it.
    """
    if sys.platform == "win32":
        sp = env_dir / "Lib" / "site-packages"
    else:
        matches = glob.glob(str(env_dir / "lib/python*/site-packages"))
        sp = Path(matches[0]) if matches else None
    return sp if sp and sp.exists() else None
