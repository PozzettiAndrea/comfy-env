"""Process isolation for ComfyUI nodes - wraps FUNCTION methods to run in isolated env."""

import atexit
import glob
import inspect
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.types import DEFAULT_HEALTH_CHECK_TIMEOUT

_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")
_CLEANUP_DONE = False

# ---------------------------------------------------------------------------
# Persistent worker pool — one worker per isolation env, reused across calls.
# Workers auto-restart on crash (native segfault, etc.).
# ---------------------------------------------------------------------------
_WORKER_POOL: Dict[str, Any] = {}  # str(env_dir) -> (SubprocessWorker, generation)
_WORKER_PATCHERS: Dict[str, Dict[str, Any]] = {}  # str(env_dir) -> {model_id: SubprocessModelPatcher}
_POOL_LOCK = threading.Lock()
_WORKER_GENERATION = 0  # Monotonically increasing; incremented on each new worker


def _log(msg: str) -> None:
    """Print to stderr with flush -- survives process crashes."""
    print(msg, file=sys.stderr, flush=True)


def _cleanup_stale_workers():
    """Kill orphaned worker processes and remove stale temp directories on startup.

    Only kills workers whose parent process no longer exists - safe for multiple
    ComfyUI instances running on the same machine.
    """
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True

    temp_dir = tempfile.gettempdir()

    # Find stale socket files
    socket_patterns = [
        "/dev/shm/comfy_worker_*.sock",  # Linux shared memory
        os.path.join(temp_dir, "comfy_worker_*.sock"),  # Fallback
    ]

    for pattern in socket_patterns:
        for sock_file in glob.glob(pattern):
            try:
                os.unlink(sock_file)
                print(f"[comfy-env] Removed stale socket: {sock_file}")
            except Exception:
                pass

    # Kill only ORPHANED worker processes (parent PID no longer exists)
    # This is safe for multiple ComfyUI instances on same machine
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline') or []
                if any('persistent_worker.py' in arg for arg in cmdline):
                    parent_pid = proc.info.get('ppid')
                    # Check if parent process still exists
                    if parent_pid and not psutil.pid_exists(parent_pid):
                        print(f"[comfy-env] Killing orphaned worker (parent {parent_pid} dead): {proc.pid}")
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # psutil not available - use /proc on Linux
        if sys.platform == 'linux':
            for pid_dir in glob.glob('/proc/[0-9]*'):
                try:
                    pid = int(os.path.basename(pid_dir))
                    with open(f'{pid_dir}/cmdline', 'rb') as f:
                        cmdline = f.read().decode('utf-8', errors='ignore')
                    if 'persistent_worker.py' in cmdline:
                        with open(f'{pid_dir}/stat', 'r') as f:
                            stat = f.read().split()
                            ppid = int(stat[3])
                        # Check if parent exists
                        if not os.path.exists(f'/proc/{ppid}'):
                            print(f"[comfy-env] Killing orphaned worker (parent {ppid} dead): {pid}")
                            os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

    # Find and remove stale temp directories
    stale_dirs = glob.glob(os.path.join(temp_dir, "comfyui_pvenv_*"))
    for stale_dir in stale_dirs:
        try:
            # Check if any process is using this directory
            dir_in_use = False
            try:
                import psutil
                for proc in psutil.process_iter(['pid', 'cmdline', 'cwd']):
                    try:
                        cwd = proc.info.get('cwd') or ''
                        cmdline = ' '.join(proc.info.get('cmdline') or [])
                        if stale_dir in cwd or stale_dir in cmdline:
                            dir_in_use = True
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except ImportError:
                # No psutil - check if worker script exists and is recent (< 1 hour)
                worker_script = os.path.join(stale_dir, 'persistent_worker.py')
                if os.path.exists(worker_script):
                    age_hours = (time.time() - os.path.getmtime(worker_script)) / 3600
                    if age_hours < 1:
                        dir_in_use = True  # Assume in use if recent

            if not dir_in_use:
                shutil.rmtree(stale_dir)
                print(f"[comfy-env] Removed stale temp dir: {stale_dir}")
        except Exception:
            pass


def _is_enabled() -> bool:
    return os.environ.get("USE_COMFY_ENV", "1").lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------------------
# Isolation environment setup (shared by metadata scan + SubprocessWorker)
# ---------------------------------------------------------------------------

def _build_isolation_env_win32(env: dict, python: Path) -> dict:
    """Windows: minimal PATH with env + Library/bin + system dirs."""
    env["COMFYUI_HOST_PYTHON_DIR"] = str(Path(sys.executable).parent)
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
        if _DEBUG:
            dll_count = len([f for f in library_bin.iterdir() if f.suffix.lower() == ".dll"])
            _log(f"[comfy-env] {env_root.name}: Library/bin has {dll_count} DLLs")
    else:
        if _DEBUG:
            _log(f"[comfy-env] {env_root.name}: Library/bin NOT FOUND at {library_bin}")
    env["PATH"] = ";".join(minimal_path_parts)
    env["COMFYUI_PIXI_LIBRARY_BIN"] = str(library_bin) if library_bin.is_dir() else ""
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _build_isolation_env_darwin(env: dict, python: Path) -> dict:
    """macOS: add env's lib dir to DYLD_LIBRARY_PATH."""
    lib_dir = python.parent.parent / "lib"
    if lib_dir.is_dir():
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)
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

    if sys.platform == "win32":
        return _build_isolation_env_win32(env, python)
    elif sys.platform == "darwin":
        return _build_isolation_env_darwin(env, python)
    else:
        return _build_isolation_env_linux(env, python)


def _get_env_paths(env_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Get (site_packages, lib_dir) from env."""
    if sys.platform == "win32":
        sp = env_dir / "Lib" / "site-packages"
        lib = env_dir / "Library" / "bin"
    else:
        matches = glob.glob(str(env_dir / "lib/python*/site-packages"))
        sp = Path(matches[0]) if matches else None
        lib = env_dir / "lib"
    return (sp, lib) if sp and sp.exists() else (None, None)


def _find_env_dir(node_dir: Path) -> Optional[Path]:
    """Find _env_* directory in node_dir."""
    try:
        for item in node_dir.iterdir():
            if item.name.startswith("_env_") and item.is_dir():
                # On Windows, resolve junctions to keep paths under MAX_PATH for LoadLibrary
                if sys.platform == "win32" and item.is_junction():
                    return item.resolve()
                return item
    except OSError:
        pass
    return None


def _get_python_version(env_dir: Path) -> Optional[str]:
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists(): return None
    try:
        import subprocess
        r = subprocess.run([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception: return None


def _get_host_torch_sp() -> Optional[Path]:
    """Get the host's site-packages directory containing torch, or None if torch isn't imported."""
    try:
        import torch
        return Path(torch.__file__).parent.parent
    except (ImportError, AttributeError):
        return None


def _should_share_torch(env_dir: Path) -> bool:
    """Determine if host torch should be shared with this worker env.

    Shares automatically when the host has torch and the worker's Python
    major.minor matches the host's.  Torch is a C extension — sharing across
    Python minor versions would crash, so there's no manual override.
    """
    host_torch_sp = _get_host_torch_sp()
    if host_torch_sp is None:
        return False

    host_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    worker_version = _get_python_version(env_dir)
    if worker_version is None:
        return False

    match = host_version == worker_version
    if _DEBUG:
        _log(f"[comfy-env] share_torch: host={host_version}, worker={worker_version}, match={match}")
    return match


def _create_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                   lib_path: Optional[str] = None, env_vars: Optional[dict] = None,
                   health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
    """Create a fresh subprocess worker."""
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    from .workers.subprocess import SubprocessWorker
    if _DEBUG:
        print(f"[comfy-env] SubprocessWorker: {python}")
        if env_vars:
            print(f"[comfy-env] env_vars: {env_vars}")
    return SubprocessWorker(
        python=str(python), working_dir=working_dir, sys_path=sys_path,
        name=working_dir.name, env=env_vars, health_check_timeout=health_check_timeout
    )


def _handle_progress(request: dict) -> dict:
    """Parent-side callback: forward subprocess progress to ComfyUI frontend.

    Also checks if the user clicked cancel — if so, returns an error
    so the subprocess can stop processing.
    """
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except Exception:
        return {"status": "error", "error": "Processing interrupted by user"}
    try:
        import comfy.utils
        if comfy.utils.PROGRESS_BAR_HOOK:
            value = request.get("value", 0)
            total = request.get("total", 1)
            comfy.utils.PROGRESS_BAR_HOOK(value, total, None)
    except Exception:
        pass
    return {}


def _handle_vram_budget(request: dict) -> dict:
    """Parent-side callback: free VRAM for subprocess model loading.

    Called when the worker's shimmed load_models_gpu() needs to load models.
    The parent evicts its own models (via free_memory) so the subprocess's
    real load_models_gpu() sees enough free VRAM via get_free_memory().
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return {"device": "cuda"}

    total_requested = request.get("total_size", 0)
    device = mm.get_torch_device()

    if _DEBUG:
        free_before = mm.get_free_memory(device)
        _log(f"[comfy-env] VRAM request: {total_requested / 1e9:.2f}GB, "
             f"free before eviction: {free_before / 1e9:.2f}GB")

    # Evict parent-side models to make room (with 10% headroom)
    mm.free_memory(total_requested * 1.1, device)

    if _DEBUG:
        free_after = mm.get_free_memory(device)
        _log(f"[comfy-env] VRAM after eviction: {free_after / 1e9:.2f}GB")

    return {
        "device": str(device),
        "extra_reserved_vram": mm.EXTRA_RESERVED_VRAM,
        "vram_state": mm.vram_state.name,
    }


def _cleanup_stale_patchers(env_dir):
    """Mark stale SubprocessModelPatchers for cleanup.

    Called when a worker is replaced (crash/restart).  We clear the patcher
    registry so they won't be re-registered.  The patchers themselves stay in
    ComfyUI's current_loaded_models — the safety net in _send_device_command
    handles "not registered" IPC errors gracefully, and free_memory will
    remove them during its normal unload loop.

    We must NOT modify current_loaded_models here because this callback can
    fire inside free_memory's iteration (via model_unload → send_command →
    _ensure_started → _on_restart), which would invalidate captured indices.
    """
    key = str(env_dir)
    old_patchers = _WORKER_PATCHERS.pop(key, None)
    if not old_patchers:
        return
    _log(f"[comfy-env] Invalidated {len(old_patchers)} stale model patchers "
         f"(will be cleaned up during next unload)")


def _get_or_create_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                          lib_path: Optional[str] = None, env_vars: Optional[dict] = None,
                          health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
    """Get existing worker for this env, or create a new one.

    Returns (worker, generation) tuple.  The generation is a monotonically
    increasing integer used to detect stale ModelPatchers after worker restart.
    """
    global _WORKER_GENERATION
    key = str(env_dir)
    with _POOL_LOCK:
        entry = _WORKER_POOL.get(key)
        if entry is not None:
            worker, gen = entry
            if worker.is_alive():
                return worker, gen
            # Dead — clean up stale patchers before replacing worker
            _cleanup_stale_patchers(env_dir)
            try:
                worker.shutdown()
            except Exception:
                pass
        _WORKER_GENERATION += 1
        gen = _WORKER_GENERATION
        worker = _create_worker(env_dir, working_dir, sys_path, lib_path, env_vars, health_check_timeout)
        # Register bidirectional RPC callbacks
        worker.register_callback("request_vram_budget", _handle_vram_budget)
        worker.register_callback("report_progress", _handle_progress)
        # Clean up stale patchers if worker restarts transparently via _ensure_started()
        worker._on_restart = lambda: _cleanup_stale_patchers(env_dir)
        _WORKER_POOL[key] = (worker, gen)
        return worker, gen


def _remove_worker(env_dir):
    """Remove a dead worker from the pool (called after crash)."""
    key = str(env_dir)
    with _POOL_LOCK:
        entry = _WORKER_POOL.pop(key, None)
        _WORKER_PATCHERS.pop(key, None)
        if entry is not None:
            worker, _ = entry
            try:
                worker.shutdown()
            except Exception:
                pass


def _shutdown_all_workers():
    """Shut down all persistent workers. Called via atexit."""
    with _POOL_LOCK:
        for key, (worker, _gen) in list(_WORKER_POOL.items()):
            try:
                worker.shutdown()
            except Exception:
                pass
        _WORKER_POOL.clear()
        _WORKER_PATCHERS.clear()


atexit.register(_shutdown_all_workers)


def _register_new_patchers(env_dir, worker, generation):
    """Create SubprocessModelPatchers for any models auto-detected during the last call.

    Called after each call_method.  The worker's Module.to()/cuda() hooks
    auto-register nn.Modules that land on CUDA; the worker returns their
    metadata in response['_new_models'].  We create patchers here and register
    them with ComfyUI's memory manager so they participate in VRAM eviction.
    """
    new_models = getattr(worker, '_last_new_models', [])
    if not new_models:
        return

    from .model_patcher import SubprocessModelPatcher

    try:
        import comfy.model_management
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
    except Exception:
        return

    key = str(env_dir)
    patchers = _WORKER_PATCHERS.setdefault(key, {})

    created = []
    for ref in new_models:
        model_id = ref["id"]
        if model_id in patchers:
            continue  # Already tracked
        patcher = SubprocessModelPatcher(
            worker=worker,
            worker_generation=generation,
            model_id=model_id,
            model_size=ref["size"],
            load_device=load_device,
            offload_device=offload_device,
            kind=ref.get("kind", "other"),
        )
        # Mark as already loaded (the model is on GPU right now)
        patcher.model.device = load_device
        patcher.model.model_loaded_weight_memory = ref["size"]
        patchers[model_id] = patcher
        created.append(model_id)

    if created:
        if _DEBUG:
            _log(f"[comfy-env] Created {len(created)} model patchers: {created}")
        # Register with ComfyUI memory manager (models are already on GPU)
        comfy.model_management.load_models_gpu(list(patchers.values()))


def _load_worker_models(env_dir):
    """Ensure all tracked models for this worker are on GPU before a call.

    Called before each call_method.  If ComfyUI evicted models between calls,
    load_models_gpu will send IPC commands to move them back.
    """
    key = str(env_dir)
    patchers = _WORKER_PATCHERS.get(key)
    if not patchers:
        return

    try:
        import comfy.model_management
        comfy.model_management.load_models_gpu(list(patchers.values()))
    except Exception:
        pass


def register_nodes(nodes_package: str = "nodes") -> tuple:
    """Discover and register all nodes -- main-process and isolation.

    Usage in custom node __init__.py:
        from comfy_env import register_nodes
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

    For main-process dirs (no comfy-env.toml): imports normally.
    For isolation dirs (comfy-env.toml + _env_*): subprocess metadata scan + proxy classes.

    Args:
        nodes_package: Name of the nodes subpackage (default: "nodes")

    Returns:
        (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
    """
    import importlib
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .metadata import fetch_metadata, build_proxy_class

    # Log version
    try:
        from importlib.metadata import version as get_version
        _log(f"[comfy-env] Version: {get_version('comfy-env')}")
    except Exception:
        pass

    _cleanup_stale_workers()

    # Get caller info
    frame = inspect.stack()[1]
    caller_module = inspect.getmodule(frame.frame)
    pkg_dir = Path(frame.filename).resolve().parent
    caller_pkg_name = caller_module.__name__ if caller_module else None

    if _DEBUG:
        _log(f"[comfy-env] register_nodes: pkg_dir={pkg_dir}, caller={caller_pkg_name}")

    nodes_dir = pkg_dir / nodes_package
    if not nodes_dir.is_dir():
        _log(f"[comfy-env] No '{nodes_package}/' directory in {pkg_dir}")
        return {}, {}

    # Discover isolation configs
    isolation_envs = {}  # {resolved_dir: env_config}
    config_files = list(pkg_dir.rglob("comfy-env.toml"))

    try:
        import folder_paths
        comfyui_base = folder_paths.base_path
    except ImportError:
        comfyui_base = None

    for cf in config_files:
        if cf.name == "comfy-env-root.toml":
            continue
        env_dir = _find_env_dir(cf.parent)
        if not env_dir:
            continue
        sp, lib = _get_env_paths(env_dir)
        if not sp:
            continue

        env_vars = {}
        health_check_timeout = DEFAULT_HEALTH_CHECK_TIMEOUT
        try:
            import tomli
            with open(cf, "rb") as f:
                toml_data = tomli.load(f)
                env_vars = {str(k): str(v) for k, v in toml_data.get("env_vars", {}).items()}
                health_check_timeout = float(toml_data.get("options", {}).get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))
        except Exception as e:
            _log(f"[comfy-env] Failed to parse {cf}: {e}")
        if comfyui_base:
            env_vars["COMFYUI_BASE"] = str(comfyui_base)

        # Determine if this env should inherit host torch
        share_torch_active = _should_share_torch(env_dir)
        host_torch_sp = _get_host_torch_sp() if share_torch_active else None

        if share_torch_active and host_torch_sp:
            _log(f"[comfy-env] {cf.parent.name}: sharing host torch from {host_torch_sp}")

        package_root = pkg_dir
        isolation_envs[cf.parent.resolve()] = {
            "dir": cf.parent,
            "env_dir": env_dir,
            "sp": sp,
            "lib": lib,
            "env_vars": env_vars,
            "health_check_timeout": health_check_timeout,
            "package_root": package_root,
            "share_torch": share_torch_active,
            "host_torch_sp": host_torch_sp,
        }

    if _DEBUG:
        _log(f"[comfy-env] Found {len(isolation_envs)} isolation env(s)")

    all_mappings = {}
    all_display = {}
    enabled = _is_enabled() and os.environ.get("COMFYUI_ISOLATION_WORKER") != "1"

    # ==================================================================
    # Discover and import node sources
    # ==================================================================
    # Two patterns (mutually exclusive):
    #   1. nodes/ itself is the source (isolation or direct)
    #   2. Subdirectories of nodes/ are individual sources
    # Check root first; fall through to subdirs if root yields nothing.

    root_resolved = nodes_dir.resolve()

    # --- Pattern 1: nodes/ root ---
    if root_resolved in isolation_envs and enabled:
        # Isolation env at root -- subprocess scan
        env = isolation_envs[root_resolved]
        _log(f"[comfy-env] Importing {nodes_package} (isolation root)...")
        try:
            root_meta = fetch_metadata(
                env_dir=env["env_dir"],
                node_dir=nodes_dir,
                package_name=nodes_package,
                working_dir=pkg_dir,
                env_vars=env["env_vars"],
            )
            root_nodes = root_meta.get("nodes", {})
            root_display = root_meta.get("display", {})

            package_root = env["package_root"]
            sys_path_list = [str(env["sp"]), str(package_root)]
            if env.get("share_torch") and env.get("host_torch_sp"):
                sys_path_list.insert(0, str(env["host_torch_sp"]))
            lib_path = str(env["lib"]) if env["lib"] else None

            for name, meta in root_nodes.items():
                all_mappings[name] = build_proxy_class(
                    node_name=name,
                    meta=meta,
                    env_dir=env["env_dir"],
                    package_root=package_root,
                    sys_path=sys_path_list,
                    lib_path=lib_path,
                    env_vars=env["env_vars"],
                    health_check_timeout=env["health_check_timeout"],
                )
            all_display.update(root_display)
            _log(f"[comfy-env] Imported {nodes_package} root: {len(root_nodes)} nodes (isolation)")
        except Exception as e:
            _log(f"[comfy-env] Failed to scan {nodes_package} root: {e}")

    elif root_resolved not in isolation_envs:
        # No isolation at root -- try direct import
        _log(f"[comfy-env] Importing {nodes_package} (root)...")
        try:
            mod = importlib.import_module(f".{nodes_package}", package=caller_pkg_name)
            mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
            display = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
            all_mappings.update(mappings)
            all_display.update(display)
            _log(f"[comfy-env] Imported {nodes_package} root: {len(mappings)} nodes")
        except Exception as e:
            _log(f"[comfy-env] Failed to import {nodes_package} root: {e}")

    # --- Pattern 2: subdirectories (only if root yielded nothing) ---
    # Skip if root was an isolation env (even if scan returned 0 nodes) -- subdirs
    # are part of that isolation env and must not be direct-imported.
    if not all_mappings and root_resolved not in isolation_envs:
        main_dirs = []
        isolation_dirs = []

        for subdir in sorted(nodes_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not (subdir / "__init__.py").exists():
                continue
            if subdir.name.startswith("_") or subdir.name.startswith("."):
                continue

            if subdir.resolve() in isolation_envs:
                isolation_dirs.append(subdir)
            else:
                main_dirs.append(subdir)

        # Import main-process dirs normally
        for subdir in main_dirs:
            module_path = f".{nodes_package}.{subdir.name}"
            _log(f"[comfy-env] Importing {subdir.name}...")
            try:
                mod = importlib.import_module(module_path, package=caller_pkg_name)
                mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
                display = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
                all_mappings.update(mappings)
                all_display.update(display)
                _log(f"[comfy-env] Imported {subdir.name}: {len(mappings)} nodes")
            except Exception as e:
                _log(f"[comfy-env] Failed to import {module_path}: {e}")

        # Subprocess-scan isolation dirs (in parallel)
        if enabled and isolation_dirs:
            def _scan_isolation(subdir):
                env = isolation_envs[subdir.resolve()]
                package_name = f"{nodes_package}.{subdir.name}"
                return subdir, env, fetch_metadata(
                    env_dir=env["env_dir"],
                    node_dir=subdir,
                    package_name=package_name,
                    working_dir=pkg_dir,
                    env_vars=env["env_vars"],
                )

            with ThreadPoolExecutor(max_workers=len(isolation_dirs)) as executor:
                futures = {executor.submit(_scan_isolation, d): d for d in isolation_dirs}
                for future in as_completed(futures):
                    try:
                        subdir, env, metadata = future.result()
                    except Exception as e:
                        subdir = futures[future]
                        _log(f"[comfy-env] Metadata scan failed for {subdir.name}: {e}")
                        continue

                    nodes_meta = metadata.get("nodes", {})
                    display = metadata.get("display", {})

                    package_root = env["package_root"]
                    sys_path_list = [str(env["sp"]), str(package_root)]
                    if env.get("share_torch") and env.get("host_torch_sp"):
                        sys_path_list.insert(0, str(env["host_torch_sp"]))
                    lib_path = str(env["lib"]) if env["lib"] else None

                    for name, meta in nodes_meta.items():
                        all_mappings[name] = build_proxy_class(
                            node_name=name,
                            meta=meta,
                            env_dir=env["env_dir"],
                            package_root=package_root,
                            sys_path=sys_path_list,
                            lib_path=lib_path,
                            env_vars=env["env_vars"],
                            health_check_timeout=env["health_check_timeout"],
                        )

                    all_display.update(display)
                    if nodes_meta:
                        _log(f"[comfy-env] Registered {len(nodes_meta)} isolation nodes from {subdir.name}")

        elif isolation_dirs and not enabled:
            if _DEBUG:
                _log(f"[comfy-env] Isolation disabled, skipping {len(isolation_dirs)} dirs")

    # Report skipped isolation dirs (no _env_* installed)
    for cf in config_files:
        if cf.name == "comfy-env-root.toml":
            continue
        if cf.parent.resolve() not in isolation_envs:
            env_dir = _find_env_dir(cf.parent)
            if not env_dir:
                _log(f"[comfy-env] No env for {cf.parent.name} -- run 'comfy-env install'")

    _log(f"[comfy-env] Registered {len(all_mappings)} total nodes")

    return all_mappings, all_display
