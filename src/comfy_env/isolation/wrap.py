"""Process isolation for ComfyUI nodes - wraps FUNCTION methods to run in isolated env."""

import glob
import inspect
import os
import shutil
import signal
import sys
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.types import DEFAULT_HEALTH_CHECK_TIMEOUT

_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")
_CLEANUP_DONE = False


def _log(msg: str) -> None:
    """Print to stderr with flush — survives process crashes."""
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
                return item
    except OSError:
        pass
    return None


def _find_package_root(start_dir: Path) -> Path:
    """Find package root for import resolution.

    If start_dir contains __init__.py (is a package), return its parent
    so the package itself is importable. Otherwise return start_dir.
    """
    start_dir = start_dir.resolve()
    if (start_dir / "__init__.py").exists():
        return start_dir.parent
    return start_dir


def _get_python_version(env_dir: Path) -> Optional[str]:
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists(): return None
    try:
        import subprocess
        r = subprocess.run([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                          capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception: return None


def _create_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                   lib_path: Optional[str] = None, env_vars: Optional[dict] = None,
                   health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
    """Create a fresh subprocess worker. Never reused - caller must shutdown after use."""
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


def _wrap_node_class(cls: type, env_dir: Path, working_dir: Path, sys_path: list[str],
                     lib_path: Optional[str] = None, env_vars: Optional[dict] = None,
                     health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT) -> type:
    func_name = getattr(cls, "FUNCTION", None)
    if not func_name: return cls
    original = getattr(cls, func_name, None)
    if not original: return cls

    try:
        source = Path(inspect.getfile(cls)).resolve()
        module_name = str(source.relative_to(working_dir).with_suffix("")).replace("/", ".").replace("\\", ".")
    except (TypeError, OSError, ValueError):
        module_name = source.stem if 'source' in dir() else cls.__module__

    @wraps(original)
    def proxy(self, **kwargs):
        # Create fresh worker for each call - never reuse to avoid stale state
        worker = _create_worker(env_dir, working_dir, sys_path, lib_path, env_vars, health_check_timeout)
        try:
            # Prepare tensors for IPC
            try:
                from .tensor_utils import prepare_for_ipc_recursive
                kwargs = {k: prepare_for_ipc_recursive(v) for k, v in kwargs.items()}
            except ImportError: pass

            result = worker.call_method(
                module_name=module_name, class_name=cls.__name__, method_name=func_name,
                self_state=self.__dict__.copy() if hasattr(self, "__dict__") else None,
                kwargs=kwargs, timeout=600.0,
            )

            try:
                from .tensor_utils import prepare_for_ipc_recursive
                result = prepare_for_ipc_recursive(result)
            except ImportError: pass
            return result
        finally:
            # Always shutdown worker after use
            worker.shutdown()

    setattr(cls, func_name, proxy)
    cls._comfy_env_isolated = True
    return cls


def wrap_nodes() -> None:
    """Auto-wrap nodes for isolation. Call from __init__.py after NODE_CLASS_MAPPINGS."""
    # Log version for debugging
    try:
        from importlib.metadata import version as get_version
        print(f"[comfy-env] Version: {get_version('comfy-env')}")
    except Exception:
        pass

    _cleanup_stale_workers()

    if not _is_enabled() or os.environ.get("COMFYUI_ISOLATION_WORKER") == "1":
        return

    frame = inspect.stack()[1]
    caller_module = inspect.getmodule(frame.frame)
    if not caller_module: return

    mappings = getattr(caller_module, "NODE_CLASS_MAPPINGS", None)
    if not mappings: return

    pkg_dir = Path(frame.filename).resolve().parent
    config_files = list(pkg_dir.rglob("comfy-env.toml"))
    if not config_files: return

    try:
        import folder_paths
        comfyui_base = folder_paths.base_path
    except ImportError:
        comfyui_base = None

    envs = []
    for cf in config_files:
        env_dir = _find_env_dir(cf.parent)
        sp, lib = _get_env_paths(env_dir) if env_dir else (None, None)
        if not env_dir or not sp: continue

        env_vars = {}
        health_check_timeout = DEFAULT_HEALTH_CHECK_TIMEOUT
        try:
            import tomli
            with open(cf, "rb") as f:
                toml_data = tomli.load(f)
                env_vars = {str(k): str(v) for k, v in toml_data.get("env_vars", {}).items()}
                health_check_timeout = float(toml_data.get("options", {}).get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))
                print(f"[comfy-env] Parsed {cf}: health_check_timeout={health_check_timeout}")
        except Exception as e:
            print(f"[comfy-env] Failed to parse {cf}: {e}")
        if comfyui_base: env_vars["COMFYUI_BASE"] = str(comfyui_base)

        envs.append({"dir": cf.parent, "env_dir": env_dir, "sp": sp, "lib": lib, "env_vars": env_vars, "health_check_timeout": health_check_timeout})

    wrapped = 0
    for name, cls in mappings.items():
        if not hasattr(cls, "FUNCTION"): continue
        try:
            src = Path(inspect.getfile(cls)).resolve()
        except (TypeError, OSError): continue

        for e in envs:
            try:
                src.relative_to(e["dir"])
                # Find package root by walking up until no __init__.py
                package_root = _find_package_root(e["dir"])
                _wrap_node_class(cls, e["env_dir"], package_root, [str(e["sp"]), str(package_root)],
                               str(e["lib"]) if e["lib"] else None, e["env_vars"], e.get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))
                wrapped += 1
                break
            except ValueError: continue

    if wrapped: print(f"[comfy-env] Wrapped {wrapped} nodes")


def wrap_isolated_nodes(node_class_mappings: Dict[str, type], nodes_dir: Path) -> Dict[str, type]:
    """Wrap nodes from a directory with comfy-env.toml for isolation."""
    _cleanup_stale_workers()

    if not _is_enabled() or os.environ.get("COMFYUI_ISOLATION_WORKER") == "1":
        return node_class_mappings

    try:
        import folder_paths
        comfyui_base = folder_paths.base_path
    except ImportError:
        comfyui_base = None

    nodes_dir = Path(nodes_dir).resolve()
    config = nodes_dir / "comfy-env.toml"
    if not config.exists():
        print(f"[comfy-env] No comfy-env.toml in {nodes_dir}")
        return node_class_mappings

    env_vars = {}
    health_check_timeout = DEFAULT_HEALTH_CHECK_TIMEOUT
    try:
        import tomli
        with open(config, "rb") as f:
            toml_data = tomli.load(f)
            env_vars = {str(k): str(v) for k, v in toml_data.get("env_vars", {}).items()}
            health_check_timeout = float(toml_data.get("options", {}).get("health_check_timeout", DEFAULT_HEALTH_CHECK_TIMEOUT))
            print(f"[comfy-env] Parsed {config}: health_check_timeout={health_check_timeout}")
    except Exception as e:
        print(f"[comfy-env] Failed to parse {config}: {e}")
    if comfyui_base: env_vars["COMFYUI_BASE"] = str(comfyui_base)

    env_dir = _find_env_dir(nodes_dir)
    sp, lib = _get_env_paths(env_dir) if env_dir else (None, None)
    if not env_dir or not sp:
        print(f"[comfy-env] No env found. Run 'comfy-env install' in {nodes_dir}")
        return node_class_mappings

    sys_path = [str(sp), str(nodes_dir)]
    lib_path = str(lib) if lib else None

    print(f"[comfy-env] Wrapping {len(node_class_mappings)} nodes from {nodes_dir.name}")
    for cls in node_class_mappings.values():
        if hasattr(cls, "FUNCTION"):
            _wrap_node_class(cls, env_dir, nodes_dir, sys_path, lib_path, env_vars, health_check_timeout)

    return node_class_mappings


def register_nodes(nodes_package: str = "nodes") -> tuple:
    """Discover and register all nodes — main-process and isolation.

    Replaces the old pattern of:
        from .nodes import NODE_CLASS_MAPPINGS
        wrap_nodes()

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

        package_root = pkg_dir
        isolation_envs[cf.parent.resolve()] = {
            "dir": cf.parent,
            "env_dir": env_dir,
            "sp": sp,
            "lib": lib,
            "env_vars": env_vars,
            "health_check_timeout": health_check_timeout,
            "package_root": package_root,
        }

    if _DEBUG:
        _log(f"[comfy-env] Found {len(isolation_envs)} isolation env(s)")

    all_mappings = {}
    all_display = {}
    enabled = _is_enabled() and os.environ.get("COMFYUI_ISOLATION_WORKER") != "1"

    # Walk node subdirectories
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
                _log(f"[comfy-env] No env for {cf.parent.name} — run 'comfy-env install'")

    _log(f"[comfy-env] Registered {len(all_mappings)} total nodes "
         f"({len(main_dirs)} main + {len(isolation_dirs)} isolation dirs)")

    return all_mappings, all_display
