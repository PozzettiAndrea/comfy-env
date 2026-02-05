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


def _env_name(dir_name: str) -> str:
    return f"_env_{dir_name.lower().replace('-', '_').lstrip('comfyui_')}"


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
    """Find env dir: junction (_*) -> _env_<name> -> .pixi -> .venv"""
    # Look for junction directories (start with _ and are symlinks)
    for item in node_dir.iterdir():
        if item.name.startswith("_") and item.is_dir() and item.resolve() != item:
            resolved = item.resolve()
            if resolved.exists():
                return resolved

    # Fallback to old patterns
    for candidate in [node_dir / _env_name(node_dir.name),
                     node_dir / ".pixi/envs/default",
                     node_dir / ".venv"]:
        if candidate.exists(): return candidate
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
