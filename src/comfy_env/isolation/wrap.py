"""Process isolation for ComfyUI nodes - wraps FUNCTION methods to run in isolated env."""

import atexit
import glob
import inspect
import os
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")
_workers: Dict[str, Any] = {}
_workers_lock = threading.Lock()


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
    """Find env dir: marker -> _env_<name> -> .pixi -> .venv"""
    marker = node_dir / ".comfy-env-marker.toml"
    if marker.exists():
        try:
            import tomli
            with open(marker, "rb") as f:
                env_path = tomli.load(f).get("env", {}).get("path")
            if env_path and Path(env_path).exists():
                return Path(env_path)
        except Exception: pass

    for candidate in [node_dir / _env_name(node_dir.name),
                     node_dir / ".pixi/envs/default",
                     node_dir / ".venv"]:
        if candidate.exists(): return candidate
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


def _get_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                lib_path: Optional[str] = None, env_vars: Optional[dict] = None):
    cache_key = str(env_dir)
    with _workers_lock:
        if cache_key in _workers and _workers[cache_key].is_alive():
            return _workers[cache_key]

        host_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        iso_ver = _get_python_version(env_dir)
        python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")

        if iso_ver and iso_ver != host_ver:
            # Different Python version - must use SubprocessWorker
            from .workers.subprocess import SubprocessWorker
            print(f"[comfy-env] SubprocessWorker: {python} ({iso_ver} vs {host_ver})")
            worker = SubprocessWorker(python=str(python), working_dir=working_dir, sys_path=sys_path, name=working_dir.name)
        else:
            # Same version - use MPWorker with venv Python for true isolation (like pyisolate)
            # This fixes Windows where spawn would otherwise re-import main.py
            from .workers.mp import MPWorker
            print(f"[comfy-env] MPWorker: {python}")
            worker = MPWorker(name=working_dir.name, sys_path=sys_path, lib_path=lib_path,
                            env_vars=env_vars, python=str(python))

        _workers[cache_key] = worker
        return worker


@atexit.register
def _shutdown_workers():
    with _workers_lock:
        for w in _workers.values():
            try: w.shutdown()
            except Exception: pass
        _workers.clear()


def _wrap_node_class(cls: type, env_dir: Path, working_dir: Path, sys_path: list[str],
                     lib_path: Optional[str] = None, env_vars: Optional[dict] = None) -> type:
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
        worker = _get_worker(env_dir, working_dir, sys_path, lib_path, env_vars)
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

    setattr(cls, func_name, proxy)
    cls._comfy_env_isolated = True
    return cls


def wrap_nodes() -> None:
    """Auto-wrap nodes for isolation. Call from __init__.py after NODE_CLASS_MAPPINGS."""
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
        try:
            import tomli
            with open(cf, "rb") as f:
                env_vars = {str(k): str(v) for k, v in tomli.load(f).get("env_vars", {}).items()}
        except Exception: pass
        if comfyui_base: env_vars["COMFYUI_BASE"] = str(comfyui_base)

        envs.append({"dir": cf.parent, "env_dir": env_dir, "sp": sp, "lib": lib, "env_vars": env_vars})

    wrapped = 0
    for name, cls in mappings.items():
        if not hasattr(cls, "FUNCTION"): continue
        try:
            src = Path(inspect.getfile(cls)).resolve()
        except (TypeError, OSError): continue

        for e in envs:
            try:
                src.relative_to(e["dir"])
                _wrap_node_class(cls, e["env_dir"], e["dir"], [str(e["sp"]), str(e["dir"])],
                               str(e["lib"]) if e["lib"] else None, e["env_vars"])
                wrapped += 1
                break
            except ValueError: continue

    if wrapped: print(f"[comfy-env] Wrapped {wrapped} nodes")


def wrap_isolated_nodes(node_class_mappings: Dict[str, type], nodes_dir: Path) -> Dict[str, type]:
    """Wrap nodes from a directory with comfy-env.toml for isolation."""
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
    try:
        import tomli
        with open(config, "rb") as f:
            env_vars = {str(k): str(v) for k, v in tomli.load(f).get("env_vars", {}).items()}
    except Exception: pass
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
            _wrap_node_class(cls, env_dir, nodes_dir, sys_path, lib_path, env_vars)

    return node_class_mappings
