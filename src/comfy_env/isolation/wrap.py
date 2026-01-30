"""
Process isolation for ComfyUI node packs.

This module provides wrap_isolated_nodes() which wraps node classes
to run their FUNCTION methods in an isolated Python environment.

Usage:
    # In your node pack's __init__.py:
    from pathlib import Path
    from comfy_env import wrap_isolated_nodes

    NODE_CLASS_MAPPINGS = {}

    # Main nodes (no isolation)
    from .nodes.main import NODE_CLASS_MAPPINGS as main_nodes
    NODE_CLASS_MAPPINGS.update(main_nodes)

    # Isolated nodes (has comfy-env.toml in that directory)
    from .nodes.isolated import NODE_CLASS_MAPPINGS as isolated_nodes
    NODE_CLASS_MAPPINGS.update(
        wrap_isolated_nodes(isolated_nodes, Path(__file__).parent / "nodes/isolated")
    )
"""

import atexit
import inspect
import os
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

# Debug logging (set COMFY_ENV_DEBUG=1 to enable)
_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")


def get_env_name(dir_name: str) -> str:
    """Convert directory name to env name: ComfyUI-UniRig -> _env_unirig"""
    name = dir_name.lower().replace("-", "_").lstrip("comfyui_")
    return f"_env_{name}"

# Global worker cache (one per isolated environment)
_workers: Dict[str, Any] = {}
_workers_lock = threading.Lock()


def _get_isolated_python_version(env_dir: Path) -> Optional[str]:
    """Get Python version from isolated environment."""
    if sys.platform == "win32":
        python_path = env_dir / "python.exe"
    else:
        python_path = env_dir / "bin" / "python"

    if not python_path.exists():
        return None

    import subprocess
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_worker(
    env_dir: Path,
    working_dir: Path,
    sys_path: list[str],
    lib_path: Optional[str] = None,
    env_vars: Optional[dict] = None,
):
    """Get or create a persistent worker for the isolated environment."""
    cache_key = str(env_dir)

    with _workers_lock:
        if cache_key in _workers:
            worker = _workers[cache_key]
            if worker.is_alive():
                return worker
            # Worker died, will recreate

        # Check if Python versions match
        host_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        isolated_version = _get_isolated_python_version(env_dir)

        if isolated_version and isolated_version != host_version:
            # Different Python version - must use SubprocessWorker
            from ..workers.subprocess import SubprocessWorker

            if sys.platform == "win32":
                python_path = env_dir / "python.exe"
            else:
                python_path = env_dir / "bin" / "python"

            print(f"[comfy-env] Starting isolated worker (SubprocessWorker)")
            print(f"[comfy-env]   Python: {python_path} ({isolated_version} vs host {host_version})")

            worker = SubprocessWorker(
                python=str(python_path),
                working_dir=working_dir,
                sys_path=sys_path,
                name=working_dir.name,
            )
        else:
            # Same Python version - use MPWorker (faster)
            from ..workers.mp import MPWorker

            print(f"[comfy-env] Starting isolated worker (MPWorker)")
            print(f"[comfy-env]   Env: {env_dir}")
            if env_vars:
                print(f"[comfy-env]   env_vars: {', '.join(f'{k}={v}' for k, v in env_vars.items())}")

            worker = MPWorker(
                name=working_dir.name,
                sys_path=sys_path,
                lib_path=lib_path,
                env_vars=env_vars,
            )

        _workers[cache_key] = worker
        return worker


def _shutdown_workers():
    """Shutdown all cached workers. Called at exit."""
    with _workers_lock:
        for name, worker in _workers.items():
            try:
                worker.shutdown()
            except Exception:
                pass
        _workers.clear()


atexit.register(_shutdown_workers)


def _find_env_paths(node_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """
    Find site-packages and lib directories for the isolated environment.

    Returns:
        (site_packages, lib_dir) - lib_dir is for LD_LIBRARY_PATH
    """
    import glob

    # Check _env_<name> directory first (new pattern)
    env_name = get_env_name(node_dir.name)
    env_dir = node_dir / env_name
    if env_dir.exists():
        if sys.platform == "win32":
            site_packages = env_dir / "Lib" / "site-packages"
            lib_dir = env_dir / "Library" / "bin"
        else:
            pattern = str(env_dir / "lib" / "python*" / "site-packages")
            matches = glob.glob(pattern)
            site_packages = Path(matches[0]) if matches else None
            lib_dir = env_dir / "lib"
        if site_packages and site_packages.exists():
            return site_packages, lib_dir if lib_dir.exists() else None

    # Fallback: Check old .pixi/envs/default (for backward compat)
    pixi_env = node_dir / ".pixi" / "envs" / "default"
    if pixi_env.exists():
        if sys.platform == "win32":
            site_packages = pixi_env / "Lib" / "site-packages"
            lib_dir = pixi_env / "Library" / "bin"
        else:
            pattern = str(pixi_env / "lib" / "python*" / "site-packages")
            matches = glob.glob(pattern)
            site_packages = Path(matches[0]) if matches else None
            lib_dir = pixi_env / "lib"
        if site_packages and site_packages.exists():
            return site_packages, lib_dir if lib_dir.exists() else None

    # Check .venv directory
    venv_dir = node_dir / ".venv"
    if venv_dir.exists():
        if sys.platform == "win32":
            site_packages = venv_dir / "Lib" / "site-packages"
        else:
            pattern = str(venv_dir / "lib" / "python*" / "site-packages")
            matches = glob.glob(pattern)
            site_packages = Path(matches[0]) if matches else None
        if site_packages and site_packages.exists():
            return site_packages, None  # venvs don't have separate lib

    return None, None


def _find_env_dir(node_dir: Path) -> Optional[Path]:
    """Find the environment directory (for cache key)."""
    # Check _env_<name> first
    env_name = get_env_name(node_dir.name)
    env_dir = node_dir / env_name
    if env_dir.exists():
        return env_dir
    # Fallback to old paths
    pixi_env = node_dir / ".pixi" / "envs" / "default"
    if pixi_env.exists():
        return pixi_env
    venv_dir = node_dir / ".venv"
    if venv_dir.exists():
        return venv_dir
    return None


def _find_custom_node_root(nodes_dir: Path) -> Optional[Path]:
    """
    Find the custom node root (direct child of custom_nodes/).

    Uses folder_paths to find custom_nodes directories, then finds
    which one is an ancestor of nodes_dir.

    Example: /path/custom_nodes/ComfyUI-UniRig/nodes/nodes_gpu
             -> returns /path/custom_nodes/ComfyUI-UniRig
    """
    try:
        import folder_paths
        custom_nodes_dirs = folder_paths.get_folder_paths("custom_nodes")
    except (ImportError, KeyError):
        return None

    for cn_dir in custom_nodes_dirs:
        cn_path = Path(cn_dir)
        try:
            rel = nodes_dir.relative_to(cn_path)
            if rel.parts:
                return cn_path / rel.parts[0]
        except ValueError:
            continue

    return None


def _wrap_node_class(
    cls: type,
    env_dir: Path,
    working_dir: Path,
    sys_path: list[str],
    lib_path: Optional[str] = None,
    env_vars: Optional[dict] = None,
) -> type:
    """
    Wrap a node class so its FUNCTION method runs in the isolated environment.

    Args:
        cls: The node class to wrap
        env_dir: Path to the isolated environment directory
        working_dir: Working directory for the worker
        sys_path: Additional paths to add to sys.path in the worker
        lib_path: Path to add to LD_LIBRARY_PATH for conda libraries

    Returns:
        The wrapped class (modified in place)
    """
    func_name = getattr(cls, "FUNCTION", None)
    if not func_name:
        return cls  # Not a valid ComfyUI node class

    original_method = getattr(cls, func_name, None)
    if original_method is None:
        return cls

    # Get source file for the class
    try:
        source_file = Path(inspect.getfile(cls)).resolve()
    except (TypeError, OSError):
        # Can't get source file, skip wrapping
        return cls

    # Compute relative module path from working_dir
    # e.g., /path/to/nodes/io/load_mesh.py -> nodes.io.load_mesh
    try:
        relative_path = source_file.relative_to(working_dir)
        # Convert path to module: nodes/io/load_mesh.py -> nodes.io.load_mesh
        module_name = str(relative_path.with_suffix("")).replace("/", ".").replace("\\", ".")
    except ValueError:
        # File not under working_dir, use stem as fallback
        module_name = source_file.stem

    @wraps(original_method)
    def proxy(self, **kwargs):
        if _DEBUG:
            print(f"[comfy-env] PROXY CALLED: {cls.__name__}.{func_name}", flush=True)
            print(f"[comfy-env]   kwargs keys: {list(kwargs.keys())}", flush=True)

        worker = _get_worker(env_dir, working_dir, sys_path, lib_path, env_vars)
        if _DEBUG:
            print(f"[comfy-env]   worker alive: {worker.is_alive()}", flush=True)

        # Clone tensors for IPC if needed
        try:
            from ..workers.tensor_utils import prepare_for_ipc_recursive

            kwargs = {k: prepare_for_ipc_recursive(v) for k, v in kwargs.items()}
        except ImportError:
            pass  # No torch available, skip cloning

        if _DEBUG:
            print(f"[comfy-env]   calling worker.call_method...", flush=True)
        result = worker.call_method(
            module_name=module_name,
            class_name=cls.__name__,
            method_name=func_name,
            self_state=self.__dict__.copy() if hasattr(self, "__dict__") else None,
            kwargs=kwargs,
            timeout=600.0,
        )
        if _DEBUG:
            print(f"[comfy-env]   call_method returned", flush=True)

        # Clone result tensors
        try:
            from ..workers.tensor_utils import prepare_for_ipc_recursive

            result = prepare_for_ipc_recursive(result)
        except ImportError:
            pass

        return result

    # Replace the method
    setattr(cls, func_name, proxy)

    # Mark as isolated for debugging
    cls._comfy_env_isolated = True

    return cls


def wrap_isolated_nodes(
    node_class_mappings: Dict[str, type],
    nodes_dir: Path,
) -> Dict[str, type]:
    """
    Wrap nodes from a directory that has a comfy-env.toml.

    This is the directory-based isolation API. Call it for each subdirectory
    of nodes/ that has a comfy-env.toml.

    Args:
        node_class_mappings: The NODE_CLASS_MAPPINGS dict from the nodes in this dir.
        nodes_dir: The directory containing comfy-env.toml and the node files.

    Returns:
        The same dict with node classes wrapped for isolation.

    Example:
        # __init__.py
        from comfy_env import wrap_isolated_nodes
        from pathlib import Path

        NODE_CLASS_MAPPINGS = {}

        # Native nodes (no isolation)
        from .nodes.main import NODE_CLASS_MAPPINGS as main_nodes
        NODE_CLASS_MAPPINGS.update(main_nodes)

        # Isolated nodes (has comfy-env.toml)
        from .nodes.cgal import NODE_CLASS_MAPPINGS as cgal_nodes
        NODE_CLASS_MAPPINGS.update(
            wrap_isolated_nodes(cgal_nodes, Path(__file__).parent / "nodes/cgal")
        )
    """
    # Skip if running inside worker subprocess
    if os.environ.get("COMFYUI_ISOLATION_WORKER") == "1":
        return node_class_mappings

    # Get ComfyUI base path from folder_paths (canonical source)
    try:
        import folder_paths
        comfyui_base = folder_paths.base_path
    except ImportError:
        comfyui_base = None

    nodes_dir = Path(nodes_dir).resolve()

    # Check for comfy-env.toml
    config_file = nodes_dir / "comfy-env.toml"
    if not config_file.exists():
        print(f"[comfy-env] Warning: No comfy-env.toml in {nodes_dir}")
        return node_class_mappings

    # Read env_vars from comfy-env.toml
    env_vars = {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        with open(config_file, "rb") as f:
            config = tomllib.load(f)
        env_vars_data = config.get("env_vars", {})
        env_vars = {str(k): str(v) for k, v in env_vars_data.items()}
    except Exception:
        pass  # Ignore errors reading config

    # Set COMFYUI_BASE for worker to find ComfyUI modules
    if comfyui_base:
        env_vars["COMFYUI_BASE"] = str(comfyui_base)

    # Find environment directory and paths
    env_dir = _find_env_dir(nodes_dir)
    site_packages, lib_dir = _find_env_paths(nodes_dir)

    if not env_dir or not site_packages:
        print(f"[comfy-env] Warning: Isolated environment not found")
        print(f"[comfy-env] Expected: .pixi/envs/default or .venv")
        print(f"[comfy-env] Run 'comfy-env install' in {nodes_dir}")
        return node_class_mappings

    # Build sys.path - site-packages first, then nodes_dir
    # Note: isolated modules should use absolute imports (their dir is in sys.path)
    # Relative imports would require importing parent package which may have host-only deps
    sys_path = [str(site_packages), str(nodes_dir)]

    # lib_dir for LD_LIBRARY_PATH (conda libraries)
    lib_path = str(lib_dir) if lib_dir else None

    print(f"[comfy-env] Wrapping {len(node_class_mappings)} nodes from {nodes_dir.name}")
    print(f"[comfy-env]   site-packages: {site_packages}")
    if lib_path:
        print(f"[comfy-env]   lib: {lib_path}")
    if env_vars:
        print(f"[comfy-env]   env_vars: {', '.join(f'{k}={v}' for k, v in env_vars.items())}")

    # Wrap all node classes
    for node_name, node_cls in node_class_mappings.items():
        if hasattr(node_cls, "FUNCTION"):
            _wrap_node_class(node_cls, env_dir, nodes_dir, sys_path, lib_path, env_vars)

    return node_class_mappings
