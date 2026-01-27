"""
Process isolation for ComfyUI node packs.

This module provides enable_isolation() which wraps all node classes
to run their FUNCTION methods in an isolated Python environment.

Usage:
    # In your node pack's __init__.py:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from comfy_env import enable_isolation

    enable_isolation(NODE_CLASS_MAPPINGS)  # That's it!

This requires `isolated = true` in comfy-env.toml:

    [myenv]
    python = "3.11"
    isolated = true

    [myenv.packages]
    requirements = ["my-package"]
"""

import atexit
import inspect
import os
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

# Global worker cache (one per isolated environment)
_workers: Dict[str, Any] = {}
_workers_lock = threading.Lock()


def _get_worker(
    env_name: str,
    python_path: Path,
    working_dir: Path,
    sys_path: list[str],
):
    """Get or create a persistent worker for the isolated environment."""
    from .workers.venv import PersistentVenvWorker

    cache_key = str(python_path)

    with _workers_lock:
        if cache_key in _workers:
            worker = _workers[cache_key]
            if worker.is_alive():
                return worker
            # Worker died, will recreate

        print(f"[comfy-env] Starting isolated worker: {env_name}")
        print(f"[comfy-env]   Python: {python_path}")

        worker = PersistentVenvWorker(
            python=str(python_path),
            working_dir=working_dir,
            sys_path=sys_path,
            name=env_name,
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


def _find_python_path(node_dir: Path, env_name: str) -> Optional[Path]:
    """
    Find the Python executable for the isolated environment.

    Priority:
    1. .pixi/envs/default/bin/python (pixi/conda environment)
    2. _env_{name}/bin/python (uv venv)
    3. _env_{name}/Scripts/python.exe (Windows uv venv)
    """
    # Check pixi environment first
    if sys.platform == "win32":
        pixi_python = node_dir / ".pixi" / "envs" / "default" / "python.exe"
    else:
        pixi_python = node_dir / ".pixi" / "envs" / "default" / "bin" / "python"

    if pixi_python.exists():
        return pixi_python

    # Check _env_* directory (uv venv)
    env_dir = node_dir / f"_env_{env_name}"
    if sys.platform == "win32":
        env_python = env_dir / "Scripts" / "python.exe"
    else:
        env_python = env_dir / "bin" / "python"

    if env_python.exists():
        return env_python

    return None


def _wrap_node_class(
    cls: type,
    env_name: str,
    python_path: Path,
    working_dir: Path,
    sys_path: list[str],
) -> type:
    """
    Wrap a node class so its FUNCTION method runs in the isolated environment.

    Args:
        cls: The node class to wrap
        env_name: Name of the isolated environment
        python_path: Path to the isolated Python executable
        working_dir: Working directory for the worker
        sys_path: Additional paths to add to sys.path in the worker

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
        print(f"[comfy-env] PROXY CALLED: {cls.__name__}.{func_name}", flush=True)
        print(f"[comfy-env]   kwargs keys: {list(kwargs.keys())}", flush=True)

        worker = _get_worker(env_name, python_path, working_dir, sys_path)
        print(f"[comfy-env]   worker alive: {worker.is_alive()}", flush=True)

        # Clone tensors for IPC if needed
        try:
            from .decorator import _clone_tensor_if_needed

            kwargs = {k: _clone_tensor_if_needed(v) for k, v in kwargs.items()}
        except ImportError:
            pass  # No torch available, skip cloning

        print(f"[comfy-env]   calling worker.call_method...", flush=True)
        result = worker.call_method(
            module_name=module_name,
            class_name=cls.__name__,
            method_name=func_name,
            self_state=self.__dict__.copy() if hasattr(self, "__dict__") else None,
            kwargs=kwargs,
            timeout=600.0,
        )
        print(f"[comfy-env]   call_method returned", flush=True)

        # Clone result tensors
        try:
            from .decorator import _clone_tensor_if_needed

            result = _clone_tensor_if_needed(result)
        except ImportError:
            pass

        return result

    # Replace the method
    setattr(cls, func_name, proxy)

    # Mark as isolated for debugging
    cls._comfy_env_isolated = True
    cls._comfy_env_name = env_name

    return cls


def enable_isolation(node_class_mappings: Dict[str, type]) -> None:
    """
    Enable process isolation for all nodes in a node pack.

    Call this AFTER importing NODE_CLASS_MAPPINGS. It wraps all node classes
    so their FUNCTION methods run in the isolated Python environment specified
    in comfy-env.toml.

    Requires `isolated = true` in comfy-env.toml:

        [myenv]
        python = "3.11"
        isolated = true

    Args:
        node_class_mappings: The NODE_CLASS_MAPPINGS dict from the node pack.

    Example:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        from comfy_env import enable_isolation

        enable_isolation(NODE_CLASS_MAPPINGS)

        __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    """
    # Skip if running inside worker subprocess
    if os.environ.get("COMFYUI_ISOLATION_WORKER") == "1":
        return

    # Find the calling module's directory (node pack root)
    frame = inspect.currentframe()
    if frame is None:
        print("[comfy-env] Warning: Could not get current frame")
        return

    caller_frame = frame.f_back
    if caller_frame is None:
        print("[comfy-env] Warning: Could not get caller frame")
        return

    caller_file = caller_frame.f_globals.get("__file__")
    if not caller_file:
        print("[comfy-env] Warning: Could not determine caller location")
        return

    node_dir = Path(caller_file).resolve().parent

    # Load config
    from .env.config_file import discover_config

    config = discover_config(node_dir)
    if not config:
        print(f"[comfy-env] No comfy-env.toml found in {node_dir}")
        return

    # Find isolated environment
    isolated_env = None
    env_name = None

    for name, env in config.envs.items():
        if getattr(env, "isolated", False):
            isolated_env = env
            env_name = name
            break

    if not isolated_env or not env_name:
        # No isolated env configured, silently return
        return

    # Find Python executable
    python_path = _find_python_path(node_dir, env_name)

    if not python_path:
        print(f"[comfy-env] Warning: Isolated environment not found for '{env_name}'")
        print(f"[comfy-env] Expected: .pixi/envs/default/bin/python or _env_{env_name}/bin/python")
        print(f"[comfy-env] Run 'comfy-env install' to create the environment")
        return

    # Build sys.path for the worker
    sys_path = [str(node_dir)]

    # Add nodes directory if it exists
    nodes_dir = node_dir / "nodes"
    if nodes_dir.exists():
        sys_path.append(str(nodes_dir))

    print(f"[comfy-env] Enabling isolation for {len(node_class_mappings)} nodes")
    print(f"[comfy-env]   Environment: {env_name}")
    print(f"[comfy-env]   Python: {python_path}")

    # Wrap all node classes
    wrapped_count = 0
    for node_name, node_cls in node_class_mappings.items():
        if hasattr(node_cls, "FUNCTION"):
            _wrap_node_class(node_cls, env_name, python_path, node_dir, sys_path)
            wrapped_count += 1

    print(f"[comfy-env] Wrapped {wrapped_count} node classes for isolation")
