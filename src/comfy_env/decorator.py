"""
Decorator-based API for easy subprocess isolation.

This module provides the @isolated decorator that makes it simple to run
ComfyUI node methods in isolated subprocess environments.

Architecture:
    The decorator wraps the node's FUNCTION method. When called in the HOST
    process, it forwards the call to an isolated worker (TorchMPWorker for
    same-venv, PersistentVenvWorker for different venv).

    When imported in the WORKER subprocess (COMFYUI_ISOLATION_WORKER=1),
    the decorator is a transparent no-op.

Example:
    from comfy_env import isolated

    @isolated(env="myenv")
    class MyNode:
        FUNCTION = "process"
        RETURN_TYPES = ("IMAGE",)

        def process(self, image):
            # This code runs in isolated subprocess
            import heavy_package
            return (heavy_package.run(image),)

Implementation:
    This decorator is thin sugar over the workers module. Internally it uses:
    - TorchMPWorker: Same Python, zero-copy tensor transfer via torch.mp.Queue
    - PersistentVenvWorker: Different venv, tensor transfer via torch.save/load
"""

import os
import sys
import atexit
import inspect
import logging
import threading
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("comfy_env")

# Enable verbose logging by default (can be disabled)
VERBOSE_LOGGING = os.environ.get("COMFYUI_ISOLATION_QUIET", "0") != "1"


def _log(env_name: str, msg: str):
    """Log with environment prefix."""
    if VERBOSE_LOGGING:
        print(f"[{env_name}] {msg}")


def _is_worker_mode() -> bool:
    """Check if we're running inside the worker subprocess."""
    return os.environ.get("COMFYUI_ISOLATION_WORKER") == "1"


def _describe_tensor(t) -> str:
    """Get human-readable tensor description."""
    try:
        import torch
        if isinstance(t, torch.Tensor):
            size_mb = t.numel() * t.element_size() / (1024 * 1024)
            return f"Tensor({list(t.shape)}, {t.dtype}, {t.device}, {size_mb:.1f}MB)"
    except:
        pass
    return str(type(t).__name__)


def _describe_args(args: dict) -> str:
    """Describe arguments for logging."""
    parts = []
    for k, v in args.items():
        parts.append(f"{k}={_describe_tensor(v)}")
    return ", ".join(parts) if parts else "(no args)"


def _clone_tensor_if_needed(obj: Any, smart_clone: bool = True) -> Any:
    """
    Defensively clone tensors to prevent mutation/re-share bugs.

    This handles:
    1. Input tensors that might be mutated in worker
    2. Output tensors received via IPC that can't be re-shared

    Args:
        obj: Object to process (tensor or nested structure)
        smart_clone: If True, use smart CUDA IPC detection (only clone
                    when necessary). If False, always clone.
    """
    if smart_clone:
        # Use smart detection - only clones CUDA tensors that can't be re-shared
        from .workers.tensor_utils import prepare_for_ipc_recursive
        return prepare_for_ipc_recursive(obj)

    # Fallback: always clone (original behavior)
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        elif isinstance(obj, (list, tuple)):
            cloned = [_clone_tensor_if_needed(x, smart_clone=False) for x in obj]
            return type(obj)(cloned)
        elif isinstance(obj, dict):
            return {k: _clone_tensor_if_needed(v, smart_clone=False) for k, v in obj.items()}
    except ImportError:
        pass
    return obj


def _find_node_package_dir(source_file: Path) -> Path:
    """
    Find the node package root directory by searching for comfy-env.toml.

    Walks up from the source file's directory until it finds a config file,
    or falls back to heuristics if not found.
    """
    from .env.config_file import CONFIG_FILE_NAMES

    current = source_file.parent

    # Walk up the directory tree looking for config file
    while current != current.parent:  # Stop at filesystem root
        for config_name in CONFIG_FILE_NAMES:
            if (current / config_name).exists():
                return current
        current = current.parent

    # Fallback: use old heuristic if no config found
    node_dir = source_file.parent
    if node_dir.name == "nodes":
        return node_dir.parent
    return node_dir


# ---------------------------------------------------------------------------
# Worker Management
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Configuration for an isolated worker."""
    env_name: str
    python: Optional[str] = None  # None = same Python (TorchMPWorker)
    working_dir: Optional[Path] = None
    sys_path: Optional[List[str]] = None
    timeout: float = 600.0


# Global worker cache
_workers: Dict[str, Any] = {}
_workers_lock = threading.Lock()


def _get_or_create_worker(config: WorkerConfig, log_fn: Callable):
    """Get or create a worker for the given configuration.

    Thread-safe: worker creation happens inside the lock to prevent
    race conditions where multiple threads create duplicate workers.
    """
    cache_key = f"{config.env_name}:{config.python or 'same'}"

    with _workers_lock:
        if cache_key in _workers:
            worker = _workers[cache_key]
            if worker.is_alive():
                return worker
            # Worker died, recreate
            log_fn(f"Worker died, recreating...")

        # Create new worker INSIDE the lock (fixes race condition)
        if config.python is None:
            # Same Python - use TorchMPWorker (fast, zero-copy)
            from .workers import TorchMPWorker
            log_fn(f"Creating TorchMPWorker (same Python, zero-copy tensors)")
            worker = TorchMPWorker(
                name=config.env_name,
                sys_path=config.sys_path,
            )
        else:
            # Different Python - use PersistentVenvWorker
            from .workers.venv import PersistentVenvWorker
            log_fn(f"Creating PersistentVenvWorker (python={config.python})")
            worker = PersistentVenvWorker(
                python=config.python,
                working_dir=config.working_dir,
                sys_path=config.sys_path,
                name=config.env_name,
            )

        _workers[cache_key] = worker
        return worker


def shutdown_all_processes():
    """Shutdown all cached workers. Called at exit."""
    with _workers_lock:
        for name, worker in _workers.items():
            try:
                worker.shutdown()
            except Exception as e:
                logger.debug(f"Error shutting down {name}: {e}")
        _workers.clear()


atexit.register(shutdown_all_processes)


# ---------------------------------------------------------------------------
# The @isolated Decorator
# ---------------------------------------------------------------------------

def isolated(
    env: str,
    requirements: Optional[List[str]] = None,
    config: Optional[str] = None,
    python: Optional[str] = None,
    cuda: Optional[str] = "auto",
    timeout: float = 600.0,
    log_callback: Optional[Callable[[str], None]] = None,
    import_paths: Optional[List[str]] = None,
    clone_tensors: bool = True,
    same_venv: bool = False,
):
    """
    Class decorator that runs node methods in isolated subprocess.

    The decorated class's FUNCTION method will be executed in an isolated
    Python environment. Tensors are transferred efficiently via PyTorch's
    native IPC mechanisms (CUDA IPC for GPU, shared memory for CPU).

    By default, auto-discovers config file (comfy_env_reqs.toml) and
    uses full venv isolation with PersistentVenvWorker. Use same_venv=True
    for lightweight same-venv isolation with TorchMPWorker.

    Args:
        env: Name of the isolated environment (used for logging/caching)
        requirements: [DEPRECATED] Use config file instead
        config: Path to TOML config file. If None, auto-discovers in node directory.
        python: Path to Python executable (overrides config-based detection)
        cuda: [DEPRECATED] Detected automatically
        timeout: Timeout for calls in seconds (default: 10 minutes)
        log_callback: Optional callback for logging
        import_paths: Paths to add to sys.path in worker
        clone_tensors: Clone tensors at boundary to prevent mutation bugs (default: True)
        same_venv: If True, use TorchMPWorker (same venv, just process isolation).
                   If False (default), use full venv isolation with auto-discovered config.

    Example:
        # Full venv isolation (default) - auto-discovers comfy_env_reqs.toml
        @isolated(env="sam3d")
        class MyNode:
            FUNCTION = "process"

            def process(self, image):
                import heavy_lib
                return heavy_lib.run(image)

        # Lightweight same-venv isolation (opt-in)
        @isolated(env="sam3d", same_venv=True)
        class MyLightNode:
            FUNCTION = "process"
            ...
    """
    def decorator(cls):
        # In worker mode, decorator is a no-op
        if _is_worker_mode():
            return cls

        # --- HOST MODE: Wrap the FUNCTION method ---

        func_name = getattr(cls, 'FUNCTION', None)
        if not func_name:
            raise ValueError(
                f"Node class {cls.__name__} must have FUNCTION attribute."
            )

        original_method = getattr(cls, func_name, None)
        if original_method is None:
            raise ValueError(
                f"Node class {cls.__name__} has FUNCTION='{func_name}' but "
                f"no method with that name."
            )

        # Get source file info for sys.path setup
        source_file = Path(inspect.getfile(cls))
        node_dir = source_file.parent
        node_package_dir = _find_node_package_dir(source_file)

        # Build sys.path for worker
        sys_path_additions = [str(node_dir)]
        if import_paths:
            for p in import_paths:
                full_path = node_dir / p
                sys_path_additions.append(str(full_path.resolve()))

        # Resolve python path for venv isolation
        resolved_python = python
        env_config = None

        # If same_venv=True, skip venv isolation entirely
        if same_venv:
            _log(env, "Using same-venv isolation (TorchMPWorker)")
            resolved_python = None

        # Otherwise, try to get a venv python path
        elif python:
            # Explicit python path provided
            resolved_python = python

        else:
            # Auto-discover or use explicit config
            if config:
                # Explicit config file specified
                config_file = node_package_dir / config
                if config_file.exists():
                    from .env.config_file import load_env_from_file
                    env_config = load_env_from_file(config_file, node_package_dir)
                else:
                    _log(env, f"Warning: Config file not found: {config_file}")
            else:
                # Auto-discover config file - try v2 API first
                from .env.config_file import discover_config, discover_env_config
                v2_config = discover_config(node_package_dir)
                if v2_config and env in v2_config.envs:
                    # v2 schema: get the named environment
                    env_config = v2_config.envs[env]
                    _log(env, f"Auto-discovered v2 config: {env_config.name}")
                else:
                    # Fall back to v1 API
                    env_config = discover_env_config(node_package_dir)
                    if env_config:
                        _log(env, f"Auto-discovered config: {env_config.name}")

            # If we have a config, set up the venv
            if env_config:
                from .env.manager import IsolatedEnvManager
                manager = IsolatedEnvManager(base_dir=node_package_dir)

                if not manager.is_ready(env_config):
                    _log(env, f"Setting up isolated environment...")
                    manager.setup(env_config)

                resolved_python = str(manager.get_python(env_config))
            else:
                # No config found - fall back to same-venv isolation
                _log(env, "No config found, using same-venv isolation (TorchMPWorker)")
                resolved_python = None

        # Create worker config
        worker_config = WorkerConfig(
            env_name=env,
            python=resolved_python,
            working_dir=node_dir,
            sys_path=sys_path_additions,
            timeout=timeout,
        )

        # Setup logging
        log_fn = log_callback or (lambda msg: _log(env, msg))

        # Create the proxy method
        @wraps(original_method)
        def proxy(self, *args, **kwargs):
            # Get or create worker
            worker = _get_or_create_worker(worker_config, log_fn)

            # Bind arguments to get kwargs dict
            sig = inspect.signature(original_method)
            try:
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                call_kwargs = dict(bound.arguments)
                del call_kwargs['self']
            except TypeError:
                call_kwargs = kwargs

            # Log entry with argument descriptions
            if VERBOSE_LOGGING:
                log_fn(f"-> {cls.__name__}.{func_name}({_describe_args(call_kwargs)})")

            start_time = time.time()

            try:
                # Clone tensors defensively if enabled
                if clone_tensors:
                    call_kwargs = {k: _clone_tensor_if_needed(v) for k, v in call_kwargs.items()}

                # Get module name for import in worker
                # Note: ComfyUI uses full filesystem paths as module names for custom nodes.
                # The worker's _execute_method_call handles this by using file-based imports.
                module_name = cls.__module__

                # Call worker using appropriate method
                if worker_config.python is None:
                    # TorchMPWorker - use call_method protocol (avoids pickle issues)
                    result = worker.call_method(
                        module_name=module_name,
                        class_name=cls.__name__,
                        method_name=func_name,
                        self_state=self.__dict__.copy(),
                        kwargs=call_kwargs,
                        timeout=timeout,
                    )
                else:
                    # PersistentVenvWorker - call by module/class/method path
                    result = worker.call_method(
                        module_name=source_file.stem,
                        class_name=cls.__name__,
                        method_name=func_name,
                        self_state=self.__dict__.copy() if hasattr(self, '__dict__') else None,
                        kwargs=call_kwargs,
                        timeout=timeout,
                    )

                # Clone result tensors defensively
                if clone_tensors:
                    result = _clone_tensor_if_needed(result)

                elapsed = time.time() - start_time
                if VERBOSE_LOGGING:
                    result_desc = _describe_tensor(result) if not isinstance(result, tuple) else f"tuple({len(result)} items)"
                    log_fn(f"<- {cls.__name__}.{func_name} returned {result_desc} [{elapsed:.2f}s]")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                log_fn(f"[FAIL] {cls.__name__}.{func_name} failed after {elapsed:.2f}s: {e}")
                raise

        # Store original method before replacing (for worker to access)
        cls._isolated_original_method = original_method

        # Replace method with proxy
        setattr(cls, func_name, proxy)

        # Store metadata
        cls._isolated_env = env
        cls._isolated_node_dir = node_dir

        return cls

    return decorator


# ---------------------------------------------------------------------------
# The @auto_isolate Decorator (Function-level)
# ---------------------------------------------------------------------------

def _parse_import_error(e: ImportError) -> Optional[str]:
    """Extract the module name from an ImportError."""
    # Python's ImportError has a 'name' attribute with the module name
    if hasattr(e, 'name') and e.name:
        return e.name

    # Fallback: parse from message "No module named 'xxx'"
    msg = str(e)
    if "No module named" in msg:
        # Extract 'xxx' from "No module named 'xxx'" or "No module named 'xxx.yyy'"
        import re
        match = re.search(r"No module named ['\"]([^'\"\.]+)", msg)
        if match:
            return match.group(1)

    return None


def _find_env_for_module(
    module_name: str,
    source_file: Path,
) -> Optional[Tuple[str, Path, Path]]:
    """
    Find which isolated environment contains the given module.

    Searches comfy-env.toml configs starting from the source file's directory,
    looking for the module in cuda packages, requirements, etc.

    Args:
        module_name: The module that failed to import (e.g., "cumesh")
        source_file: Path to the source file containing the function

    Returns:
        Tuple of (env_name, python_path, node_dir) or None if not found
    """
    from .env.config_file import discover_config, CONFIG_FILE_NAMES

    # Normalize module name (cumesh, pytorch3d, etc.)
    module_lower = module_name.lower().replace("-", "_").replace(".", "_")

    # Search for config file starting from source file's directory
    node_dir = source_file.parent
    while node_dir != node_dir.parent:
        for config_name in CONFIG_FILE_NAMES:
            config_path = node_dir / config_name
            if config_path.exists():
                # Found a config, check if it has our module
                config = discover_config(node_dir)
                if config is None:
                    continue

                # Check all environments in the config
                for env_name, env_config in config.envs.items():
                    # Check cuda/no_deps_requirements
                    if env_config.no_deps_requirements:
                        for req in env_config.no_deps_requirements:
                            req_name = req.split("==")[0].split(">=")[0].split("<")[0].strip()
                            req_lower = req_name.lower().replace("-", "_")
                            if req_lower == module_lower:
                                # Found it! Get the python path
                                env_path = node_dir / f"_env_{env_name}"
                                if not env_path.exists():
                                    # Try pixi path
                                    env_path = node_dir / ".pixi" / "envs" / "default"

                                if env_path.exists():
                                    python_path = env_path / "bin" / "python"
                                    if not python_path.exists():
                                        python_path = env_path / "Scripts" / "python.exe"
                                    if python_path.exists():
                                        return (env_name, python_path, node_dir)

                    # Check regular requirements too
                    if env_config.requirements:
                        for req in env_config.requirements:
                            req_name = req.split("==")[0].split(">=")[0].split("<")[0].split("[")[0].strip()
                            req_lower = req_name.lower().replace("-", "_")
                            if req_lower == module_lower:
                                env_path = node_dir / f"_env_{env_name}"
                                if not env_path.exists():
                                    env_path = node_dir / ".pixi" / "envs" / "default"

                                if env_path.exists():
                                    python_path = env_path / "bin" / "python"
                                    if not python_path.exists():
                                        python_path = env_path / "Scripts" / "python.exe"
                                    if python_path.exists():
                                        return (env_name, python_path, node_dir)

                # Config found but module not in it, stop searching
                break

        node_dir = node_dir.parent

    return None


# Cache for auto_isolate workers
_auto_isolate_workers: Dict[str, Any] = {}
_auto_isolate_lock = threading.Lock()


def _get_auto_isolate_worker(env_name: str, python_path: Path, node_dir: Path):
    """Get or create a worker for auto_isolate."""
    cache_key = f"{env_name}:{python_path}"

    with _auto_isolate_lock:
        if cache_key in _auto_isolate_workers:
            worker = _auto_isolate_workers[cache_key]
            if worker.is_alive():
                return worker

        # Create new PersistentVenvWorker
        from .workers.venv import PersistentVenvWorker

        worker = PersistentVenvWorker(
            python=str(python_path),
            working_dir=node_dir,
            sys_path=[str(node_dir)],
            name=f"auto-{env_name}",
        )

        _auto_isolate_workers[cache_key] = worker
        return worker


def auto_isolate(func: Callable) -> Callable:
    """
    Decorator that automatically runs a function in an isolated environment
    when an ImportError occurs for a package that exists in the isolated env.

    This provides seamless isolation - just write normal code with imports,
    and if the import fails in the host environment but the package is
    configured in comfy-env.toml, the function automatically retries in
    the isolated environment.

    Example:
        from comfy_env import auto_isolate

        @auto_isolate
        def process_with_cumesh(mesh, target_faces):
            import cumesh  # If this fails, function retries in isolated env
            import torch

            v = torch.tensor(mesh.vertices).cuda()
            f = torch.tensor(mesh.faces).cuda()

            cm = cumesh.CuMesh()
            cm.init(v, f)
            cm.simplify(target_faces)

            result_v, result_f = cm.read()
            return result_v.cpu().numpy(), result_f.cpu().numpy()

    How it works:
        1. Function runs normally in the host environment
        2. If ImportError occurs, decorator catches it
        3. Extracts the module name from the error (e.g., "cumesh")
        4. Searches comfy-env.toml for which env has that module
        5. Re-runs the entire function in that isolated environment
        6. Returns the result as if nothing happened

    Benefits:
        - Zero overhead when imports succeed (fast path)
        - Auto-detects which environment to use from the failed import
        - Function is the isolation boundary (clean, debuggable)
        - Works with any import pattern (top of function, conditional, etc.)

    Note:
        Arguments and return values are serialized via torch.save/load,
        so they should be tensors, numpy arrays, or pickle-able objects.
    """
    # Get source file for environment detection
    source_file = Path(inspect.getfile(func))

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Fast path: try running in host environment
            return func(*args, **kwargs)

        except ImportError as e:
            # Extract module name from error
            module_name = _parse_import_error(e)
            if module_name is None:
                # Can't determine module, re-raise
                raise

            # Find which env has this module
            env_info = _find_env_for_module(module_name, source_file)
            if env_info is None:
                # Module not in any known isolated env, re-raise
                raise

            env_name, python_path, node_dir = env_info

            _log(env_name, f"Import '{module_name}' failed in host, retrying in isolated env...")
            _log(env_name, f"  Python: {python_path}")

            # Get or create worker
            worker = _get_auto_isolate_worker(env_name, python_path, node_dir)

            # Prepare arguments - convert numpy arrays to lists for IPC
            import numpy as np

            def convert_for_ipc(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, 'vertices') and hasattr(obj, 'faces'):
                    # Trimesh-like object - convert to dict
                    return {
                        '__trimesh__': True,
                        'vertices': obj.vertices.tolist() if hasattr(obj.vertices, 'tolist') else list(obj.vertices),
                        'faces': obj.faces.tolist() if hasattr(obj.faces, 'tolist') else list(obj.faces),
                    }
                elif isinstance(obj, (list, tuple)):
                    converted = [convert_for_ipc(x) for x in obj]
                    return type(obj)(converted) if isinstance(obj, tuple) else converted
                elif isinstance(obj, dict):
                    return {k: convert_for_ipc(v) for k, v in obj.items()}
                return obj

            converted_args = [convert_for_ipc(arg) for arg in args]
            converted_kwargs = {k: convert_for_ipc(v) for k, v in kwargs.items()}

            # Call via worker
            start_time = time.time()

            result = worker.call_module(
                module=source_file.stem,
                func=func.__name__,
                *converted_args,
                **converted_kwargs,
            )

            elapsed = time.time() - start_time
            _log(env_name, f"<- {func.__name__} completed in isolated env [{elapsed:.2f}s]")

            return result

    # Mark the function as auto-isolate enabled
    wrapper._auto_isolate = True
    wrapper._source_file = source_file

    return wrapper
