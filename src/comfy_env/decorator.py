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
from typing import Any, Callable, Dict, List, Optional, Union

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
            worker = TorchMPWorker(name=config.env_name)
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
        if node_dir.name == "nodes":
            node_package_dir = node_dir.parent
        else:
            node_package_dir = node_dir

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
                log_fn(f"→ {cls.__name__}.{func_name}({_describe_args(call_kwargs)})")

            start_time = time.time()

            try:
                # Clone tensors defensively if enabled
                if clone_tensors:
                    call_kwargs = {k: _clone_tensor_if_needed(v) for k, v in call_kwargs.items()}

                # Get module name for import in worker
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
                    log_fn(f"← {cls.__name__}.{func_name} returned {result_desc} [{elapsed:.2f}s]")

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                log_fn(f"✗ {cls.__name__}.{func_name} failed after {elapsed:.2f}s: {e}")
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
