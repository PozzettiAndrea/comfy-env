"""
MPWorker - Same-venv isolation using multiprocessing.

This is the simplest and fastest worker type:
- Uses multiprocessing.Queue for IPC
- Zero-copy tensor transfer via shared memory (automatic)
- Fresh CUDA context in subprocess
- ~30ms overhead per call

Use this when you need:
- Memory isolation between nodes
- Fresh CUDA context (automatic VRAM cleanup on worker death)
- Same Python environment as host

Example:
    worker = MPWorker()

    def gpu_work(image):
        import torch
        return image * 2

    result = worker.call(gpu_work, image=my_tensor)
    worker.shutdown()
"""

import logging
import traceback
from queue import Empty as QueueEmpty
from typing import Any, Callable, Optional

from .base import Worker, WorkerError

logger = logging.getLogger("comfy_env")


# Sentinel value for shutdown
_SHUTDOWN = object()

# Message type for method calls (avoids pickling issues with functions)
_CALL_METHOD = "call_method"


# ---------------------------------------------------------------------------
# Tensor file transfer - avoids CUDA IPC issues with cudaMallocAsync
# ---------------------------------------------------------------------------

def _save_tensors_to_files(obj, file_registry=None):
    """Recursively save torch tensors to temp files for IPC."""
    if file_registry is None:
        file_registry = []

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            import tempfile
            f = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
            torch.save(obj.cpu(), f.name)  # Always save as CPU tensor
            f.close()
            file_registry.append(f.name)
            return {"__tensor_file__": f.name, "dtype": str(obj.dtype), "device": str(obj.device)}
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _save_tensors_to_files(v, file_registry) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_save_tensors_to_files(v, file_registry) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_save_tensors_to_files(v, file_registry) for v in obj)
    return obj


def _load_tensors_from_files(obj):
    """Recursively load torch tensors from temp files."""
    if isinstance(obj, dict):
        if "__tensor_file__" in obj:
            import os
            import torch
            tensor = torch.load(obj["__tensor_file__"], weights_only=True)
            os.unlink(obj["__tensor_file__"])  # Cleanup temp file
            return tensor
        return {k: _load_tensors_from_files(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_load_tensors_from_files(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_load_tensors_from_files(v) for v in obj)
    return obj


def _dump_worker_env(worker_name: str = "unknown", print_to_terminal: bool = False):
    """Dump worker environment to .comfy-env/logs/ (always) and optionally print."""
    import json
    import os
    import platform
    import sys
    from datetime import datetime
    from pathlib import Path

    log_dir = Path.cwd() / ".comfy-env" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "worker_name": worker_name,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "env_vars": dict(os.environ),
        "sys_path": sys.path,
        "modules_loaded": sorted(sys.modules.keys()),
    }

    log_file = log_dir / f"worker_{worker_name}_{os.getpid()}.json"
    log_file.write_text(json.dumps(debug_info, indent=2, default=str))

    if print_to_terminal:
        print(f"[comfy-env] === WORKER ENV DEBUG: {worker_name} ===")
        print(f"[comfy-env] Python: {sys.executable}")
        print(f"[comfy-env] Version: {sys.version.split()[0]}")
        print(f"[comfy-env] PID: {os.getpid()}, CWD: {os.getcwd()}")
        for var in ['PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'PYTHONPATH', 'OMP_NUM_THREADS', 'KMP_DUPLICATE_LIB_OK']:
            val = os.environ.get(var, '<unset>')
            if len(val) > 100:
                val = val[:100] + '...'
            print(f"[comfy-env] {var}={val}")
        print(f"[comfy-env] Env dumped to: {log_file}")


def _worker_loop(queue_in, queue_out, sys_path_additions=None, lib_path=None, env_vars=None, worker_name=None):
    """
    Worker process main loop.

    Receives work items and executes them:
    - ("call_method", module_name, class_name, method_name, self_state, kwargs): Call a method on a class
    - (func, args, kwargs): Execute a function directly
    - _SHUTDOWN: Shutdown the worker

    Runs until receiving _SHUTDOWN sentinel.

    Args:
        queue_in: Input queue for receiving work items
        queue_out: Output queue for sending results
        sys_path_additions: Paths to add to sys.path
        lib_path: Path to add to LD_LIBRARY_PATH (for conda libraries)
        env_vars: Environment variables to set (from comfy-env.toml)
        worker_name: Name of the worker (for logging)
    """
    import os
    import sys
    from pathlib import Path

    # Apply env_vars FIRST (before any library imports that might check them)
    if env_vars:
        os.environ.update(env_vars)

    # Set worker mode env var
    os.environ["COMFYUI_ISOLATION_WORKER"] = "1"

    # Always dump env to file, print to terminal if debug enabled
    print_debug = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")
    _dump_worker_env(worker_name or "unknown", print_to_terminal=print_debug)

    # DLL/library isolation - match SubprocessWorker's isolation level
    # Filter out conflicting paths from conda/mamba/etc and use proper DLL registration
    path_sep = ";" if sys.platform == "win32" else ":"

    if sys.platform == "win32":
        # Use os.add_dll_directory() for explicit DLL registration (Python 3.8+)
        if lib_path and hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_path)
            except Exception:
                pass

        # Filter conflicting paths from PATH (matches subprocess.py:1203-1212)
        current_path = os.environ.get("PATH", "")
        clean_parts = [
            p for p in current_path.split(path_sep)
            if not any(x in p.lower() for x in (".ct-envs", "conda", "mamba", "miniforge", "miniconda", "anaconda", "mingw"))
        ]
        if lib_path:
            clean_parts.insert(0, lib_path)
        os.environ["PATH"] = path_sep.join(clean_parts)
    elif sys.platform == "darwin":
        # macOS: ONLY use the isolated lib_path, don't inherit
        if lib_path:
            os.environ["DYLD_LIBRARY_PATH"] = lib_path
        else:
            os.environ.pop("DYLD_LIBRARY_PATH", None)
    else:
        # Linux: Use LD_LIBRARY_PATH
        current = os.environ.get("LD_LIBRARY_PATH", "")
        clean_parts = [
            p for p in current.split(path_sep) if p
            and not any(x in p.lower() for x in (".ct-envs", "conda", "mamba", "miniforge", "miniconda", "anaconda"))
        ]
        if lib_path:
            clean_parts.insert(0, lib_path)
        os.environ["LD_LIBRARY_PATH"] = path_sep.join(clean_parts)

    # Find ComfyUI base and add to sys.path for real folder_paths/comfy modules
    # This works because comfy.options.args_parsing=False by default, so folder_paths
    # auto-detects its base directory from __file__ location
    def _find_comfyui_base():
        cwd = Path.cwd().resolve()
        # Check common child directories (for test environments)
        for base in [cwd, cwd.parent]:
            for child in [".comfy-test-env/ComfyUI", "ComfyUI"]:
                candidate = base / child
                if (candidate / "main.py").exists() and (candidate / "comfy").exists():
                    return candidate
        # Walk up from cwd looking for ComfyUI
        current = cwd
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            current = current.parent
        # Check COMFYUI_BASE env var as fallback
        if os.environ.get("COMFYUI_BASE"):
            return Path(os.environ["COMFYUI_BASE"])
        return None

    comfyui_base = _find_comfyui_base()
    if comfyui_base and str(comfyui_base) not in sys.path:
        sys.path.insert(0, str(comfyui_base))

    # Add custom paths to sys.path for module discovery
    if sys_path_additions:
        for path in sys_path_additions:
            if path not in sys.path:
                sys.path.insert(0, path)

    while True:
        try:
            item = queue_in.get()

            # Check for shutdown signal
            if item is _SHUTDOWN:
                queue_out.put(("shutdown", None))
                break

            try:
                # Handle method call protocol
                if isinstance(item, tuple) and len(item) == 6 and item[0] == _CALL_METHOD:
                    _, module_name, class_name, method_name, self_state, kwargs = item
                    # Load tensors from files (saved by host to avoid cudaMallocAsync IPC issues)
                    kwargs = _load_tensors_from_files(kwargs)
                    result = _execute_method_call(
                        module_name, class_name, method_name, self_state, kwargs
                    )
                    # Save tensors to files to avoid CUDA IPC issues with cudaMallocAsync
                    result = _save_tensors_to_files(result)
                    queue_out.put(("ok", result))
                else:
                    # Direct function call (legacy)
                    func, args, kwargs = item
                    result = func(*args, **kwargs)
                    # Save tensors to files to avoid CUDA IPC issues with cudaMallocAsync
                    result = _save_tensors_to_files(result)
                    queue_out.put(("ok", result))

            except Exception as e:
                tb = traceback.format_exc()
                queue_out.put(("error", (str(e), tb)))

        except Exception as e:
            # Queue error - try to report, then exit
            try:
                queue_out.put(("fatal", str(e)))
            except:
                pass
            break


class PathBasedModuleFinder:
    """
    Meta path finder that handles ComfyUI's path-based module names.

    ComfyUI uses full filesystem paths as module names for custom nodes.
    This finder intercepts imports of such modules and loads them from disk.
    """

    def find_spec(self, fullname, path, target=None):
        import importlib.util
        import os

        # Only handle path-based module names (starting with /)
        if not fullname.startswith('/'):
            return None

        # Parse the module name to find base path and submodule parts
        parts = fullname.split('.')
        base_path = parts[0]
        submodule_parts = parts[1:] if len(parts) > 1 else []

        # Walk through parts to find where path ends and module begins
        for i, part in enumerate(submodule_parts):
            test_path = os.path.join(base_path, part)
            if os.path.exists(test_path):
                base_path = test_path
            else:
                # Remaining parts are module names
                submodule_parts = submodule_parts[i:]
                break
        else:
            # All parts were path components
            submodule_parts = []

        # Determine the file to load
        if submodule_parts:
            # We're importing a submodule
            current_path = base_path
            for part in submodule_parts[:-1]:
                current_path = os.path.join(current_path, part)

            submod = submodule_parts[-1]
            submod_file = os.path.join(current_path, submod + '.py')
            submod_pkg = os.path.join(current_path, submod, '__init__.py')

            if os.path.exists(submod_file):
                return importlib.util.spec_from_file_location(fullname, submod_file)
            elif os.path.exists(submod_pkg):
                return importlib.util.spec_from_file_location(
                    fullname, submod_pkg,
                    submodule_search_locations=[os.path.join(current_path, submod)]
                )
        else:
            # Top-level path-based module
            if os.path.isdir(base_path):
                init_path = os.path.join(base_path, "__init__.py")
                if os.path.exists(init_path):
                    return importlib.util.spec_from_file_location(
                        fullname, init_path,
                        submodule_search_locations=[base_path]
                    )
            elif os.path.isfile(base_path):
                return importlib.util.spec_from_file_location(fullname, base_path)

        return None


# Global flag to track if we've installed the finder
_path_finder_installed = False


def _ensure_path_finder_installed():
    """Install the PathBasedModuleFinder if not already installed."""
    import sys
    global _path_finder_installed
    if not _path_finder_installed:
        sys.meta_path.insert(0, PathBasedModuleFinder())
        _path_finder_installed = True
        logger.debug("[comfy_env] Installed PathBasedModuleFinder for path-based module names")


def _load_path_based_module(module_name: str):
    """
    Load a module that has a filesystem path as its name.

    ComfyUI uses full filesystem paths as module names for custom nodes.
    This function handles that case by using file-based imports.
    """
    import importlib.util
    import os
    import sys

    # Check if it's already in sys.modules
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Check if module_name contains submodule parts (e.g., "/path/to/pkg.submod.subsubmod")
    # In this case, we need to load the parent packages first
    if '.' in module_name:
        parts = module_name.split('.')
        # Find where the path ends and module parts begin
        # The path part won't exist as a directory when combined with module parts
        base_path = parts[0]
        submodule_parts = []

        for i, part in enumerate(parts[1:], 1):
            test_path = os.path.join(base_path, part)
            if os.path.exists(test_path):
                base_path = test_path
            else:
                # This and remaining parts are module names, not path components
                submodule_parts = parts[i:]
                break

        if submodule_parts:
            # Load parent package first
            parent_module = _load_path_based_module(base_path)

            # Now load submodules
            current_module = parent_module
            current_name = base_path
            for submod in submodule_parts:
                current_name = f"{current_name}.{submod}"
                if current_name in sys.modules:
                    current_module = sys.modules[current_name]
                else:
                    # Try to import as attribute or load from file
                    if hasattr(current_module, submod):
                        current_module = getattr(current_module, submod)
                    else:
                        # Try to load the submodule file
                        if hasattr(current_module, '__path__'):
                            for parent_path in current_module.__path__:
                                submod_file = os.path.join(parent_path, submod + '.py')
                                submod_pkg = os.path.join(parent_path, submod, '__init__.py')
                                if os.path.exists(submod_file):
                                    spec = importlib.util.spec_from_file_location(current_name, submod_file)
                                    current_module = importlib.util.module_from_spec(spec)
                                    current_module.__package__ = f"{base_path}.{'.'.join(submodule_parts[:-1])}" if len(submodule_parts) > 1 else base_path
                                    sys.modules[current_name] = current_module
                                    spec.loader.exec_module(current_module)
                                    break
                                elif os.path.exists(submod_pkg):
                                    spec = importlib.util.spec_from_file_location(current_name, submod_pkg,
                                        submodule_search_locations=[os.path.dirname(submod_pkg)])
                                    current_module = importlib.util.module_from_spec(spec)
                                    sys.modules[current_name] = current_module
                                    spec.loader.exec_module(current_module)
                                    break
                        else:
                            raise ModuleNotFoundError(f"Cannot find submodule {submod} in {current_name}")
            return current_module

    # Simple path-based module (no submodule parts)
    if os.path.isdir(module_name):
        init_path = os.path.join(module_name, "__init__.py")
        submodule_search_locations = [module_name]
    else:
        init_path = module_name
        submodule_search_locations = None

    if not os.path.exists(init_path):
        raise ModuleNotFoundError(f"Cannot find module at path: {module_name}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=submodule_search_locations
    )
    module = importlib.util.module_from_spec(spec)

    # Set up package attributes for relative imports
    if os.path.isdir(module_name):
        module.__path__ = [module_name]
        module.__package__ = module_name
    else:
        module.__package__ = module_name.rsplit('.', 1)[0] if '.' in module_name else ''

    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _execute_method_call(module_name: str, class_name: str, method_name: str,
                         self_state: dict, kwargs: dict) -> Any:
    """
    Execute a method call in the worker process.

    This function imports the class fresh and calls the original (un-decorated) method.
    """
    import importlib
    import os
    import sys

    # Import the module
    logger.debug(f"Attempting to import module_name={module_name}")

    # Check if module_name is a filesystem path (ComfyUI uses paths as module names)
    # This happens because ComfyUI's load_custom_node uses the full path as sys_module_name
    if module_name.startswith('/') or (os.sep in module_name and not module_name.startswith('.')):
        # Check if the base path exists to confirm it's a path-based module
        base_path = module_name.split('.')[0] if '.' in module_name else module_name
        if os.path.exists(base_path):
            logger.debug(f"Detected path-based module name, using file-based import")
            # Install the meta path finder to handle relative imports within the package
            _ensure_path_finder_installed()
            module = _load_path_based_module(module_name)
        else:
            # Doesn't look like a valid path, try standard import
            module = importlib.import_module(module_name)
    else:
        # Standard module name - use importlib.import_module
        module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Create instance with proper __slots__ handling
    instance = object.__new__(cls)

    # Handle both __slots__ and __dict__ based classes
    if hasattr(cls, '__slots__'):
        # Class uses __slots__ - set attributes individually
        for slot in cls.__slots__:
            if slot in self_state:
                setattr(instance, slot, self_state[slot])
        # Also check for __dict__ slot (hybrid classes)
        if '__dict__' in cls.__slots__ or hasattr(instance, '__dict__'):
            for key, value in self_state.items():
                if key not in cls.__slots__:
                    setattr(instance, key, value)
    else:
        # Standard class with __dict__
        instance.__dict__.update(self_state)

    # Get the ORIGINAL method stored by the decorator, not the proxy
    # This avoids the infinite recursion of proxy -> worker -> proxy
    original_method = getattr(cls, '_isolated_original_method', None)
    if original_method is None:
        # Fallback: class wasn't decorated, use the method directly
        original_method = getattr(cls, method_name)
        return original_method(instance, **kwargs)

    # Call the original method (it's an unbound function, pass instance)
    return original_method(instance, **kwargs)


class MPWorker(Worker):
    """
    Worker using torch.multiprocessing for same-venv isolation.

    Features:
    - Zero-copy CUDA tensor transfer (via CUDA IPC handles)
    - Zero-copy CPU tensor transfer (via shared memory)
    - Fresh CUDA context (subprocess has independent GPU state)
    - Automatic cleanup on worker death

    The subprocess uses 'spawn' start method, ensuring a clean Python
    interpreter without inherited state from the parent.
    """

    def __init__(self, name: Optional[str] = None, sys_path: Optional[list] = None, lib_path: Optional[str] = None, env_vars: Optional[dict] = None):
        """
        Initialize the worker.

        Args:
            name: Optional name for logging/debugging.
            sys_path: Optional list of paths to add to sys.path in worker process.
            lib_path: Optional path to add to LD_LIBRARY_PATH (for conda libraries).
            env_vars: Optional environment variables to set in worker process.
        """
        self.name = name or "MPWorker"
        self._sys_path = sys_path or []
        self._lib_path = lib_path
        self._env_vars = env_vars or {}
        self._process = None
        self._queue_in = None
        self._queue_out = None
        self._started = False
        self._shutdown = False

    def _ensure_started(self):
        """Lazily start the worker process on first call."""
        if self._shutdown:
            raise RuntimeError(f"{self.name}: Worker has been shut down")

        if self._started:
            if not self._process.is_alive():
                raise RuntimeError(f"{self.name}: Worker process died unexpectedly")
            return

        # Import torch here to avoid import at module level
        import os
        import sys

        # Clear conda/pixi environment variables FIRST, before importing multiprocessing
        # These can cause the child process to pick up the wrong Python interpreter
        # or stdlib, leading to sys.version mismatch errors in platform module
        conda_env_vars = [
            'CONDA_PREFIX',
            'CONDA_DEFAULT_ENV',
            'CONDA_PYTHON_EXE',
            'CONDA_EXE',
            'CONDA_SHLVL',
            'PYTHONHOME',
            'PYTHONPATH',  # Also clear PYTHONPATH to prevent pixi paths
            '_CE_CONDA',
            '_CE_M',
        ]
        saved_env = {}
        for var in conda_env_vars:
            if var in os.environ:
                saved_env[var] = os.environ.pop(var)

        # Also remove pixi paths from LD_LIBRARY_PATH
        ld_lib = os.environ.get('LD_LIBRARY_PATH', '')
        if '.pixi' in ld_lib:
            saved_env['LD_LIBRARY_PATH'] = ld_lib
            # Filter out pixi paths
            new_ld_lib = ':'.join(p for p in ld_lib.split(':') if '.pixi' not in p)
            if new_ld_lib:
                os.environ['LD_LIBRARY_PATH'] = new_ld_lib
            else:
                os.environ.pop('LD_LIBRARY_PATH', None)

        import torch.multiprocessing as mp

        try:
            # Use spawn to get clean subprocess (no inherited CUDA context)
            ctx = mp.get_context('spawn')

            # Explicitly set the spawn executable to the current Python
            # This prevents pixi/conda from hijacking the spawn process
            import multiprocessing.spawn as mp_spawn
            original_exe = mp_spawn.get_executable()
            if original_exe != sys.executable.encode() and original_exe != sys.executable:
                print(f"[comfy-env] Warning: spawn executable was {original_exe}, forcing to {sys.executable}")
            mp_spawn.set_executable(sys.executable)

            self._queue_in = ctx.Queue()
            self._queue_out = ctx.Queue()
            self._process = ctx.Process(
                target=_worker_loop,
                args=(self._queue_in, self._queue_out, self._sys_path, self._lib_path, self._env_vars, self.name),
                daemon=True,
            )
            self._process.start()
            self._started = True

            # Restore original executable setting
            mp_spawn.set_executable(original_exe)
        finally:
            # Restore env vars in parent process
            os.environ.update(saved_env)

    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function in the worker process.

        Args:
            func: Function to execute. Must be picklable (module-level or staticmethod).
            *args: Positional arguments.
            timeout: Timeout in seconds (None = no timeout, default).
            **kwargs: Keyword arguments.

        Returns:
            Return value of func(*args, **kwargs).

        Raises:
            WorkerError: If func raises an exception.
            TimeoutError: If execution exceeds timeout.
            RuntimeError: If worker process dies.
        """
        self._ensure_started()

        # Send work item
        self._queue_in.put((func, args, kwargs))

        return self._get_result(timeout)

    def call_method(
        self,
        module_name: str,
        class_name: str,
        method_name: str,
        self_state: dict,
        kwargs: dict,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute a class method in the worker process.

        This uses a string-based protocol to avoid pickle issues with decorated methods.
        The worker imports the module fresh and calls the original (un-decorated) method.

        Args:
            module_name: Full module path (e.g., 'my_package.nodes.my_node')
            class_name: Class name (e.g., 'MyNode')
            method_name: Method name (e.g., 'process')
            self_state: Instance __dict__ to restore
            kwargs: Method keyword arguments
            timeout: Timeout in seconds (None = no timeout, default).

        Returns:
            Return value of method.

        Raises:
            WorkerError: If method raises an exception.
            TimeoutError: If execution exceeds timeout.
            RuntimeError: If worker process dies.
        """
        self._ensure_started()

        # Save tensors to files to avoid CUDA IPC issues with cudaMallocAsync
        kwargs = _save_tensors_to_files(kwargs)

        # Send method call request using protocol
        self._queue_in.put((
            _CALL_METHOD,
            module_name,
            class_name,
            method_name,
            self_state,
            kwargs,
        ))

        return self._get_result(timeout)

    def _get_result(self, timeout: Optional[float]) -> Any:
        """Wait for and return result from worker."""
        try:
            status, result = self._queue_out.get(timeout=timeout)
        except QueueEmpty:
            # Timeout - use graceful escalation
            self._handle_timeout(timeout)
            # _handle_timeout always raises, but just in case:
            raise TimeoutError(f"{self.name}: Call timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"{self.name}: Failed to get result: {e}")

        # Handle response
        if status == "ok":
            # Load tensors from temp files
            result = _load_tensors_from_files(result)
            return result
        elif status == "error":
            msg, tb = result
            raise WorkerError(msg, traceback=tb)
        elif status == "fatal":
            self._shutdown = True
            raise RuntimeError(f"{self.name}: Fatal worker error: {result}")
        else:
            raise RuntimeError(f"{self.name}: Unknown response status: {status}")

    def shutdown(self) -> None:
        """Shut down the worker process."""
        if self._shutdown or not self._started:
            return

        self._shutdown = True

        try:
            # Send shutdown signal
            self._queue_in.put(_SHUTDOWN)

            # Wait for acknowledgment
            try:
                self._queue_out.get(timeout=5.0)
            except:
                pass

            # Wait for process to exit
            self._process.join(timeout=5.0)

            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)

        except Exception:
            # Force kill if anything goes wrong
            if self._process and self._process.is_alive():
                self._process.kill()

    def _handle_timeout(self, timeout: float) -> None:
        """
        Handle timeout with graceful escalation.

        Instead of immediately killing the worker (which can leak GPU memory),
        try graceful shutdown first, then escalate to SIGTERM, then SIGKILL.

        Inspired by pyisolate's timeout handling pattern.
        """
        logger.warning(f"{self.name}: Call timed out after {timeout}s, attempting graceful shutdown")

        # Stage 1: Send shutdown signal, wait 3s for graceful exit
        try:
            self._queue_in.put(_SHUTDOWN)
            self._queue_out.get(timeout=3.0)
            self._process.join(timeout=2.0)
            if not self._process.is_alive():
                self._shutdown = True
                raise TimeoutError(f"{self.name}: Graceful shutdown after timeout ({timeout}s)")
        except QueueEmpty:
            pass
        except TimeoutError:
            raise
        except Exception:
            pass

        # Stage 2: SIGTERM, wait 5s
        if self._process.is_alive():
            logger.warning(f"{self.name}: Graceful shutdown failed, sending SIGTERM")
            self._process.terminate()
            self._process.join(timeout=5.0)

        # Stage 3: SIGKILL as last resort
        if self._process.is_alive():
            logger.error(f"{self.name}: SIGTERM failed, force killing worker (may leak GPU memory)")
            self._process.kill()
            self._process.join(timeout=1.0)

        self._shutdown = True
        raise TimeoutError(f"{self.name}: Call timed out after {timeout}s")

    def is_alive(self) -> bool:
        """Check if worker process is running or can be started."""
        if self._shutdown:
            return False
        # Not started yet = can still be started = "alive"
        if not self._started:
            return True
        return self._process.is_alive()

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"<MPWorker name={self.name!r} status={status}>"
