"""
VenvWorker - Cross-venv isolation using subprocess + shared memory.

This worker supports calling functions in a different Python environment:
- Uses subprocess.Popen to run in different venv
- Transfers tensors via torch.save/load through /dev/shm (RAM-backed)
- One memcpy per tensor per direction
- ~100-500ms overhead per call (subprocess spawn + tensor I/O)

Use this when you need:
- Different PyTorch version
- Incompatible native library dependencies
- Different Python version

Example:
    worker = VenvWorker(
        python="/path/to/other/venv/bin/python",
        working_dir="/path/to/code",
    )

    # Call a function by module path
    result = worker.call_module(
        module="my_module",
        func="my_function",
        image=my_tensor,
    )
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import Worker, WorkerError


def _serialize_for_ipc(obj, visited=None):
    """
    Convert objects with broken __module__ paths to dicts for IPC.

    ComfyUI sets weird __module__ values (file paths) on custom node classes,
    which breaks pickle deserialization in the worker. This converts such
    objects to a serializable dict format.
    """
    if visited is None:
        visited = {}  # Maps id -> serialized result

    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]  # Return cached serialized result

    # Check if this is a custom object with broken module path
    if (hasattr(obj, '__dict__') and
        hasattr(obj, '__class__') and
        not isinstance(obj, (dict, list, tuple, type)) and
        obj.__class__.__name__ not in ('Tensor', 'ndarray', 'module')):

        cls = obj.__class__
        module = getattr(cls, '__module__', '')

        # Check if module looks like a file path or is problematic for pickling
        # This catches: file paths, custom_nodes imports, and modules starting with /
        is_problematic = (
            '/' in module or
            '\\' in module or
            module.startswith('/') or
            'custom_nodes' in module or
            module == '' or
            module == '__main__'
        )
        if is_problematic:
            # Convert to serializable dict and cache it
            result = {
                '__isolated_object__': True,
                '__class_name__': cls.__name__,
                '__attrs__': {k: _serialize_for_ipc(v, visited) for k, v in obj.__dict__.items()},
            }
            visited[obj_id] = result
            return result

    # Recurse into containers
    if isinstance(obj, dict):
        result = {k: _serialize_for_ipc(v, visited) for k, v in obj.items()}
        visited[obj_id] = result
        return result
    elif isinstance(obj, list):
        result = [_serialize_for_ipc(v, visited) for v in obj]
        visited[obj_id] = result
        return result
    elif isinstance(obj, tuple):
        result = tuple(_serialize_for_ipc(v, visited) for v in obj)
        visited[obj_id] = result
        return result

    # Primitives and other objects - cache and return as-is
    visited[obj_id] = obj
    return obj


# Worker script template - minimal, runs in target venv
_WORKER_SCRIPT = '''
import sys
import json
import traceback
from types import SimpleNamespace

def _deserialize_isolated_objects(obj):
    """Reconstruct objects serialized with __isolated_object__ marker."""
    if isinstance(obj, dict):
        if obj.get("__isolated_object__"):
            # Reconstruct as SimpleNamespace (supports .attr access)
            attrs = {k: _deserialize_isolated_objects(v) for k, v in obj.get("__attrs__", {}).items()}
            ns = SimpleNamespace(**attrs)
            ns.__class_name__ = obj.get("__class_name__", "Unknown")
            return ns
        return {k: _deserialize_isolated_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_isolated_objects(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_isolated_objects(v) for v in obj)
    return obj

def main():
    # Read request from file
    request_path = sys.argv[1]
    response_path = sys.argv[2]

    with open(request_path, 'r') as f:
        request = json.load(f)

    try:
        # Setup paths
        for p in request.get("sys_path", []):
            if p not in sys.path:
                sys.path.insert(0, p)

        # Import torch for tensor I/O
        import torch

        # Load inputs
        inputs_path = request.get("inputs_path")
        if inputs_path:
            inputs = torch.load(inputs_path, weights_only=False)
            inputs = _deserialize_isolated_objects(inputs)
        else:
            inputs = {}

        # Import and call function
        module_name = request["module"]
        func_name = request["func"]

        module = __import__(module_name, fromlist=[func_name])
        func = getattr(module, func_name)

        result = func(**inputs)

        # Save outputs
        outputs_path = request.get("outputs_path")
        if outputs_path:
            torch.save(result, outputs_path)

        response = {"status": "ok"}

    except Exception as e:
        response = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    with open(response_path, 'w') as f:
        json.dump(response, f)

if __name__ == "__main__":
    main()
'''


def _get_shm_dir() -> Path:
    """Get shared memory directory for efficient tensor transfer."""
    # Linux: /dev/shm is RAM-backed tmpfs
    if sys.platform == 'linux' and os.path.isdir('/dev/shm'):
        return Path('/dev/shm')
    # Fallback to regular temp
    return Path(tempfile.gettempdir())


class VenvWorker(Worker):
    """
    Worker using subprocess for cross-venv isolation.

    This worker spawns a new Python process for each call, using
    a different Python interpreter (from another venv). Tensors are
    transferred via torch.save/load through shared memory.

    For long-running workloads, consider using persistent mode which
    keeps the subprocess alive between calls.
    """

    def __init__(
        self,
        python: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        sys_path: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        persistent: bool = True,
    ):
        """
        Initialize the worker.

        Args:
            python: Path to Python executable in target venv.
            working_dir: Working directory for subprocess.
            sys_path: Additional paths to add to sys.path in subprocess.
            env: Additional environment variables.
            name: Optional name for logging.
            persistent: If True, keep subprocess alive between calls (faster).
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"VenvWorker({self.python.parent.parent.name})"
        self.persistent = persistent

        # Verify Python exists
        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        # Create temp directory for IPC files
        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_venv_'))
        self._shm_dir = _get_shm_dir()

        # Persistent process state
        self._process: Optional[subprocess.Popen] = None
        self._shutdown = False

        # Write worker script
        self._worker_script = self._temp_dir / "worker.py"
        self._worker_script.write_text(_WORKER_SCRIPT)

    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function - NOT SUPPORTED for VenvWorker.

        VenvWorker cannot pickle arbitrary functions across venv boundaries.
        Use call_module() instead to call functions by module path.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{self.name}: VenvWorker cannot call arbitrary functions. "
            f"Use call_module(module='...', func='...', **kwargs) instead."
        )

    def call_module(
        self,
        module: str,
        func: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Call a function by module path in the isolated venv.

        Args:
            module: Module name (e.g., "my_package.my_module").
            func: Function name within the module.
            timeout: Timeout in seconds (None = 600s default).
            **kwargs: Keyword arguments passed to the function.
                     Must be torch.save-compatible (tensors, dicts, etc.).

        Returns:
            Return value of module.func(**kwargs).

        Raises:
            WorkerError: If function raises an exception.
            TimeoutError: If execution exceeds timeout.
        """
        if self._shutdown:
            raise RuntimeError(f"{self.name}: Worker has been shut down")

        timeout = timeout or 600.0  # 10 minute default

        # Create unique ID for this call
        call_id = str(uuid.uuid4())[:8]

        # Paths for IPC (use shm for tensors, temp for json)
        inputs_path = self._shm_dir / f"comfyui_venv_{call_id}_in.pt"
        outputs_path = self._shm_dir / f"comfyui_venv_{call_id}_out.pt"
        request_path = self._temp_dir / f"request_{call_id}.json"
        response_path = self._temp_dir / f"response_{call_id}.json"

        try:
            # Save inputs via torch.save (handles tensors natively)
            # Serialize custom objects with broken __module__ paths first
            import torch
            if kwargs:
                serialized_kwargs = _serialize_for_ipc(kwargs)
                torch.save(serialized_kwargs, str(inputs_path))

            # Build request
            request = {
                "module": module,
                "func": func,
                "sys_path": [str(self.working_dir)] + self.sys_path,
                "inputs_path": str(inputs_path) if kwargs else None,
                "outputs_path": str(outputs_path),
            }

            request_path.write_text(json.dumps(request))

            # Build environment
            env = os.environ.copy()
            env.update(self.extra_env)
            env["COMFYUI_ISOLATION_WORKER"] = "1"

            # Run subprocess
            cmd = [
                str(self.python),
                str(self._worker_script),
                str(request_path),
                str(response_path),
            ]

            process = subprocess.Popen(
                cmd,
                cwd=str(self.working_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise TimeoutError(f"{self.name}: Call timed out after {timeout}s")

            # Check for process error
            if process.returncode != 0:
                raise WorkerError(
                    f"Subprocess failed with code {process.returncode}",
                    traceback=stderr.decode('utf-8', errors='replace'),
                )

            # Read response
            if not response_path.exists():
                raise WorkerError(
                    f"No response file",
                    traceback=stderr.decode('utf-8', errors='replace'),
                )

            response = json.loads(response_path.read_text())

            if response["status"] == "error":
                raise WorkerError(
                    response.get("error", "Unknown error"),
                    traceback=response.get("traceback"),
                )

            # Load result
            if outputs_path.exists():
                result = torch.load(str(outputs_path), weights_only=False)
                return result
            else:
                return None

        finally:
            # Cleanup IPC files
            for path in [inputs_path, outputs_path, request_path, response_path]:
                try:
                    if path.exists():
                        path.unlink()
                except:
                    pass

    def shutdown(self) -> None:
        """Shut down the worker and clean up resources."""
        if self._shutdown:
            return

        self._shutdown = True

        # Clean up temp directory
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass

    def is_alive(self) -> bool:
        """VenvWorker spawns fresh process per call, so always 'alive' if not shutdown."""
        return not self._shutdown

    def __repr__(self):
        return f"<VenvWorker name={self.name!r} python={self.python}>"


def _queue_worker_entrypoint(to_host_queue, from_host_queue, shm_dir, sys_paths):
    """
    Worker process entrypoint using queue-based IPC.

    This function runs in the isolated venv subprocess and handles
    requests from the host process.

    Args:
        to_host_queue: Queue for sending messages to host (worker -> host)
        from_host_queue: Queue for receiving messages from host (host -> worker)
        shm_dir: Shared memory directory for tensor files
        sys_paths: List of paths to add to sys.path
    """
    import sys
    import traceback
    from pathlib import Path
    from types import SimpleNamespace

    # Setup paths
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    def _deserialize_isolated_objects(obj):
        """Reconstruct objects serialized with __isolated_object__ marker."""
        if isinstance(obj, dict):
            if obj.get("__isolated_object__"):
                attrs = {k: _deserialize_isolated_objects(v) for k, v in obj.get("__attrs__", {}).items()}
                ns = SimpleNamespace(**attrs)
                ns.__class_name__ = obj.get("__class_name__", "Unknown")
                return ns
            return {k: _deserialize_isolated_objects(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_deserialize_isolated_objects(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(_deserialize_isolated_objects(v) for v in obj)
        return obj

    # Signal ready
    to_host_queue.put({"status": "ready"})

    import torch

    while True:
        try:
            request = from_host_queue.get()
        except Exception:
            break

        if request is None or request.get("method") == "shutdown":
            break

        try:
            request_type = request.get("type", "call_module")
            module_name = request["module"]
            inputs_path = request.get("inputs_path")
            outputs_path = request.get("outputs_path")

            # Load inputs
            if inputs_path:
                inputs = torch.load(inputs_path, weights_only=False)
                inputs = _deserialize_isolated_objects(inputs)
            else:
                inputs = {}

            # Import module
            module = __import__(module_name, fromlist=[""])

            if request_type == "call_method":
                # Call a method on a class instance
                class_name = request["class_name"]
                method_name = request["method_name"]
                self_state = request.get("self_state")

                cls = getattr(module, class_name)
                instance = object.__new__(cls)
                if self_state:
                    instance.__dict__.update(self_state)
                method = getattr(instance, method_name)
                result = method(**inputs)
            else:
                # Call a module-level function
                func_name = request["func"]
                func = getattr(module, func_name)
                result = func(**inputs)

            # Save result
            if outputs_path:
                torch.save(result, outputs_path)

            to_host_queue.put({"status": "ok"})

        except Exception as e:
            to_host_queue.put({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


class PersistentVenvWorker(Worker):
    """
    Persistent version of VenvWorker that keeps subprocess alive.

    This reduces per-call overhead by ~200-400ms by avoiding subprocess spawn.
    Uses multiprocessing.Queue for cross-platform communication with zero-copy
    tensor transfer support.

    Use this for high-frequency calls to the same venv.
    """

    def __init__(
        self,
        python: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        sys_path: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        share_torch: bool = True,
    ):
        """
        Initialize persistent worker.

        Args:
            python: Path to Python executable in target venv.
            working_dir: Working directory for subprocess.
            sys_path: Additional paths to add to sys.path.
            env: Additional environment variables.
            name: Optional name for logging.
            share_torch: Use torch.multiprocessing for zero-copy tensor transfer.
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"PersistentVenvWorker({self.python.parent.parent.name})"
        self.share_torch = share_torch

        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_pvenv_'))
        self._shm_dir = _get_shm_dir()
        self._process = None  # Will be multiprocessing.Process
        self._to_worker = None  # Queue: host -> worker
        self._from_worker = None  # Queue: worker -> host
        self._transport = None
        self._shutdown = False
        self._lock = threading.Lock()
        self._mp = None  # multiprocessing module reference

    def _find_comfyui_base(self) -> Optional[Path]:
        """Find ComfyUI base directory by walking up from working_dir."""
        current = self.working_dir.resolve()
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            current = current.parent
        return None

    def _ensure_started(self):
        """Start persistent worker process if not running."""
        if self._shutdown:
            raise RuntimeError(f"{self.name}: Worker has been shut down")

        if self._process is not None and self._process.is_alive():
            return  # Already running

        from ..ipc.transport import create_queue_pair, QueueTransport

        # Create queue pair for bidirectional communication
        self._to_worker, self._from_worker, self._mp = create_queue_pair(
            share_torch=self.share_torch
        )

        # Set up environment
        env = os.environ.copy()
        env.update(self.extra_env)
        env["COMFYUI_ISOLATION_WORKER"] = "1"

        # Find ComfyUI base and set env var for folder_paths stub
        comfyui_base = self._find_comfyui_base()
        if comfyui_base:
            env["COMFYUI_BASE"] = str(comfyui_base)

        # Add stubs directory to sys_path for folder_paths etc.
        stubs_dir = Path(__file__).parent.parent / "stubs"
        all_sys_path = [str(stubs_dir), str(self.working_dir)] + self.sys_path

        # On Windows, set VIRTUAL_ENV for proper venv activation
        if os.name == "nt":
            env["VIRTUAL_ENV"] = str(self.python.parent.parent)

        # Set the Python executable for multiprocessing
        self._mp.set_executable(str(self.python))

        # Launch worker process
        self._process = self._mp.Process(
            target=_queue_worker_entrypoint,
            args=(
                self._to_worker,
                self._from_worker,
                str(self._shm_dir),
                all_sys_path,
            ),
        )
        self._process.start()

        # Create transport wrapper
        self._transport = QueueTransport(self._to_worker, self._from_worker)

        # Wait for ready signal
        try:
            msg = self._transport.recv(timeout=30)
        except Exception:
            if self._process.is_alive():
                self._process.terminate()
            raise RuntimeError(f"{self.name}: Worker failed to start (timeout)")

        if msg.get("status") != "ready":
            raise RuntimeError(f"{self.name}: Unexpected ready message: {msg}")

    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Not supported - use call_module()."""
        raise NotImplementedError(
            f"{self.name}: Use call_module(module='...', func='...') instead."
        )

    def call_method(
        self,
        module_name: str,
        class_name: str,
        method_name: str,
        self_state: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Call a class method by module/class/method path.

        Args:
            module_name: Module containing the class (e.g., "depth_estimate").
            class_name: Class name (e.g., "SAM3D_DepthEstimate").
            method_name: Method name (e.g., "estimate_depth").
            self_state: Optional dict to populate instance __dict__.
            kwargs: Keyword arguments for the method.
            timeout: Timeout in seconds.

        Returns:
            Return value of the method.
        """
        import queue as queue_module

        with self._lock:
            self._ensure_started()

            timeout = timeout or 600.0
            call_id = str(uuid.uuid4())[:8]

            import torch
            inputs_path = self._shm_dir / f"comfyui_pvenv_{call_id}_in.pt"
            outputs_path = self._shm_dir / f"comfyui_pvenv_{call_id}_out.pt"

            try:
                # Serialize kwargs
                if kwargs:
                    serialized_kwargs = _serialize_for_ipc(kwargs)
                    torch.save(serialized_kwargs, str(inputs_path))

                # Send request with class info
                request = {
                    "type": "call_method",
                    "module": module_name,
                    "class_name": class_name,
                    "method_name": method_name,
                    "self_state": self_state,
                    "inputs_path": str(inputs_path) if kwargs else None,
                    "outputs_path": str(outputs_path),
                }
                self._transport.send(request)

                # Wait for response with timeout
                try:
                    response = self._transport.recv(timeout=timeout)
                except queue_module.Empty:
                    self._process.terminate()
                    self._shutdown = True
                    raise TimeoutError(f"{self.name}: Call timed out")

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                if outputs_path.exists():
                    return torch.load(str(outputs_path), weights_only=False)
                return None

            finally:
                for p in [inputs_path, outputs_path]:
                    try:
                        p.unlink()
                    except:
                        pass

    def call_module(
        self,
        module: str,
        func: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Call a function by module path."""
        import queue as queue_module

        with self._lock:
            self._ensure_started()

            timeout = timeout or 600.0
            call_id = str(uuid.uuid4())[:8]

            # Save inputs
            import torch
            inputs_path = self._shm_dir / f"comfyui_pvenv_{call_id}_in.pt"
            outputs_path = self._shm_dir / f"comfyui_pvenv_{call_id}_out.pt"

            try:
                if kwargs:
                    serialized_kwargs = _serialize_for_ipc(kwargs)
                    torch.save(serialized_kwargs, str(inputs_path))

                # Send request
                request = {
                    "type": "call_module",
                    "module": module,
                    "func": func,
                    "inputs_path": str(inputs_path) if kwargs else None,
                    "outputs_path": str(outputs_path),
                }
                self._transport.send(request)

                # Wait for response with timeout
                try:
                    response = self._transport.recv(timeout=timeout)
                except queue_module.Empty:
                    self._process.terminate()
                    self._shutdown = True
                    raise TimeoutError(f"{self.name}: Call timed out")

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                # Load result
                if outputs_path.exists():
                    return torch.load(str(outputs_path), weights_only=False)
                return None

            finally:
                for p in [inputs_path, outputs_path]:
                    try:
                        p.unlink()
                    except:
                        pass

    def shutdown(self) -> None:
        """Shut down the persistent worker."""
        if self._shutdown:
            return
        self._shutdown = True

        # Send shutdown signal via queue
        if self._transport:
            try:
                self._transport.send({"method": "shutdown"})
            except:
                pass
            self._transport.close()

        # Wait for process to exit
        if self._process:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)

        # Clean up queues
        if self._to_worker:
            try:
                self._to_worker.close()
            except:
                pass
        if self._from_worker:
            try:
                self._from_worker.close()
            except:
                pass

        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def is_alive(self) -> bool:
        if self._shutdown:
            return False
        if self._process is None:
            return False
        return self._process.is_alive()

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"<PersistentVenvWorker name={self.name!r} status={status}>"
