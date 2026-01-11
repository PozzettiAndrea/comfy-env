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
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        return obj
    visited.add(obj_id)

    # Check if this is a custom object with broken module path
    if (hasattr(obj, '__dict__') and
        hasattr(obj, '__class__') and
        not isinstance(obj, (dict, list, tuple, type)) and
        obj.__class__.__name__ not in ('Tensor', 'ndarray', 'module')):

        cls = obj.__class__
        module = getattr(cls, '__module__', '')

        # Check if module looks like a file path (contains / or \)
        if '/' in module or '\\' in module:
            # Convert to serializable dict
            return {
                '__isolated_object__': True,
                '__class_name__': cls.__name__,
                '__attrs__': {k: _serialize_for_ipc(v, visited) for k, v in obj.__dict__.items()},
            }

    # Recurse into containers
    if isinstance(obj, dict):
        return {k: _serialize_for_ipc(v, visited) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_ipc(v, visited) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_serialize_for_ipc(v, visited) for v in obj)

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


class PersistentVenvWorker(Worker):
    """
    Persistent version of VenvWorker that keeps subprocess alive.

    This reduces per-call overhead by ~200-400ms by avoiding subprocess spawn.
    Uses Unix Domain Sockets for communication (similar to original decorator approach,
    but simpler and more explicit).

    Use this for high-frequency calls to the same venv.
    """

    def __init__(
        self,
        python: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        sys_path: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize persistent worker.

        Args:
            python: Path to Python executable in target venv.
            working_dir: Working directory for subprocess.
            sys_path: Additional paths to add to sys.path.
            env: Additional environment variables.
            name: Optional name for logging.
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"PersistentVenvWorker({self.python.parent.parent.name})"

        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_pvenv_'))
        self._shm_dir = _get_shm_dir()
        self._process: Optional[subprocess.Popen] = None
        self._socket_path: Optional[Path] = None
        self._transport = None
        self._shutdown = False
        self._lock = threading.Lock()

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

        if self._process is not None and self._process.poll() is None:
            return  # Already running

        # Import transport (reuse from existing code)
        from ..ipc.transport import UnixSocketTransport, cleanup_socket

        import socket

        # Create socket
        self._socket_path = self._temp_dir / "worker.sock"
        cleanup_socket(str(self._socket_path))

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self._socket_path))
        server.listen(1)
        server.settimeout(30)

        # Write and run persistent worker script
        worker_script = self._temp_dir / "persistent_worker.py"
        worker_script.write_text(_PERSISTENT_WORKER_SCRIPT)

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

        self._process = subprocess.Popen(
            [
                str(self.python),
                str(worker_script),
                str(self._socket_path),
                str(self._shm_dir),
                json.dumps(all_sys_path),
            ],
            cwd=str(self.working_dir),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Start stderr forwarding
        def forward_stderr():
            for line in self._process.stderr:
                print(f"[{self.name}] {line.decode('utf-8', errors='replace').rstrip()}")

        threading.Thread(target=forward_stderr, daemon=True).start()

        # Wait for connection
        try:
            conn, _ = server.accept()
        except socket.timeout:
            self._process.kill()
            raise RuntimeError(f"{self.name}: Worker failed to connect")
        finally:
            server.close()

        self._transport = UnixSocketTransport(conn)

        # Wait for ready
        msg = self._transport.recv()
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

                # Wait for response
                import select
                ready, _, _ = select.select([self._transport.fileno()], [], [], timeout)

                if not ready:
                    self._process.kill()
                    self._shutdown = True
                    raise TimeoutError(f"{self.name}: Call timed out")

                response = self._transport.recv()

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
                import select
                ready, _, _ = select.select([self._transport.fileno()], [], [], timeout)

                if not ready:
                    self._process.kill()
                    self._shutdown = True
                    raise TimeoutError(f"{self.name}: Call timed out")

                response = self._transport.recv()

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

        if self._transport:
            try:
                self._transport.send({"method": "shutdown"})
            except:
                pass
            self._transport.close()

        if self._process:
            self._process.wait(timeout=5)
            if self._process.poll() is None:
                self._process.kill()

        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def is_alive(self) -> bool:
        if self._shutdown:
            return False
        if self._process is None:
            return False
        return self._process.poll() is None

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"<PersistentVenvWorker name={self.name!r} status={status}>"


# Script for persistent worker subprocess
_PERSISTENT_WORKER_SCRIPT = '''
import sys
import json
import socket
import struct
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
    socket_path = sys.argv[1]
    shm_dir = sys.argv[2]
    sys_paths = json.loads(sys.argv[3])

    # Setup paths
    for p in sys_paths:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Connect to host
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)

    def send(obj):
        data = json.dumps(obj).encode()
        sock.sendall(struct.pack(">I", len(data)) + data)

    def recv():
        header = sock.recv(4)
        if not header:
            return None
        length = struct.unpack(">I", header)[0]
        data = b""
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk:
                break
            data += chunk
        return json.loads(data)

    # Signal ready
    send({"status": "ready"})

    import torch

    while True:
        request = recv()
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

            send({"status": "ok"})

        except Exception as e:
            send({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    sock.close()

if __name__ == "__main__":
    main()
'''
