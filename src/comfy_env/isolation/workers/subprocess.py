"""
SubprocessWorker - Cross-venv isolation using persistent subprocess + socket IPC.

This worker supports calling functions in a different Python environment:
- Uses a persistent subprocess to avoid spawn overhead
- Socket-based IPC for commands/responses
- Transfers tensors via torch.save/load over socket
- ~50-100ms overhead per call

Use this when you need:
- Different PyTorch version
- Incompatible native library dependencies
- Different Python version

Example:
    worker = SubprocessWorker(
        python="/path/to/other/venv/bin/python",
        working_dir="/path/to/code",
    )

    # Call a method by module path
    result = worker.call_method(
        module_name="my_module",
        class_name="MyClass",
        method_name="process",
        kwargs={"image": my_tensor},
    )
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .base import Worker, WorkerError
from ...config import DEFAULT_HEALTH_CHECK_TIMEOUT

# Debug logging -- granular categories from debug.py
from ...debug import (
    SERIALIZE as _DBG_SERIALIZE, IPC as _DBG_IPC,
    WORKER as _DBG_WORKER, MODELS as _DBG_MODELS,
)
_DEBUG = any((_DBG_SERIALIZE, _DBG_IPC, _DBG_WORKER, _DBG_MODELS))  # backward compat

# Shared IPC constants needed directly by SubprocessWorker
from ._ipc_shared import (
    SOCKET_ACCEPT_TIMEOUT,
    _export_pool_fd,
    _import_pool_from_fd,
    _recv_fd,
    _send_fd,
    _cleanup_shm,
)

# =============================================================================
# Parent-side IPC code -- imported from _ipc_parent module
# =============================================================================

from ._ipc_parent import (
    # Socket utilities
    _has_af_unix,
    _get_socket_dir,
    _create_server_socket,
    _connect_to_socket,
    SocketTransport,
    # Tensor lifecycle
    _TensorKeeper,
    _parent_tensor_keeper,
    _parent_fd_registry,
    _cleanup_parent_fds,
    _serialize_tensor_native_parent,
    # CUDA IPC
    _probe_cuda_ipc,
    _serialize_cuda_ipc,
    _deserialize_cuda_ipc,
    _cuda_ipc_metadata_cache,
    _cuda_ipc_cache_tensors,
    # Pool IPC
    _POOL_IPC_ENABLED,
    _pool_ipc_metadata_cache,
    _pool_ipc_cache_tensors,
    _pool_ipc_available,
    _deserialize_pool_ipc,
    _serialize_pool_ipc_parent,
    # Serialization
    _to_shm,
    _from_shm,
    _deserialize_tensor_ref,
    _cleanup_ipc_cache,
    _serialize_for_ipc,
    _get_shm_dir,
)

# Module-level state that SubprocessWorker accesses directly
import comfy_env.isolation.workers._ipc_parent as _ipc_parent
# Module-level mutable state aliases -- SubprocessWorker sets these before
# calling _from_shm so the deserialization can find the worker's pool handle.
# We reference _ipc_parent's attributes directly for mutations.


# Persistent worker script - loaded from _persistent_worker.py at import time.
# The file is also copied to the worker's temp directory at startup.
_WORKER_SCRIPT_PATH = Path(__file__).parent / "_persistent_worker.py"
_PERSISTENT_WORKER_SCRIPT = _WORKER_SCRIPT_PATH.read_text(encoding="utf-8")


class SubprocessWorker(Worker):
    """
    Cross-venv worker using persistent subprocess + socket IPC.

    Uses Unix domain sockets (or TCP localhost on older Windows) for IPC.
    This completely separates IPC from stdout/stderr, so C libraries
    printing to stdout (like Blender) won't corrupt the protocol.

    Benefits:
    - Works on Windows with different venv Python (full isolation)
    - Compiled CUDA extensions load correctly in the venv
    - ~50-100ms per call (persistent subprocess avoids spawn overhead)
    - Tensor transfer via shared memory files
    - Immune to stdout pollution from C libraries

    Use this for calls to isolated venvs with different Python/dependencies.
    """

    def __init__(
        self,
        python: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        sys_path: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT,
    ):
        """
        Initialize persistent worker.

        Args:
            python: Path to Python executable in target venv.
            working_dir: Working directory for subprocess.
            sys_path: Additional paths to add to sys.path.
            env: Additional environment variables.
            name: Optional name for logging.
            health_check_timeout: Timeout in seconds for worker health checks.
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"SubprocessWorker({self.python.parent.parent.name})"
        self.health_check_timeout = health_check_timeout

        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_pvenv_'))
        self._shm_dir = _get_shm_dir()
        self._process: Optional[subprocess.Popen] = None
        self._shutdown = False
        self._lock = threading.RLock()  # Reentrant: VRAM eviction callbacks re-enter via send_command
        self._last_new_models = []  # Auto-detected models from last call
        self._callback_handlers: Dict[str, Callable] = {}  # Bidirectional RPC callbacks
        self._call_counter = 0  # Monotonic call ID for request correlation
        self._on_restart = None  # Called when worker process is replaced (stale model cleanup)

        # Socket IPC
        self._server_socket: Optional[socket.socket] = None
        self._socket_addr: Optional[str] = None
        self._transport: Optional[SocketTransport] = None
        self._worker_pool = None  # imported pool handle for PoolIPC (worker's shareable pool)

        # Stderr inherits from parent (no pipe -- avoids tqdm/\r deadlock)

        # Write worker script and shared IPC module to temp directory.
        # The shared module is written alongside the worker so that in Phase 4
        # the worker can `import _ipc_shared` from the same directory.
        self._worker_script = self._temp_dir / "persistent_worker.py"
        self._worker_script.write_text(_PERSISTENT_WORKER_SCRIPT, encoding="utf-8")
        _ipc_shared_src = Path(__file__).parent / "_ipc_shared.py"
        shutil.copy2(_ipc_shared_src, self._temp_dir / "_ipc_shared.py")

    def _find_comfyui_base(self) -> Optional[Path]:
        """Find ComfyUI base directory."""
        # Use folder_paths.base_path (canonical source) if available
        try:
            import folder_paths
            return Path(folder_paths.base_path)
        except ImportError:
            pass

        # Fallback: Check common child directories (for test environments)
        for base in [self.working_dir, self.working_dir.parent]:
            for child in [".comfy-test-env/ComfyUI", "ComfyUI"]:
                candidate = base / child
                if (candidate / "main.py").exists() and (candidate / "comfy").exists():
                    return candidate

        # Fallback: Walk up from working_dir (standard ComfyUI custom_nodes layout)
        current = self.working_dir.resolve()
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            current = current.parent
        return None

    def _check_socket_health(self) -> bool:
        """Check if socket connection is healthy using a quick ping."""
        if not self._transport:
            print(f"[{self.name}] Health check: no transport", file=sys.stderr, flush=True)
            return False
        try:
            # Send a ping request with configurable timeout
            print(f"[{self.name}] Health check: ping (timeout={self.health_check_timeout}s)...", file=sys.stderr, flush=True)
            self._transport.send({"method": "ping"})
            response = self._transport.recv(timeout=self.health_check_timeout)
            ok = response is not None and response.get("status") == "pong"
            print(f"[{self.name}] Health check: {'ok' if ok else 'failed'}", file=sys.stderr, flush=True)
            return ok
        except Exception as e:
            print(f"[{self.name}] Socket health check exception: {e}", file=sys.stderr, flush=True)
            return False

    def _kill_worker(self) -> None:
        """Kill the worker process and clean up resources."""
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except (OSError, ProcessLookupError):
                pass
            self._process = None
        if self._transport:
            try:
                self._transport.close()
            except OSError:
                pass
            self._transport = None
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None
        self._worker_pool = None
        # Clear stale pool IPC caches -- pointer export data from dead worker's
        # pool is invalid and would corrupt CUDA state if reused with new pool
        _pool_ipc_metadata_cache.clear()
        _pool_ipc_cache_tensors.clear()

    def _worker_exit_diagnostic(self) -> str:
        """Collect diagnostic info when the worker process dies unexpectedly."""
        lines = []
        if self._process:
            rc = self._process.poll()
            lines.append(f"  exit code: {rc}")
            if rc is not None and rc < 0:
                import signal as _sig
                try:
                    sig_name = _sig.Signals(-rc).name
                    lines.append(f"  killed by signal: {sig_name}")
                except (ValueError, AttributeError):
                    pass
        # Check worker debug log
        worker_log = os.path.join(tempfile.gettempdir(), "comfy_worker_debug.log")
        if os.path.exists(worker_log):
            try:
                with open(worker_log, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    tail = content.strip().split("\n")[-20:]
                    lines.append(f"  worker log ({worker_log}, last 20 lines):")
                    for l in tail:
                        lines.append(f"    {l}")
            except Exception:
                pass
        # Check faulthandler dump
        fault_file = os.path.join(tempfile.gettempdir(), "comfy_worker_faulthandler.txt")
        if os.path.exists(fault_file):
            try:
                with open(fault_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read().strip()
                if content:
                    lines.append(f"  faulthandler dump ({fault_file}):")
                    for l in content.split("\n")[-20:]:
                        lines.append(f"    {l}")
            except Exception:
                pass
        return "\n".join(lines) if lines else "  (no diagnostic info available)"

    def _ensure_started(self):
        """Start persistent worker subprocess if not running."""
        if self._shutdown:
            raise RuntimeError(f"{self.name}: Worker has been shut down")

        if self._process is not None and self._process.poll() is None:
            # Process is running, but check if socket is healthy
            if self._transport and self._check_socket_health():
                return  # All good
            # Socket is dead/unhealthy - restart worker
            print(f"[{self.name}] Socket unhealthy, restarting worker...", file=sys.stderr, flush=True)
            self._kill_worker()
            if self._on_restart:
                try:
                    self._on_restart()
                except Exception:
                    pass
            self._last_new_models = []  # Clear stale model registry
            self._process = None  # Prevent second _on_restart fire below

        # Process is dead or never started -- fire restart callback if replacing
        if self._process is not None and self._on_restart:
            try:
                self._on_restart()
            except Exception:
                pass
            self._last_new_models = []  # Clear stale model registry

        # Clean up any previous socket
        if self._transport:
            self._transport.close()
            self._transport = None
        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

        # Create server socket for IPC
        self._server_socket, self._socket_addr = _create_server_socket()

        # Set up environment (shared with metadata scan)
        from ..wrap import build_isolation_env
        env = build_isolation_env(self.python, self.extra_env)

        # Propagate --cpu flag to subprocess so get_torch_device() returns cpu there too
        try:
            from comfy.cli_args import args as _parent_args
            if getattr(_parent_args, 'cpu', False):
                env["COMFY_CPU"] = "1"
        except Exception:
            pass

        # Find ComfyUI base and add to sys_path for real folder_paths/comfy modules
        # This works because comfy.options.args_parsing=False by default, so folder_paths
        # auto-detects its base directory from __file__ location
        comfyui_base = self._find_comfyui_base()
        if comfyui_base:
            env["COMFYUI_BASE"] = str(comfyui_base)  # Keep for fallback/debugging

        # Build sys_path: ComfyUI first (for real modules), then working_dir, then extras
        all_sys_path = []
        if comfyui_base:
            all_sys_path.append(str(comfyui_base))
        all_sys_path.append(str(self.working_dir))
        all_sys_path.extend(self.sys_path)

        print(f"[{self.name}] python: {self.python}", flush=True)
        print(f"[{self.name}] sys_path sent to worker: {all_sys_path}", flush=True)

        # Launch subprocess with the venv/pixi Python, passing socket address.
        # For pixi environments, route through `pixi run -e <env> --frozen` so
        # pixi handles activation (PATH, CONDA_PREFIX, [activation.env] vars
        # like KMP_DUPLICATE_LIB_OK, libomp/MKL setup). Hand-rolling the
        # activation worked for PATH but missed the [activation.env] block,
        # which is what set KMP_DUPLICATE_LIB_OK=TRUE — without it, torch's
        # OMP guard or delay-loaded DLLs failed at `import torch` with
        # WinError 127 / OMP Error #15. `--frozen` avoids re-resolving the
        # lockfile per worker (the original perf concern).
        is_pixi = '.pixi' in str(self.python)
        if _DBG_WORKER:
            print(f"[SubprocessWorker] is_pixi={is_pixi}, python={self.python}", flush=True)
        if is_pixi:
            # <workspace>/.pixi/envs/<env_name>/python.exe (Windows) or
            # <workspace>/.pixi/envs/<env_name>/bin/python (POSIX). Walk up to
            # find the workspace dir (parent of `.pixi/`) and the env name.
            if sys.platform == "win32":
                pixi_env_root = self.python.parent
            else:
                pixi_env_root = self.python.parent.parent
            env_name = pixi_env_root.name
            # pixi_env_root is <workspace>/.pixi/envs/<env_name>; walk up 3 levels.
            workspace_dir = pixi_env_root.parent.parent.parent

            from ...packages.pixi import PIXI

            cmd = [
                PIXI, "run", "--as-is",
                "--manifest-path", str(workspace_dir / "pixi.toml"),
                "-e", env_name,
                "python", str(self._worker_script), self._socket_addr,
            ]
            launch_env = env
        else:
            cmd = [str(self.python), str(self._worker_script), self._socket_addr]
            launch_env = env

        if _DBG_WORKER:
            print(f"[SubprocessWorker] launching cmd={cmd}...", flush=True)
        # Verify socket before spawning worker
        if self._socket_addr.startswith("abstract://"):
            if _DBG_WORKER:
                print(f"[SubprocessWorker] socket_addr={self._socket_addr} (abstract, no filesystem path)", flush=True)
        elif self._socket_addr.startswith("unix://"):
            _sock_path = self._socket_addr[7:]
            _sock_exists = os.path.exists(_sock_path)
            if _DBG_WORKER:
                print(f"[SubprocessWorker] socket_addr={self._socket_addr} exists={_sock_exists}", flush=True)
            if not _sock_exists:
                print(f"[SubprocessWorker] WARNING: socket file missing before worker spawn!", flush=True)
                if _DBG_WORKER:
                    print(f"[SubprocessWorker] socket dir={os.path.dirname(_sock_path)} dir_exists={os.path.isdir(os.path.dirname(_sock_path))}", flush=True)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,  # Inherit parent stderr (avoids pipe deadlock with tqdm)
            cwd=str(self.working_dir),
            env=launch_env,
        )

        # Accept connection from worker with timeout
        self._server_socket.settimeout(SOCKET_ACCEPT_TIMEOUT)
        try:
            client_sock, _ = self._server_socket.accept()
        except socket.timeout:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except (OSError, ProcessLookupError):
                pass
            raise RuntimeError(f"{self.name}: Worker failed to connect (timeout). Check stderr output above.")
        finally:
            self._server_socket.settimeout(None)

        self._transport = SocketTransport(client_sock)

        # Send config
        config = {
            "sys_paths": all_sys_path,
        }
        self._transport.send(config)

        # Wait for ready signal (skip any log messages from worker startup)
        try:
            while True:
                msg = self._transport.recv(timeout=60)
                if not msg:
                    raise RuntimeError(f"{self.name}: Worker failed to send ready signal")
                if msg.get("type") == "log":
                    # Worker sends log messages during startup (e.g. comfy imports)
                    print(f"[worker:{self.name}] {msg.get('message', '')}", file=sys.stderr, flush=True)
                    continue
                break
        except (ConnectionError, ConnectionResetError, OSError) as e:
            diag = self._worker_exit_diagnostic()
            raise RuntimeError(f"{self.name}: Worker died during startup: {e}\n{diag}") from e

        if msg.get("status") != "ready":
            raise RuntimeError(f"{self.name}: Unexpected ready message: {msg}")

        # --- Pool IPC handshake (receive worker's shareable pool FD) ---
        # Check worker's env vars (not parent's _POOL_IPC_ENABLED which may be False)
        self._worker_pool = None
        _worker_wants_pool_ipc = (
            self.extra_env.get("COMFY_ENV_POOL_IPC", "").lower() in ("1", "true", "yes")
            or _pool_ipc_available()
        )
        if _worker_wants_pool_ipc and sys.platform == "linux" and _has_af_unix():
            try:
                worker_fd = _recv_fd(client_sock, timeout=5)
                confirm = self._transport.recv(timeout=5)
                if confirm and confirm.get("type") == "pool_fd_sent":
                    self._worker_pool = _import_pool_from_fd(worker_fd)
                    os.close(worker_fd)
                    if _DBG_IPC:
                        print(f"[{self.name}] Pool IPC: imported worker pool", file=sys.stderr, flush=True)
                else:
                    os.close(worker_fd)
            except Exception as e:
                if _DBG_IPC:
                    print(f"[{self.name}] Pool IPC handshake failed: {e}", file=sys.stderr, flush=True)
                self._worker_pool = None

        # --- Send parent's shareable pool FD to worker (for parent->worker zero-copy) ---
        if _ipc_parent._parent_shareable_pool is not None and _has_af_unix():
            try:
                parent_pool_fd = _export_pool_fd(_ipc_parent._parent_shareable_pool)
                self._transport.send({"type": "parent_pool_fd_sent"})
                _send_fd(client_sock, parent_pool_fd)
                os.close(parent_pool_fd)
                if _DBG_IPC:
                    print(f"[{self.name}] Pool IPC: sent parent pool FD to worker",
                          file=sys.stderr, flush=True)
            except Exception as e:
                if _DBG_IPC:
                    print(f"[{self.name}] Parent pool FD send failed: {e}",
                          file=sys.stderr, flush=True)
        else:
            self._transport.send({"type": "no_parent_pool"})

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

    def register_callback(self, method: str, handler: Callable) -> None:
        """Register a handler for worker callbacks (bidirectional RPC).

        When the worker calls _call_parent(method, ...) during execution,
        the parent dispatches to the registered handler and sends the result back.

        Args:
            method: Callback method name (e.g., "request_vram_budget").
            handler: Function(request_dict) -> result_dict.
        """
        self._callback_handlers[method] = handler

    def _handle_callback(self, request: dict) -> dict:
        """Execute a callback request from the worker."""
        method = request.get("method")
        if _DBG_WORKER:
            print(f"[SubprocessWorker] callback '{method}' from call_id={request.get('call_id')}", file=sys.stderr, flush=True)
        handler = self._callback_handlers.get(method)
        if not handler:
            return {"type": "callback_response", "status": "error",
                    "error": f"Unknown callback method: {method}"}
        try:
            result = handler(request)
            return {"type": "callback_response", "status": "ok", "result": result}
        except Exception as e:
            return {"type": "callback_response", "status": "error", "error": str(e)}

    def _send_request(self, request: dict, timeout: float) -> dict:
        """Send request via socket and read response with timeout."""
        if not self._transport:
            raise RuntimeError(f"{self.name}: Transport not initialized")

        call_id = request.get("call_id")
        if _DBG_WORKER:
            msg_type = request.get("type", request.get("method", "?"))
            print(f"[SubprocessWorker] call_id={call_id} sending {msg_type}", file=sys.stderr, flush=True)

        # Send request
        self._transport.send(request)

        # Read response with timeout, handling log/callback messages along the way
        try:
            while True:
                response = self._transport.recv(timeout=timeout)
                if response is None:
                    break  # Timeout

                # Handle log messages from worker
                if response.get("type") == "log":
                    msg = response.get("message", "")
                    print(f"[worker:{self.name}] {msg}", file=sys.stderr, flush=True)
                    continue  # Keep waiting for actual response

                # Handle callback from worker (bidirectional RPC)
                if response.get("type") == "callback":
                    callback_response = self._handle_callback(response)
                    self._transport.send(callback_response)
                    continue  # Keep waiting for actual response

                # Got a real response
                break
        except ConnectionError as e:
            # Socket closed - check if worker process died
            self._shutdown = True
            exit_code = None
            if self._process:
                exit_code = self._process.poll()

            diag = self._worker_exit_diagnostic()
            if exit_code is not None:
                raise RuntimeError(
                    f"{self.name}: Worker process died with exit code {exit_code}.\n{diag}"
                ) from e
            else:
                raise RuntimeError(
                    f"{self.name}: Socket closed but worker process still running.\n{diag}"
                ) from e

        if response is None:
            # Timeout - kill process
            try:
                self._process.kill()
            except (OSError, ProcessLookupError):
                pass
            self._shutdown = True
            raise TimeoutError(f"{self.name}: call_id={call_id} timed out after {timeout}s")

        return response

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
        import sys
        if _DBG_WORKER:
            print(f"[SubprocessWorker] call_method: {module_name}.{class_name}.{method_name}", file=sys.stderr, flush=True)

        with self._lock:
            if _DBG_WORKER:
                print(f"[SubprocessWorker] acquired lock, ensuring started...", file=sys.stderr, flush=True)
            self._ensure_started()
            if _DBG_WORKER:
                print(f"[SubprocessWorker] worker started/confirmed", file=sys.stderr, flush=True)

            timeout = timeout or 600.0
            shm_registry = []

            try:
                # Serialize kwargs to shared memory
                if kwargs:
                    if _DBG_SERIALIZE:
                        for k, v in kwargs.items():
                            if hasattr(v, 'shape'):
                                print(f"[comfy-env] PRE-SERIALIZE '{k}' shape: {v.shape}", file=sys.stderr, flush=True)
                    if _DBG_SERIALIZE:
                        print(f"[SubprocessWorker] serializing kwargs to shm...", file=sys.stderr, flush=True)
                    kwargs_meta = _to_shm(kwargs, shm_registry)
                    if _DBG_SERIALIZE:
                        print(f"[SubprocessWorker] created {len(shm_registry)} shm blocks", file=sys.stderr, flush=True)
                else:
                    kwargs_meta = None

                # Send request with shared memory metadata
                self._call_counter += 1
                request = {
                    "type": "call_method",
                    "call_id": self._call_counter,
                    "module": module_name,
                    "class_name": class_name,
                    "method_name": method_name,
                    "self_state": _serialize_for_ipc(self_state) if self_state else None,
                    "kwargs": kwargs_meta,
                }
                if _DBG_WORKER:
                    print(f"[SubprocessWorker] sending request via socket...", file=sys.stderr, flush=True)
                response = self._send_request(request, timeout)
                if _DBG_WORKER:
                    print(f"[SubprocessWorker] got response: {response.get('status')}", file=sys.stderr, flush=True)

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                # Store newly auto-registered models (for proxy to create patchers)
                self._last_new_models = response.get("_new_models", [])

                # Reconstruct result from shared memory
                result_meta = response.get("result")
                if result_meta is not None:
                    _ipc_parent._active_worker_pool = self._worker_pool
                    try:
                        return _from_shm(result_meta)
                    finally:
                        _ipc_parent._active_worker_pool = None
                return None

            finally:
                _cleanup_shm(shm_registry)
                _cleanup_parent_fds(_parent_fd_registry)
                _cleanup_ipc_cache()

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
            shm_registry = []

            try:
                kwargs_meta = _to_shm(kwargs, shm_registry) if kwargs else None

                self._call_counter += 1
                request = {
                    "type": "call_module",
                    "call_id": self._call_counter,
                    "module": module,
                    "func": func,
                    "kwargs": kwargs_meta,
                }
                response = self._send_request(request, timeout)

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                result_meta = response.get("result")
                if result_meta is not None:
                    _ipc_parent._active_worker_pool = self._worker_pool
                    try:
                        return _from_shm(result_meta)
                    finally:
                        _ipc_parent._active_worker_pool = None
                return None

            finally:
                _cleanup_shm(shm_registry)
                _cleanup_parent_fds(_parent_fd_registry)
                _cleanup_ipc_cache()

    def send_command(self, method, **params):
        """Send a management command to the worker (model device moves, etc.).

        Uses the same socket transport as call_method but expects a simple
        JSON response (no shared-memory result).
        """
        with self._lock:
            self._ensure_started()
            self._call_counter += 1
            request = {"method": method, "call_id": self._call_counter, **params}
            response = self._send_request(request, timeout=60.0)
            if response.get("status") == "error":
                raise RuntimeError(
                    f"{self.name}: Command '{method}' failed: "
                    f"{response.get('error', 'Unknown error')}"
                )
            return response

    def shutdown(self) -> None:
        """Shut down the persistent worker."""
        if self._shutdown:
            return
        self._shutdown = True

        # Send shutdown signal via socket
        if self._transport and self._process and self._process.poll() is None:
            try:
                self._transport.send({"method": "shutdown"})
            except OSError:
                pass

        # Close transport and socket
        if self._transport:
            self._transport.close()
            self._transport = None

        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            # Clean up unix socket file
            if self._socket_addr and self._socket_addr.startswith("unix://"):
                try:
                    Path(self._socket_addr[7:]).unlink()
                except OSError:
                    pass
            self._server_socket = None

        # Wait for process to exit
        if self._process and self._process.poll() is None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass

        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def is_alive(self) -> bool:
        if self._shutdown:
            return False
        if self._process is None:
            return False
        return self._process.poll() is None

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"<SubprocessWorker name={self.name!r} status={status}>"
