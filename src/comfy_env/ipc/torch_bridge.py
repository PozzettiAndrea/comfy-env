"""
TorchBridge - Zero-copy CUDA tensor sharing via torch.multiprocessing.

This bridge enables passing PyTorch tensors between processes WITHOUT
copying data. CUDA tensors stay in GPU memory and are shared directly.

Requirements:
- Both host and worker must have the same PyTorch version
- Use TorchWorker as the base class for your worker

This is ideal for ML nodes where you pass large tensors between processes.
"""

import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import torch.multiprocessing as mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _worker_process(
    worker_cls: type,
    to_worker: "mp.Queue",
    from_worker: "mp.Queue",
    worker_script: Optional[Path] = None,
    setup_args: Optional[Dict] = None,
):
    """
    Entry point for the worker process.

    This runs in a separate process spawned by torch.multiprocessing.
    """
    try:
        # If worker_script is provided, import the worker class from it
        if worker_script is not None:
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("worker_module", worker_script)
            module = importlib.util.module_from_spec(spec)
            sys.modules["worker_module"] = module
            spec.loader.exec_module(module)

            # Find the TorchWorker subclass in the module
            worker_cls = None
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, TorchWorker) and obj is not TorchWorker:
                    worker_cls = obj
                    break

            if worker_cls is None:
                from_worker.put({
                    "id": "startup",
                    "error": f"No TorchWorker subclass found in {worker_script}",
                    "result": None,
                })
                return

        # Create worker instance
        worker = worker_cls()
        worker._to_host = from_worker
        worker._from_host = to_worker

        # Call setup
        if hasattr(worker, 'setup'):
            worker.setup(**(setup_args or {}))

        # Signal ready
        from_worker.put({"id": "startup", "result": "ready", "error": None})

        # Main loop
        while True:
            try:
                request = to_worker.get()

                if request is None:
                    break

                req_id = request.get("id", "unknown")
                method = request.get("method")
                args = request.get("args", {})

                if method == "shutdown":
                    from_worker.put({"id": req_id, "result": "ok", "error": None})
                    break

                if method == "ping":
                    from_worker.put({"id": req_id, "result": "pong", "error": None})
                    continue

                # Get registered method
                handler = worker._methods.get(method)
                if handler is None:
                    from_worker.put({
                        "id": req_id,
                        "error": f"Unknown method: {method}",
                        "result": None,
                    })
                    continue

                # Call the method
                try:
                    result = handler(**args)
                    from_worker.put({"id": req_id, "result": result, "error": None})
                except Exception as e:
                    from_worker.put({
                        "id": req_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                        "result": None,
                    })

            except Exception as e:
                # Log but continue
                print(f"[TorchWorker] Error in main loop: {e}")
                traceback.print_exc()

    except Exception as e:
        # Startup error
        try:
            from_worker.put({
                "id": "startup",
                "error": f"Worker startup failed: {e}\n{traceback.format_exc()}",
                "result": None,
            })
        except Exception:
            pass


class TorchWorker:
    """
    Base class for workers that use torch.multiprocessing for zero-copy tensors.

    Unlike BaseWorker which uses stdin/stdout JSON, TorchWorker uses
    torch.multiprocessing.Queue which enables zero-copy CUDA tensor sharing.

    Example:
        from comfy_env import TorchWorker, register

        class MyWorker(TorchWorker):
            def setup(self):
                import torch
                self.model = load_model()

            @register("process")
            def process(self, tensor):
                # tensor is a CUDA tensor - no copy happened!
                return self.model(tensor)

        if __name__ == "__main__":
            # Note: With TorchBridge, you don't need this
            # The bridge spawns the worker automatically
            pass
    """

    def __init__(self):
        self._methods: Dict[str, Callable] = {}
        self._to_host = None
        self._from_host = None

        # Auto-register methods decorated with @register
        for name in dir(self):
            method = getattr(self, name)
            if callable(method) and hasattr(method, '_register_name'):
                self._methods[method._register_name] = method

    def setup(self, **kwargs):
        """Override this to initialize your worker (load models, etc)."""
        pass

    def log(self, message: str):
        """Log a message (goes to stderr in the worker process)."""
        print(f"[{self.__class__.__name__}] {message}")


class TorchBridge:
    """
    Bridge for zero-copy CUDA tensor sharing with workers.

    Unlike WorkerBridge which uses subprocess+JSON, TorchBridge uses
    torch.multiprocessing which enables zero-copy sharing of CUDA tensors.

    Requirements:
    - PyTorch must be installed in both host and worker
    - Worker must be a TorchWorker subclass

    Example:
        from comfy_env import TorchBridge, TorchWorker, register
        from pathlib import Path

        # In your node:
        bridge = TorchBridge(worker_script=Path("my_worker.py"))

        # CUDA tensors are passed without copying!
        output = bridge.process(tensor=my_cuda_tensor)

        # Or using call():
        output = bridge.call("process", tensor=my_cuda_tensor)
    """

    _instances: Dict[str, "TorchBridge"] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        worker_script: Optional[Path] = None,
        worker_cls: Optional[type] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        auto_start: bool = True,
    ):
        """
        Initialize the TorchBridge.

        Args:
            worker_script: Path to a .py file containing a TorchWorker subclass
            worker_cls: Alternatively, pass the worker class directly
            log_callback: Optional callback for logging
            auto_start: Whether to auto-start on first call
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for TorchBridge. "
                "Install it with: pip install torch"
            )

        if worker_script is None and worker_cls is None:
            raise ValueError("Either worker_script or worker_cls must be provided")

        self.worker_script = Path(worker_script) if worker_script else None
        self.worker_cls = worker_cls
        self.log = log_callback or print
        self.auto_start = auto_start

        self._to_worker: Optional["mp.Queue"] = None
        self._from_worker: Optional["mp.Queue"] = None
        self._process: Optional["mp.Process"] = None
        self._process_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        """Check if worker process is running."""
        with self._process_lock:
            return self._process is not None and self._process.is_alive()

    def start(self, **setup_args) -> None:
        """
        Start the worker process.

        Args:
            **setup_args: Arguments to pass to worker's setup() method
        """
        with self._process_lock:
            if self._process is not None and self._process.is_alive():
                return

            # Set spawn method if not already set
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass  # Already set

            self._to_worker = mp.Queue()
            self._from_worker = mp.Queue()

            self.log("Starting TorchBridge worker process...")

            self._process = mp.Process(
                target=_worker_process,
                args=(
                    self.worker_cls,
                    self._to_worker,
                    self._from_worker,
                    self.worker_script,
                    setup_args,
                ),
            )
            self._process.start()

            # Wait for ready signal
            try:
                response = self._from_worker.get(timeout=60.0)
                if response.get("error"):
                    raise RuntimeError(f"Worker startup failed: {response['error']}")
                self.log("Worker started successfully")
            except Exception as e:
                self.stop()
                raise RuntimeError(f"Worker failed to start: {e}")

    def stop(self) -> None:
        """Stop the worker process."""
        with self._process_lock:
            if self._process is None:
                return

            self.log("Stopping worker...")

            # Send shutdown
            try:
                if self._to_worker:
                    self._to_worker.put({"method": "shutdown", "id": "shutdown"})
            except Exception:
                pass

            # Wait for graceful shutdown
            if self._process.is_alive():
                self._process.join(timeout=5.0)

            # Force kill if needed
            if self._process.is_alive():
                self.log("Worker didn't stop gracefully, terminating...")
                self._process.terminate()
                self._process.join(timeout=2.0)

            if self._process.is_alive():
                self._process.kill()

            # Cleanup queues
            try:
                if self._to_worker:
                    self._to_worker.close()
                if self._from_worker:
                    self._from_worker.close()
            except Exception:
                pass

            self._process = None
            self._to_worker = None
            self._from_worker = None
            self.log("Worker stopped")

    def call(self, method: str, timeout: float = 300.0, **kwargs) -> Any:
        """
        Call a method on the worker.

        Args:
            method: Method name to call
            timeout: Timeout in seconds
            **kwargs: Arguments (can include CUDA tensors!)

        Returns:
            The method's return value
        """
        if self.auto_start and not self.is_running:
            self.start()

        if not self.is_running:
            raise RuntimeError("Worker is not running")

        req_id = str(uuid.uuid4())[:8]

        self._to_worker.put({
            "id": req_id,
            "method": method,
            "args": kwargs,
        })

        # Wait for response
        try:
            response = self._from_worker.get(timeout=timeout)
        except Exception:
            raise TimeoutError(f"Timeout waiting for response after {timeout}s")

        if response.get("error"):
            tb = response.get("traceback", "")
            raise RuntimeError(f"Worker error: {response['error']}\n{tb}")

        return response.get("result")

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Enable direct method calls: bridge.process(tensor=t)"""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        def method_wrapper(*args, timeout: float = 300.0, **kwargs) -> Any:
            if args:
                raise TypeError(
                    f"bridge.{name}() doesn't accept positional arguments. "
                    f"Use keyword arguments: bridge.{name}(arg1=val1, arg2=val2)"
                )
            return self.call(name, timeout=timeout, **kwargs)

        return method_wrapper

    def __enter__(self) -> "TorchBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
