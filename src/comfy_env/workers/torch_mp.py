"""
TorchMPWorker - Same-venv isolation using torch.multiprocessing.

This is the simplest and fastest worker type:
- Uses torch.multiprocessing.Queue for IPC
- Zero-copy tensor transfer via CUDA IPC (automatic)
- Fresh CUDA context in subprocess
- ~30ms overhead per call

Use this when you need:
- Memory isolation between nodes
- Fresh CUDA context (automatic VRAM cleanup on worker death)
- Same Python environment as host

Example:
    worker = TorchMPWorker()

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


def _worker_loop(queue_in, queue_out):
    """
    Worker process main loop.

    Receives work items and executes them:
    - ("call_method", module_name, class_name, method_name, self_state, kwargs): Call a method on a class
    - (func, args, kwargs): Execute a function directly
    - _SHUTDOWN: Shutdown the worker

    Runs until receiving _SHUTDOWN sentinel.
    """
    import importlib
    import os
    import sys

    # Set worker mode env var
    os.environ["COMFYUI_ISOLATION_WORKER"] = "1"

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
                    result = _execute_method_call(
                        module_name, class_name, method_name, self_state, kwargs
                    )
                    queue_out.put(("ok", result))
                else:
                    # Direct function call (legacy)
                    func, args, kwargs = item
                    result = func(*args, **kwargs)
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


def _execute_method_call(module_name: str, class_name: str, method_name: str,
                         self_state: dict, kwargs: dict) -> Any:
    """
    Execute a method call in the worker process.

    This function imports the class fresh and calls the original (un-decorated) method.
    """
    import importlib

    # Import the module
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


class TorchMPWorker(Worker):
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

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the worker.

        Args:
            name: Optional name for logging/debugging.
        """
        self.name = name or "TorchMPWorker"
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
        import torch.multiprocessing as mp

        # Use spawn to get clean subprocess (no inherited CUDA context)
        ctx = mp.get_context('spawn')

        self._queue_in = ctx.Queue()
        self._queue_out = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._queue_in, self._queue_out),
            daemon=True,
        )
        self._process.start()
        self._started = True

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
        return f"<TorchMPWorker name={self.name!r} status={status}>"
