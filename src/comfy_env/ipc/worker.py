"""
BaseWorker - Base class for worker scripts that run in isolated environments.

Node developers extend this class to define their worker's functionality.
"""

import sys
import json
import traceback
from typing import Any, Callable, Dict, Optional
from functools import wraps

from .protocol import encode_object, decode_object


# Global registry of methods
_method_registry: Dict[str, Callable] = {}


def register(name: Optional[str] = None):
    """
    Decorator to register a method as callable from the bridge.

    Args:
        name: Optional method name (defaults to function name)

    Example:
        class MyWorker(BaseWorker):
            @register("process")
            def process_image(self, image):
                return processed_image

            @register()  # Uses function name "do_something"
            def do_something(self, x):
                return x * 2
    """
    def decorator(func: Callable) -> Callable:
        method_name = name if name else func.__name__
        _method_registry[method_name] = func

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

    return decorator


class BaseWorker:
    """
    Base class for isolated worker processes.

    Subclass this to create a worker that handles requests from WorkerBridge.

    The worker runs a main loop that:
    1. Reads JSON requests from stdin
    2. Dispatches to registered methods
    3. Writes JSON responses to stdout

    Example:
        class MyWorker(BaseWorker):
            def setup(self):
                # Called once when worker starts
                import torch
                self.model = load_my_model()

            @register("inference")
            def run_inference(self, image, params):
                return self.model(image, **params)

        if __name__ == "__main__":
            MyWorker().run()
    """

    def __init__(self):
        """Initialize worker."""
        self._methods: Dict[str, Callable] = {}
        self._running = True

        # Register decorated methods
        for method_name, func in _method_registry.items():
            # Bind method to this instance
            self._methods[method_name] = lambda *args, f=func, **kwargs: f(self, *args, **kwargs)

        # Also check instance methods decorated with @register
        for name in dir(self):
            method = getattr(self, name)
            if hasattr(method, '__wrapped__') and name in _method_registry:
                self._methods[name] = method

    def setup(self) -> None:
        """
        Called once when the worker starts, before processing any requests.

        Override this to load models, initialize state, etc.
        """
        pass

    def teardown(self) -> None:
        """
        Called when the worker is shutting down.

        Override this to cleanup resources.
        """
        pass

    def log(self, message: str) -> None:
        """
        Log a message to stderr (visible in main process).

        Args:
            message: Message to log
        """
        print(f"[Worker] {message}", file=sys.stderr, flush=True)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a single request.

        Args:
            request: Request dict with 'method' and 'args' keys

        Returns:
            Response dict with 'result' or 'error' key
        """
        method_name = request.get("method")
        args = request.get("args", {})
        request_id = request.get("id", "unknown")

        # Handle built-in commands
        if method_name == "ping":
            return {"id": request_id, "result": "pong"}

        if method_name == "shutdown":
            self._running = False
            return {"id": request_id, "result": "shutting_down"}

        if method_name == "list_methods":
            return {"id": request_id, "result": list(self._methods.keys())}

        # Find and call registered method
        if method_name not in self._methods:
            return {
                "id": request_id,
                "error": f"Unknown method: {method_name}",
                "traceback": f"Available methods: {list(self._methods.keys())}",
            }

        try:
            # Decode any encoded objects in args
            decoded_args = decode_object(args)

            # Call method
            result = self._methods[method_name](**decoded_args)

            # Encode result for JSON
            encoded_result = encode_object(result)

            return {"id": request_id, "result": encoded_result}

        except Exception as e:
            return {
                "id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def run(self) -> None:
        """
        Main worker loop - reads from stdin, writes to stdout.

        This method blocks until shutdown is requested.
        """
        # Suppress library output that could interfere with JSON protocol
        import warnings
        import logging
        import os

        warnings.filterwarnings("ignore")
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        logging.disable(logging.WARNING)

        self.log("Worker starting...")

        # Run setup
        try:
            self.setup()
            self.log("Setup complete, ready for requests")
        except Exception as e:
            self.log(f"Setup failed: {e}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)

        # Main loop
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response), flush=True)

                    if not self._running:
                        break

                except json.JSONDecodeError as e:
                    error_response = {
                        "id": "unknown",
                        "error": f"Invalid JSON: {e}",
                    }
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            self.log("Interrupted")
        finally:
            self.log("Shutting down...")
            self.teardown()
            self.log("Goodbye")
