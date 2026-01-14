"""
WorkerBridge - Main IPC class for communicating with isolated workers.

This is the primary interface that ComfyUI node developers use.
"""

import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..env.config import IsolatedEnv
from ..env.manager import IsolatedEnvManager
from .protocol import encode_object, decode_object


class WorkerBridge:
    """
    Bridge for communicating with a worker process in an isolated environment.

    This class manages the worker process lifecycle and handles IPC.

    Features:
    - Lazy worker startup (starts on first call)
    - Singleton pattern (one worker per environment)
    - Auto-restart on crash
    - Graceful shutdown
    - Timeout support

    Example:
        from comfy_env import IsolatedEnv, WorkerBridge

        env = IsolatedEnv(
            name="my-node",
            python="3.10",
            cuda="12.8",
            requirements=["torch==2.8.0"],
        )

        bridge = WorkerBridge(env, worker_script=Path("worker.py"))

        # Direct method calls (preferred)
        result = bridge.process(image=my_image)

        # Or use explicit call() method
        result = bridge.call("process", image=my_image)
    """

    # Singleton instances by environment hash
    _instances: Dict[str, "WorkerBridge"] = {}
    _instances_lock = threading.Lock()

    def __init__(
        self,
        env: IsolatedEnv,
        worker_script: Path,
        base_dir: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        auto_start: bool = True,
    ):
        """
        Initialize the bridge.

        Args:
            env: Isolated environment configuration
            worker_script: Path to the worker Python script
            base_dir: Base directory for environments (default: worker_script's parent)
            log_callback: Optional callback for logging (default: print)
            auto_start: Whether to auto-start worker on first call (default: True)
        """
        self.env = env
        self.worker_script = Path(worker_script)
        self.base_dir = base_dir or self.worker_script.parent
        self.log = log_callback or print
        self.auto_start = auto_start

        self._manager = IsolatedEnvManager(self.base_dir, log_callback=log_callback)
        self._process: Optional[subprocess.Popen] = None
        self._process_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None

    @classmethod
    def get_instance(
        cls,
        env: IsolatedEnv,
        worker_script: Path,
        base_dir: Optional[Path] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> "WorkerBridge":
        """
        Get or create a singleton bridge instance for an environment.

        Args:
            env: Isolated environment configuration
            worker_script: Path to the worker Python script
            base_dir: Base directory for environments
            log_callback: Optional callback for logging

        Returns:
            WorkerBridge instance (reused if same env hash)
        """
        env_hash = env.get_env_hash()

        with cls._instances_lock:
            if env_hash not in cls._instances:
                cls._instances[env_hash] = cls(
                    env=env,
                    worker_script=worker_script,
                    base_dir=base_dir,
                    log_callback=log_callback,
                )
            return cls._instances[env_hash]

    @classmethod
    def from_config_file(
        cls,
        node_dir: Path,
        worker_script: Optional[Path] = None,
        config_file: Optional[str] = None,
        log_callback: Optional[Callable[[str], None]] = None,
        auto_start: bool = True,
    ) -> "WorkerBridge":
        """
        Create WorkerBridge from a config file.

        This is a convenience method for loading environment configuration
        from a TOML file instead of defining it programmatically.

        Args:
            node_dir: Directory containing the config file (and base_dir for env)
            worker_script: Path to worker script (optional - auto-discovered from config or convention)
            config_file: Specific config file name (default: auto-discover)
            log_callback: Optional callback for logging
            auto_start: Whether to auto-start worker on first call (default: True)

        Returns:
            Configured WorkerBridge instance

        Raises:
            FileNotFoundError: If no config file found or worker script not found
            ImportError: If tomli not installed (Python < 3.11)

        Example:
            # Auto-discover everything (recommended)
            bridge = WorkerBridge.from_config_file(node_dir=Path(__file__).parent)

            # Explicit worker script (backward compatible)
            bridge = WorkerBridge.from_config_file(
                node_dir=Path(__file__).parent,
                worker_script=Path(__file__).parent / "worker.py",
            )
        """
        from ..env.config_file import load_env_from_file, discover_env_config, CONFIG_FILE_NAMES

        node_dir = Path(node_dir)

        if config_file:
            env = load_env_from_file(node_dir / config_file, node_dir)
        else:
            env = discover_env_config(node_dir)
            if env is None:
                raise FileNotFoundError(
                    f"No isolation config found in {node_dir}. "
                    f"Create one of: {', '.join(CONFIG_FILE_NAMES)}"
                )

        # Resolve worker script from explicit param, config, or convention
        if worker_script is None:
            worker_script = cls._resolve_worker_script(node_dir, env)

        return cls(
            env=env,
            worker_script=worker_script,
            base_dir=node_dir,
            log_callback=log_callback,
            auto_start=auto_start,
        )

    @staticmethod
    def _resolve_worker_script(node_dir: Path, env: "IsolatedEnv") -> Path:
        """
        Resolve worker script from config or convention.

        Discovery order:
        1. Explicit package in config ([worker] package = "worker")
        2. Explicit script in config ([worker] script = "worker.py")
        3. Convention: worker/ directory with __main__.py
        4. Convention: worker.py file

        Args:
            node_dir: Node directory
            env: IsolatedEnv with optional worker_package/worker_script

        Returns:
            Path to worker script

        Raises:
            FileNotFoundError: If no worker found
        """
        # 1. Explicit package in config
        if env.worker_package:
            pkg_dir = node_dir / env.worker_package
            main_py = pkg_dir / "__main__.py"
            if main_py.exists():
                return main_py
            raise FileNotFoundError(
                f"Worker package '{env.worker_package}' not found. "
                f"Expected: {main_py}"
            )

        # 2. Explicit script in config
        if env.worker_script:
            script = node_dir / env.worker_script
            if script.exists():
                return script
            raise FileNotFoundError(
                f"Worker script '{env.worker_script}' not found. "
                f"Expected: {script}"
            )

        # 3. Convention: worker/ directory
        worker_pkg = node_dir / "worker" / "__main__.py"
        if worker_pkg.exists():
            return worker_pkg

        # 4. Convention: worker.py file
        worker_file = node_dir / "worker.py"
        if worker_file.exists():
            return worker_file

        raise FileNotFoundError(
            f"No worker found in {node_dir}. "
            f"Create worker/__main__.py, worker.py, or specify in config:\n"
            f"  [worker]\n"
            f"  package = \"worker\"  # or\n"
            f"  script = \"worker.py\""
        )

    @property
    def python_exe(self) -> Path:
        """Get the Python executable path for the isolated environment."""
        return self._manager.get_python(self.env)

    @property
    def is_running(self) -> bool:
        """Check if worker process is currently running."""
        with self._process_lock:
            return self._process is not None and self._process.poll() is None

    def ensure_environment(self, verify_packages: Optional[list] = None) -> None:
        """
        Ensure the isolated environment exists and is ready.

        Args:
            verify_packages: Optional list of packages to verify
        """
        self._manager.setup(self.env, verify_packages=verify_packages)

    def start(self) -> None:
        """
        Start the worker process.

        Does nothing if worker is already running.
        """
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                return  # Already running

            python_exe = self.python_exe
            if not python_exe.exists():
                raise RuntimeError(
                    f"Python executable not found: {python_exe}\n"
                    f"Run ensure_environment() first or check your env configuration."
                )

            if not self.worker_script.exists():
                raise RuntimeError(f"Worker script not found: {self.worker_script}")

            self.log(f"Starting worker process...")
            self.log(f"  Python: {python_exe}")
            self.log(f"  Script: {self.worker_script}")

            self._process = subprocess.Popen(
                [str(python_exe), str(self.worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Collect stderr for error reporting
            self._stderr_lines = []

            # Start stderr reader thread
            self._stderr_thread = threading.Thread(
                target=self._read_stderr,
                daemon=True,
                name=f"worker-stderr-{self.env.name}",
            )
            self._stderr_thread.start()

            # Give the process a moment to crash if it's going to (e.g., import error)
            import time
            time.sleep(0.5)

            # Check if process died immediately
            if self._process.poll() is not None:
                exit_code = self._process.returncode
                time.sleep(0.2)  # Let stderr thread collect output
                stderr_output = "\n".join(self._stderr_lines[-20:])  # Last 20 lines
                raise RuntimeError(
                    f"Worker crashed on startup (exit code {exit_code}).\n"
                    f"Stderr output:\n{stderr_output}"
                )

            # Test connection
            try:
                response = self._send_raw({"method": "ping"}, timeout=30.0)
                if response.get("result") != "pong":
                    raise RuntimeError(f"Worker ping failed: {response}")
                self.log("Worker started successfully")
            except Exception as e:
                # Collect any stderr output for debugging
                time.sleep(0.2)
                stderr_output = "\n".join(self._stderr_lines[-20:])
                self.stop()
                raise RuntimeError(f"Worker failed to start: {e}\nStderr:\n{stderr_output}")

    def _read_stderr(self) -> None:
        """Read stderr from worker and forward to log callback."""
        if not self._process or not self._process.stderr:
            return

        for line in self._process.stderr:
            line = line.rstrip()
            if line:
                self.log(line)
                # Also collect for error reporting
                if hasattr(self, '_stderr_lines'):
                    self._stderr_lines.append(line)

    def stop(self) -> None:
        """
        Stop the worker process gracefully.
        """
        with self._process_lock:
            if self._process is None or self._process.poll() is not None:
                return

            self.log("Stopping worker...")

            # Send shutdown command
            try:
                self._send_raw({"method": "shutdown"}, timeout=5.0)
            except Exception:
                pass

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.log("Worker didn't stop gracefully, terminating...")
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()

            self._process = None
            self.log("Worker stopped")

    def _send_raw(self, request: dict, timeout: float = 300.0) -> dict:
        """
        Send a raw request and wait for response.

        Args:
            request: Request dict
            timeout: Timeout in seconds

        Returns:
            Response dict
        """
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Worker process is not running")

        # Add request ID
        if "id" not in request:
            request["id"] = str(uuid.uuid4())[:8]

        # Send request
        request_json = json.dumps(request) + "\n"
        self._process.stdin.write(request_json)
        self._process.stdin.flush()

        # Read response
        import time
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for response after {timeout}s")

            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("Worker process closed unexpectedly")

            response_line = response_line.strip()
            if not response_line:
                continue

            # Skip non-JSON lines (library output that escaped)
            if response_line.startswith('[') and not response_line.startswith('[{'):
                continue

            try:
                return json.loads(response_line)
            except json.JSONDecodeError:
                continue

    def call(
        self,
        method: str,
        timeout: float = 300.0,
        **kwargs,
    ) -> Any:
        """
        Call a method on the worker.

        Args:
            method: Method name to call
            timeout: Timeout in seconds (default: 5 minutes)
            **kwargs: Arguments to pass to the method

        Returns:
            The method's return value

        Raises:
            RuntimeError: If worker returns an error
            TimeoutError: If call times out
        """
        # Auto-start if needed
        if self.auto_start and not self.is_running:
            self.start()

        # Encode arguments
        encoded_args = encode_object(kwargs)

        # Send request
        request = {
            "method": method,
            "args": encoded_args,
        }
        response = self._send_raw(request, timeout=timeout)

        # Check for error
        if "error" in response and response["error"]:
            error_msg = response["error"]
            tb = response.get("traceback", "")
            raise RuntimeError(f"Worker error: {error_msg}\n{tb}")

        # Decode and return result
        return decode_object(response.get("result"))

    def __enter__(self) -> "WorkerBridge":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops worker."""
        self.stop()

    def __del__(self) -> None:
        """Destructor - stops worker."""
        try:
            self.stop()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """
        Enable direct method calls on the bridge.

        Instead of:
            result = bridge.call("inference", image=img)

        You can do:
            result = bridge.inference(image=img)

        Args:
            name: Method name to call on the worker

        Returns:
            A callable that forwards to self.call()
        """
        # Don't intercept private/magic attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Return a wrapper that calls the method on the worker
        def method_wrapper(*args, timeout: float = 300.0, **kwargs) -> Any:
            # If positional args provided, raise helpful error
            if args:
                raise TypeError(
                    f"bridge.{name}() doesn't accept positional arguments. "
                    f"Use keyword arguments: bridge.{name}(arg1=val1, arg2=val2)"
                )
            return self.call(name, timeout=timeout, **kwargs)

        return method_wrapper
