"""
Base Worker Interface - Protocol for all worker implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Worker(ABC):
    """
    Abstract base class for process isolation workers.

    All workers must implement:
    - call(): Execute a function in the isolated process
    - shutdown(): Clean up resources

    Workers should be used as context managers when possible:

        with MPWorker() as worker:
            result = worker.call(my_func, arg1, arg2)
    """

    @abstractmethod
    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute a function in the isolated worker process.

        Args:
            func: The function to execute. Must be picklable (top-level or staticmethod).
            *args: Positional arguments passed to func.
            timeout: Optional timeout in seconds (None = no timeout).
            **kwargs: Keyword arguments passed to func.

        Returns:
            The return value of func(*args, **kwargs).

        Raises:
            TimeoutError: If execution exceeds timeout.
            RuntimeError: If worker process dies or raises exception.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shut down the worker and release resources.

        Safe to call multiple times. After shutdown, further calls to
        call() will raise RuntimeError.
        """
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if the worker process is still running."""
        pass

    def __enter__(self) -> "Worker":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


class WorkerError(Exception):
    """Exception raised when a worker encounters an error."""

    def __init__(self, message: str, traceback: Optional[str] = None):
        super().__init__(message)
        self.worker_traceback = traceback

    def __str__(self):
        msg = super().__str__()
        if self.worker_traceback:
            msg += f"\n\nWorker traceback:\n{self.worker_traceback}"
        return msg
