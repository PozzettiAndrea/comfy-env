"""Base Worker interface -- the protocol every worker implementation satisfies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Worker(ABC):
    """Abstract base for process-isolation workers. Usable as a context manager."""

    @abstractmethod
    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Execute func in the isolated process. It must be picklable
        (top-level or staticmethod). Raises TimeoutError or RuntimeError."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release everything the worker holds. Idempotent; call() raises after."""
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
