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
    """Exception raised when a worker encounters an error.

    ``error_kind`` is the worker's own typed verdict about what it raised,
    computed at the raise site from the live exception object ("oom",
    "interrupt", or None). It exists so the host can restore the exception
    type ComfyUI's recovery arms dispatch on (execution.py:641 and :619)
    without ever matching message text. ``oom_stats`` carries three
    allocator-level integers only the worker can measure.
    """

    def __init__(self, message: str, traceback: Optional[str] = None,
                 error_kind: Optional[str] = None,
                 oom_stats: Optional[dict] = None):
        super().__init__(message)
        self.worker_traceback = traceback
        self.error_kind = error_kind
        self.oom_stats = oom_stats

    def __str__(self):
        msg = super().__str__()
        if self.worker_traceback:
            msg += f"\n\nWorker traceback:\n{self.worker_traceback}"
        return msg


#: Host-side prompt epoch counter, bumped by the PromptModelTracker.start
#: patch (once per prompt) and read by the request builders. A monotonic
#: int, never an object id (gc reuse would alias two prompts). 0 means "no
#: prompt observed yet"; senders translate 0 to None so workers use the
#: sticky-with-decay mark fallback until the first real prompt. Lives HERE,
#: not in state_sync: it is process state with one writer (the pool's
#: monkeypatch) and one reader (the senders), and the pure module's charter
#: is dict math only. Single-writer on ComfyUI's one executor thread; the
#: bare += is safe under that assumption and documented as such.
PROMPT_GEN = [0]


class InterruptRequested(RuntimeError):
    """The user cancelled the run; the in-flight worker call must stop.

    Raised by the parent's progress callback handler in place of a bare
    RuntimeError so the callback response can carry a typed error_kind
    instead of the worker re-deriving "interrupt" from message text. A
    RuntimeError subclass on purpose: old workers that still text-match
    keep working against a new parent."""
    pass
