"""
WorkerPool - Global registry and management of named workers.

Provides a simple API for getting workers by name:

    from comfy_env.workers import get_worker

    worker = get_worker("sam3d")
    result = worker.call_module("my_module", "my_func", image=tensor)

Workers are registered at startup and reused across calls:

    from comfy_env.workers import register_worker, TorchMPWorker

    register_worker("default", TorchMPWorker())
    register_worker("sam3d", PersistentVenvWorker(
        python="/path/to/venv/bin/python",
        working_dir="/path/to/nodes",
    ))
"""

import atexit
import threading
from typing import Dict, Optional, Union
from pathlib import Path

from .base import Worker


class WorkerPool:
    """
    Singleton pool of named workers.

    Manages worker lifecycle, provides access by name, handles cleanup.
    """

    _instance: Optional["WorkerPool"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._workers: Dict[str, Worker] = {}
        self._factories: Dict[str, callable] = {}
        self._worker_lock = threading.Lock()

    def register(
        self,
        name: str,
        worker: Optional[Worker] = None,
        factory: Optional[callable] = None,
    ) -> None:
        """
        Register a worker or worker factory.

        Args:
            name: Name to register under.
            worker: Pre-created worker instance.
            factory: Callable that creates worker on first use (lazy).

        Only one of worker or factory should be provided.
        """
        if worker is not None and factory is not None:
            raise ValueError("Provide either worker or factory, not both")
        if worker is None and factory is None:
            raise ValueError("Must provide worker or factory")

        with self._worker_lock:
            # Shutdown existing worker if replacing
            if name in self._workers:
                try:
                    self._workers[name].shutdown()
                except:
                    pass

            if worker is not None:
                self._workers[name] = worker
                self._factories.pop(name, None)
            else:
                self._factories[name] = factory
                self._workers.pop(name, None)

    def get(self, name: str) -> Worker:
        """
        Get a worker by name.

        Args:
            name: Registered worker name.

        Returns:
            The worker instance.

        Raises:
            KeyError: If no worker registered with that name.
        """
        with self._worker_lock:
            # Check for existing worker
            if name in self._workers:
                worker = self._workers[name]
                if worker.is_alive():
                    return worker
                # Worker died, try to recreate from factory
                if name not in self._factories:
                    raise RuntimeError(f"Worker '{name}' died and no factory to recreate")

            # Create from factory
            if name in self._factories:
                worker = self._factories[name]()
                self._workers[name] = worker
                return worker

            raise KeyError(f"No worker registered with name: {name}")

    def shutdown(self, name: Optional[str] = None) -> None:
        """
        Shutdown workers.

        Args:
            name: If provided, shutdown only this worker.
                  If None, shutdown all workers.
        """
        with self._worker_lock:
            if name is not None:
                if name in self._workers:
                    try:
                        self._workers[name].shutdown()
                    except:
                        pass
                    del self._workers[name]
            else:
                for worker in self._workers.values():
                    try:
                        worker.shutdown()
                    except:
                        pass
                self._workers.clear()

    def list_workers(self) -> Dict[str, str]:
        """
        List all registered workers.

        Returns:
            Dict of name -> status string.
        """
        with self._worker_lock:
            result = {}
            for name, worker in self._workers.items():
                status = "alive" if worker.is_alive() else "dead"
                result[name] = f"{type(worker).__name__} ({status})"
            for name in self._factories:
                if name not in result:
                    result[name] = f"factory (not started)"
            return result


# Global pool instance
_pool = WorkerPool()


def get_worker(name: str) -> Worker:
    """
    Get a worker by name from the global pool.

    Args:
        name: Registered worker name.

    Returns:
        Worker instance.

    Example:
        worker = get_worker("sam3d")
        result = worker.call_module("my_module", "my_func", image=tensor)
    """
    return _pool.get(name)


def register_worker(
    name: str,
    worker: Optional[Worker] = None,
    factory: Optional[callable] = None,
) -> None:
    """
    Register a worker in the global pool.

    Args:
        name: Name to register under.
        worker: Pre-created worker instance.
        factory: Callable that creates worker on demand.

    Example:
        # Register pre-created worker
        register_worker("default", TorchMPWorker())

        # Register factory for lazy creation
        register_worker("sam3d", factory=lambda: PersistentVenvWorker(
            python="/path/to/venv/bin/python",
        ))
    """
    _pool.register(name, worker=worker, factory=factory)


def shutdown_workers(name: Optional[str] = None) -> None:
    """
    Shutdown workers in the global pool.

    Args:
        name: If provided, shutdown only this worker.
              If None, shutdown all workers.
    """
    _pool.shutdown(name)


def list_workers() -> Dict[str, str]:
    """
    List all registered workers.

    Returns:
        Dict of name -> status description.
    """
    return _pool.list_workers()


# Register default worker (TorchMPWorker) on import
def _register_default():
    from .torch_mp import TorchMPWorker
    register_worker("default", factory=lambda: TorchMPWorker(name="default"))


_register_default()

# Cleanup on exit
atexit.register(lambda: shutdown_workers())
