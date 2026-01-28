"""
Workers - Process isolation for ComfyUI nodes.

This module provides two isolation tiers:

Tier 1: MPWorker (same Python, fresh CUDA context)
    - Uses multiprocessing.Queue
    - Zero-copy tensor transfer via shared memory
    - ~30ms overhead per call
    - Use for: Memory isolation, fresh CUDA context

Tier 2: SubprocessWorker (different Python/venv)
    - Persistent subprocess + socket IPC
    - ~50-100ms overhead per call
    - Use for: Different PyTorch versions, incompatible deps

Usage:
    from comfy_env.workers import MPWorker, SubprocessWorker

    # Create worker directly
    worker = MPWorker()
    result = worker.call(my_function, arg1, arg2)

    # Or use SubprocessWorker for isolated Python
    worker = SubprocessWorker(python="/path/to/venv/bin/python")
    result = worker.call(my_function, image=tensor)
"""

from .base import Worker, WorkerError
from .mp import MPWorker
from .subprocess import SubprocessWorker

__all__ = [
    "Worker",
    "WorkerError",
    "MPWorker",
    "SubprocessWorker",
]
