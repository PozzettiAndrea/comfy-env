"""
Workers - Simple, explicit process isolation for ComfyUI nodes.

This module provides three isolation tiers:

Tier 1: TorchMPWorker (same Python, fresh CUDA context)
    - Uses torch.multiprocessing.Queue
    - Zero-copy tensor transfer via CUDA IPC
    - ~30ms overhead per call
    - Use for: Memory isolation, fresh CUDA context

Tier 2: VenvWorker (different Python/venv)
    - Uses subprocess + torch.save/load via /dev/shm
    - One memcpy per tensor direction
    - ~100-500ms overhead per call
    - Use for: Different PyTorch versions, incompatible deps

Tier 3: ContainerWorker (full isolation) [future]
    - Docker with GPU passthrough
    - Use for: Different CUDA versions, hermetic environments

Usage:
    from comfy_env.workers import get_worker, TorchMPWorker

    # Get a named worker from the pool
    worker = get_worker("sam3d")
    result = worker.call(my_function, image=tensor)

    # Or create directly
    worker = TorchMPWorker()
    result = worker.call(my_function, arg1, arg2)
"""

from .base import Worker
from .torch_mp import TorchMPWorker
from .venv import VenvWorker, PersistentVenvWorker
from .pool import WorkerPool, get_worker, register_worker, shutdown_workers, list_workers

__all__ = [
    "Worker",
    "TorchMPWorker",
    "VenvWorker",
    "PersistentVenvWorker",
    "WorkerPool",
    "get_worker",
    "register_worker",
    "shutdown_workers",
    "list_workers",
]
