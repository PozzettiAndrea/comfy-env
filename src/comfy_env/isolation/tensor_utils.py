"""Tensor utilities for IPC - prevents GC races and handles CUDA re-share."""

import collections
import logging
import threading
import time
from typing import Any

# _ipc_shared is a standalone leaf (imports nothing from comfy_env), so this
# can be a top-level DOWNWARD import rather than a function-body bandage.
from .workers._ipc_shared import _cuda_ipc_metadata_cache

logger = logging.getLogger("comfy_env")


class TensorKeeper:
    """Keep tensor references during IPC to prevent premature GC."""

    def __init__(self, retention_seconds: float = 30.0):
        self.retention_seconds = retention_seconds
        self._keeper: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def keep(self, t: Any) -> None:
        try:
            import torch
            if not isinstance(t, torch.Tensor): return
        except ImportError: return

        now = time.time()
        with self._lock:
            self._keeper.append((now, t))
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                self._keeper.popleft()


_tensor_keeper = TensorKeeper()
keep_tensor = lambda t: _tensor_keeper.keep(t)


def prepare_tensor_for_ipc(t: Any) -> Any:
    """Prepare tensor for IPC. With handle forwarding, cloning is rarely needed."""
    try:
        import torch
        if not isinstance(t, torch.Tensor) or not t.is_cuda: return t

        # Check if the IPC handle cache has this tensor -- if so, no clone needed
        # because _serialize_cuda_ipc will forward the cached handle directly.
        storage_id = id(t.untyped_storage())
        if storage_id in _cuda_ipc_metadata_cache:
            return t  # Cache hit -- forwarding will handle it

        import torch.multiprocessing.reductions as reductions
        try:
            reductions.reduce_tensor(t)
            return t
        except RuntimeError as e:
            err_str = str(e)
            if "cudaMallocAsync" in err_str or "shareIpcHandle" in err_str:
                return t  # Pool IPC will handle this
            if "received from another process" in err_str:
                # No cache hit and can't reduce -- must clone as fallback
                size_mb = t.numel() * t.element_size() / (1024 * 1024)
                if size_mb > 100:
                    logger.warning(f"Cloning large CUDA tensor ({size_mb:.1f}MB) for IPC")
                return t.clone()
            raise
    except ImportError: return t


def prepare_for_ipc_recursive(obj: Any) -> Any:
    """Recursively prepare tensors for IPC and keep references."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            prepared = prepare_tensor_for_ipc(obj)
            keep_tensor(prepared)
            return prepared
        elif isinstance(obj, list): return [prepare_for_ipc_recursive(x) for x in obj]
        elif isinstance(obj, tuple): return tuple(prepare_for_ipc_recursive(x) for x in obj)
        elif isinstance(obj, dict): return {k: prepare_for_ipc_recursive(v) for k, v in obj.items()}
    except ImportError: pass
    return obj
