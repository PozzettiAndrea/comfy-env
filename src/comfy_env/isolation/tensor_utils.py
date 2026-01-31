"""Tensor utilities for IPC - prevents GC races and handles CUDA re-share."""

import collections
import logging
import threading
import time
from typing import Any

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

    def keep_recursive(self, obj: Any) -> None:
        try:
            import torch
            if isinstance(obj, torch.Tensor): self.keep(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj: self.keep_recursive(item)
            elif isinstance(obj, dict):
                for v in obj.values(): self.keep_recursive(v)
        except ImportError: pass

    def __len__(self) -> int:
        with self._lock: return len(self._keeper)


_tensor_keeper = TensorKeeper()
keep_tensor = lambda t: _tensor_keeper.keep(t)
keep_tensors_recursive = lambda obj: _tensor_keeper.keep_recursive(obj)


def prepare_tensor_for_ipc(t: Any) -> Any:
    """Clone tensor if it was received via IPC (can't be re-shared)."""
    try:
        import torch
        if not isinstance(t, torch.Tensor) or not t.is_cuda: return t

        import torch.multiprocessing.reductions as reductions
        try:
            reductions.reduce_tensor(t)
            return t
        except RuntimeError as e:
            if "received from another process" in str(e):
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
