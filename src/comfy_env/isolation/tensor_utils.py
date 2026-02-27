"""Tensor utilities for IPC - prevents GC races and handles CUDA re-share."""

import collections
import ctypes
import logging
import os
import sys
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
    """Prepare tensor for IPC. With handle forwarding, cloning is rarely needed."""
    try:
        import torch
        if not isinstance(t, torch.Tensor) or not t.is_cuda: return t

        # Check if the IPC handle cache has this tensor — if so, no clone needed
        # because _serialize_cuda_ipc will forward the cached handle directly.
        try:
            from .workers.subprocess import _cuda_ipc_metadata_cache
            storage_id = id(t.untyped_storage())
            if storage_id in _cuda_ipc_metadata_cache:
                return t  # Cache hit — forwarding will handle it
        except (ImportError, Exception):
            pass

        import torch.multiprocessing.reductions as reductions
        try:
            reductions.reduce_tensor(t)
            return t
        except RuntimeError as e:
            if "received from another process" in str(e):
                # No cache hit and can't reduce — must clone as fallback
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


# =============================================================================
# Shared memory release via madvise(MADV_DONTNEED)
# =============================================================================

_libc = None
_PAGE_SIZE = None
_MADV_DONTNEED = 4


def _get_libc():
    """Lazy-load libc for madvise (Linux only)."""
    global _libc, _PAGE_SIZE
    if _libc is None and sys.platform == "linux":
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
    return _libc


def release_tensor(t: Any) -> bool:
    """Release shared memory pages from this process's RSS via madvise(MADV_DONTNEED).

    Call this after you've copied the data you need (e.g., to a numpy array).
    The tensor remains valid -- re-accessing it will fault pages back in from
    the page cache -- but the physical pages are removed from this process's
    resident set.

    Returns True if pages were released, False if not applicable (non-Linux,
    non-shared tensor, import error, etc.).
    """
    try:
        import torch
        if not isinstance(t, torch.Tensor):
            return False

        libc = _get_libc()
        if libc is None:
            return False

        ptr = t.data_ptr()
        size = t.nelement() * t.element_size()
        if size == 0:
            return False

        # Align to page boundaries
        aligned_ptr = ptr & ~(_PAGE_SIZE - 1)
        aligned_size = size + (ptr - aligned_ptr)

        ret = libc.madvise(
            ctypes.c_void_p(aligned_ptr),
            ctypes.c_size_t(aligned_size),
            _MADV_DONTNEED,
        )
        if ret == 0:
            size_mb = size / (1024 * 1024)
            logger.debug(f"release_tensor: madvise DONTNEED on {size_mb:.0f} MB")
            return True
        else:
            errno = ctypes.get_errno()
            logger.debug(f"release_tensor: madvise failed errno={errno}")
            return False
    except Exception:
        return False


def release_tensors_recursive(obj: Any) -> int:
    """Recursively release all tensors in a nested structure.

    Returns count of tensors released.
    """
    count = 0
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if release_tensor(obj):
                count += 1
        elif isinstance(obj, dict):
            for v in obj.values():
                count += release_tensors_recursive(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                count += release_tensors_recursive(v)
    except ImportError:
        pass
    return count
