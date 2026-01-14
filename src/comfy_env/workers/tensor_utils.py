"""
Tensor utilities for robust IPC handling.

Patterns borrowed from pyisolate (MIT licensed):
- TensorKeeper: Prevents GC race conditions
- CUDA IPC re-share detection: Graceful handling of received tensors
"""

import collections
import logging
import threading
import time
from typing import Any

logger = logging.getLogger("comfy_env")


# ---------------------------------------------------------------------------
# TensorKeeper - Prevents GC Race Conditions
# ---------------------------------------------------------------------------

class TensorKeeper:
    """
    Keeps strong references to tensors during IPC to prevent premature GC.

    Problem this solves:
        When a tensor is serialized for IPC, the serialization returns
        immediately but the receiving process may not have opened the
        shared memory yet. If the sending process's tensor gets garbage
        collected, the shared memory file is deleted, causing
        "No such file or directory" errors on the receiver.

    Solution:
        Keep strong references to tensors for a configurable window
        (default 30 seconds) to ensure the receiver has time to open them.

    Usage:
        keeper = TensorKeeper()
        keeper.keep(tensor)  # Call before putting on queue
    """

    def __init__(self, retention_seconds: float = 30.0):
        """
        Args:
            retention_seconds: How long to keep tensor references.
                              30s is safe for slow systems.
        """
        self.retention_seconds = retention_seconds
        self._keeper: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def keep(self, t: Any) -> None:
        """Keep a strong reference to tensor for retention_seconds."""
        # Only keep torch tensors
        try:
            import torch
            if not isinstance(t, torch.Tensor):
                return
        except ImportError:
            return

        now = time.time()
        with self._lock:
            self._keeper.append((now, t))

            # Cleanup old entries
            while self._keeper:
                timestamp, _ = self._keeper[0]
                if now - timestamp > self.retention_seconds:
                    self._keeper.popleft()
                else:
                    break

    def keep_recursive(self, obj: Any) -> None:
        """Recursively keep all tensors in a nested structure."""
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                self.keep(obj)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    self.keep_recursive(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    self.keep_recursive(v)
        except ImportError:
            pass

    def __len__(self) -> int:
        """Return number of tensors currently being kept."""
        with self._lock:
            return len(self._keeper)


# Global instance
_tensor_keeper = TensorKeeper()


def keep_tensor(t: Any) -> None:
    """Keep a tensor reference to prevent GC during IPC."""
    _tensor_keeper.keep(t)


def keep_tensors_recursive(obj: Any) -> None:
    """Keep all tensor references in a nested structure."""
    _tensor_keeper.keep_recursive(obj)


# ---------------------------------------------------------------------------
# CUDA IPC Re-share Detection
# ---------------------------------------------------------------------------

def prepare_tensor_for_ipc(t: Any) -> Any:
    """
    Prepare a tensor for IPC, handling CUDA IPC re-share limitation.

    Problem this solves:
        Tensors received via CUDA IPC cannot be re-shared. If a node
        receives a tensor via IPC and tries to return it, you get:
        "RuntimeError: Attempted to send CUDA tensor received from
        another process; this is not currently supported."

    Solution:
        Detect this situation and clone the tensor. Log a warning for
        large tensors so users can optimize their pipelines.

    Args:
        t: A tensor (or non-tensor, which is returned as-is)

    Returns:
        The tensor, possibly cloned if it was received via IPC.
    """
    try:
        import torch
        if not isinstance(t, torch.Tensor):
            return t

        if not t.is_cuda:
            # CPU tensors don't have this limitation
            return t

        # Test if tensor can be shared
        import torch.multiprocessing.reductions as reductions
        try:
            func, args = reductions.reduce_tensor(t)
            return t  # Can be shared as-is
        except RuntimeError as e:
            if "received from another process" in str(e):
                # This tensor was received via IPC and can't be re-shared
                tensor_size_mb = t.numel() * t.element_size() / (1024 * 1024)
                if tensor_size_mb > 100:
                    logger.warning(
                        f"PERFORMANCE: Cloning large CUDA tensor ({tensor_size_mb:.1f}MB) "
                        "received from another process. Consider modifying the node "
                        "to avoid returning unmodified input tensors."
                    )
                else:
                    logger.debug(
                        f"Cloning CUDA tensor ({tensor_size_mb:.2f}MB) received from another process"
                    )
                return t.clone()
            raise

    except ImportError:
        return t


def prepare_for_ipc_recursive(obj: Any) -> Any:
    """
    Recursively prepare all tensors in a nested structure for IPC.

    Also keeps tensor references to prevent GC.
    """
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            prepared = prepare_tensor_for_ipc(obj)
            keep_tensor(prepared)
            return prepared
        elif isinstance(obj, list):
            return [prepare_for_ipc_recursive(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(prepare_for_ipc_recursive(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: prepare_for_ipc_recursive(v) for k, v in obj.items()}
    except ImportError:
        pass
    return obj
