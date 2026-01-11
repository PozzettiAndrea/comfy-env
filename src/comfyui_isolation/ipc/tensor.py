"""
Tensor Serialization - Zero-copy tensor sharing via CUDA IPC and shared memory.

This module provides efficient tensor transfer between processes:
- CUDA tensors: Use CUDA IPC handles (zero-copy, ~0ms for any size)
- CPU tensors: Use shared memory via file_system strategy (zero-copy)

Based on patterns from pyisolate's tensor_serializer.py.
"""

import base64
import collections
import logging
import sys
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TensorKeeper - Prevents GC race conditions
# ---------------------------------------------------------------------------

class TensorKeeper:
    """
    Keeps strong references to serialized tensors to prevent premature GC.

    When we serialize a tensor for IPC, we return a handle/path to shared memory.
    If the original tensor is garbage collected before the receiver opens the
    shared memory, the data is lost. TensorKeeper holds references for a short
    window to prevent this race condition.

    Based on pyisolate's TensorKeeper pattern.
    """

    def __init__(self, retention_seconds: float = 30.0):
        """
        Args:
            retention_seconds: How long to keep references (default 30s)
        """
        self.retention_seconds = retention_seconds
        self._keeper: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def keep(self, tensor: Any) -> None:
        """
        Keep a reference to a tensor.

        Args:
            tensor: The tensor to keep alive
        """
        now = time.time()
        with self._lock:
            self._keeper.append((now, tensor))

            # Cleanup old references
            while self._keeper:
                timestamp, _ = self._keeper[0]
                if now - timestamp > self.retention_seconds:
                    self._keeper.popleft()
                else:
                    break


# Global tensor keeper instance
_tensor_keeper = TensorKeeper()


# ---------------------------------------------------------------------------
# Tensor Serialization
# ---------------------------------------------------------------------------

def serialize_tensor(tensor: Any) -> Dict[str, Any]:
    """
    Serialize a PyTorch tensor for IPC transfer.

    Uses zero-copy methods when possible:
    - CUDA tensors: CUDA IPC handles
    - CPU tensors: Shared memory (file_system strategy)

    Args:
        tensor: PyTorch tensor to serialize

    Returns:
        Dict with tensor metadata and IPC handle/path
    """
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    if tensor.is_cuda:
        return _serialize_cuda_tensor(tensor)
    else:
        return _serialize_cpu_tensor(tensor)


def _serialize_cpu_tensor(tensor: Any) -> Dict[str, Any]:
    """
    Serialize CPU tensor using shared memory (file_system strategy).

    The tensor is moved to shared memory, and we return the path/key
    that the receiver can use to access it.
    """
    import torch
    import torch.multiprocessing.reductions as reductions

    # Keep tensor alive until receiver opens shared memory
    _tensor_keeper.keep(tensor)

    # Move to shared memory if not already
    if not tensor.is_shared():
        tensor.share_memory_()

    # Get storage reduction info
    storage = tensor.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == 'rebuild_storage_filename':
        # file_system strategy - sargs: (cls, manager_path, storage_key, size)
        return {
            "_type": "tensor_ipc",
            "device": "cpu",
            "strategy": "file_system",
            "manager_path": sargs[1].decode('utf-8') if isinstance(sargs[1], bytes) else sargs[1],
            "storage_key": sargs[2].decode('utf-8') if isinstance(sargs[2], bytes) else sargs[2],
            "storage_size": sargs[3],
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(),
            "requires_grad": tensor.requires_grad,
        }
    elif sfunc.__name__ == 'rebuild_storage_fd':
        # Force file_system strategy for compatibility
        import torch.multiprocessing as mp
        mp.set_sharing_strategy('file_system')
        tensor.share_memory_()
        return _serialize_cpu_tensor(tensor)
    else:
        # Fallback: pickle the tensor data (slow path)
        logger.warning(f"Unknown storage reduction: {sfunc.__name__}, using pickle fallback")
        return _serialize_tensor_fallback(tensor)


def _serialize_cuda_tensor(tensor: Any) -> Dict[str, Any]:
    """
    Serialize CUDA tensor using CUDA IPC handles.

    This is zero-copy - we only transfer the IPC handle, not the data.
    The receiver uses the handle to map the same GPU memory.
    """
    import torch
    import torch.multiprocessing.reductions as reductions

    try:
        func, args = reductions.reduce_tensor(tensor)
    except RuntimeError as e:
        if "received from another process" in str(e):
            # Tensor was received via IPC and can't be re-shared
            # Need to clone it (expensive but necessary)
            tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            if tensor_size_mb > 100:
                logger.warning(
                    f"Cloning large CUDA tensor ({tensor_size_mb:.1f}MB) - "
                    "consider avoiding returning unmodified input tensors"
                )
            tensor = tensor.clone()
            func, args = reductions.reduce_tensor(tensor)
        else:
            raise

    # Keep tensor alive until receiver maps it
    _tensor_keeper.keep(tensor)

    # args structure for CUDA tensor:
    # (cls, size, stride, offset, storage_type, dtype, device_idx, handle,
    #  storage_size, storage_offset, requires_grad, ref_counter_handle,
    #  ref_counter_offset, event_handle, event_sync_required)
    return {
        "_type": "tensor_ipc",
        "device": "cuda",
        "device_idx": args[6],
        "shape": list(args[1]),
        "stride": list(args[2]),
        "offset": args[3],
        "dtype": str(args[5]),
        "handle": base64.b64encode(args[7]).decode('ascii'),
        "storage_size": args[8],
        "storage_offset": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": base64.b64encode(args[11]).decode('ascii'),
        "ref_counter_offset": args[12],
        "event_handle": base64.b64encode(args[13]).decode('ascii') if args[13] else None,
        "event_sync_required": args[14],
    }


def _serialize_tensor_fallback(tensor: Any) -> Dict[str, Any]:
    """
    Fallback serialization using pickle (slow, copies data).

    Used when zero-copy methods aren't available.
    """
    import pickle

    arr = tensor.cpu().numpy()
    return {
        "_type": "tensor_pickle",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "device": str(tensor.device),
        "data": base64.b64encode(pickle.dumps(arr)).decode('ascii'),
    }


# ---------------------------------------------------------------------------
# Tensor Deserialization
# ---------------------------------------------------------------------------

def deserialize_tensor(data: Dict[str, Any]) -> Any:
    """
    Deserialize a tensor from IPC format.

    Args:
        data: Dict with tensor metadata from serialize_tensor

    Returns:
        PyTorch tensor
    """
    import torch

    # Already a tensor (shouldn't happen, but handle gracefully)
    if isinstance(data, torch.Tensor):
        return data

    obj_type = data.get("_type")

    if obj_type == "tensor_ipc":
        device = data.get("device", "cpu")
        if device == "cuda":
            return _deserialize_cuda_tensor(data)
        else:
            return _deserialize_cpu_tensor(data)
    elif obj_type == "tensor_pickle":
        return _deserialize_tensor_fallback(data)
    else:
        raise ValueError(f"Unknown tensor type: {obj_type}")


def _deserialize_cpu_tensor(data: Dict[str, Any]) -> Any:
    """Deserialize CPU tensor from shared memory."""
    import torch
    import torch.multiprocessing.reductions as reductions

    strategy = data.get("strategy")
    if strategy != "file_system":
        raise RuntimeError(f"Unsupported CPU tensor strategy: {strategy}")

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    manager_path = data["manager_path"]
    storage_key = data["storage_key"]
    storage_size = data["storage_size"]

    # Convert to bytes if needed
    if isinstance(manager_path, str):
        manager_path = manager_path.encode('utf-8')
    if isinstance(storage_key, str):
        storage_key = storage_key.encode('utf-8')

    # Rebuild storage
    rebuilt_storage = reductions.rebuild_storage_filename(
        torch.UntypedStorage, manager_path, storage_key, storage_size
    )

    # Wrap in typed storage
    typed_storage = torch.storage.TypedStorage(
        wrap_storage=rebuilt_storage, dtype=dtype, _internal=True
    )

    # Rebuild tensor
    metadata = (
        data["offset"],
        tuple(data["shape"]),
        tuple(data["stride"]),
        data["requires_grad"],
    )
    tensor = reductions.rebuild_tensor(torch.Tensor, typed_storage, metadata)
    return tensor


def _deserialize_cuda_tensor(data: Dict[str, Any]) -> Any:
    """Deserialize CUDA tensor from IPC handle."""
    import torch
    import torch.multiprocessing.reductions as reductions

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    handle = base64.b64decode(data["handle"])
    ref_counter_handle = base64.b64decode(data["ref_counter_handle"])
    event_handle = base64.b64decode(data["event_handle"]) if data.get("event_handle") else None
    device_idx = data.get("device_idx", 0)

    tensor = reductions.rebuild_cuda_tensor(
        torch.Tensor,
        tuple(data["shape"]),
        tuple(data["stride"]),
        data["offset"],
        torch.storage.TypedStorage,
        dtype,
        device_idx,
        handle,
        data["storage_size"],
        data["storage_offset"],
        data["requires_grad"],
        ref_counter_handle,
        data["ref_counter_offset"],
        event_handle,
        data["event_sync_required"],
    )
    return tensor


def _deserialize_tensor_fallback(data: Dict[str, Any]) -> Any:
    """Deserialize tensor from pickle fallback format."""
    import pickle
    import torch

    arr = pickle.loads(base64.b64decode(data["data"]))
    tensor = torch.from_numpy(arr)

    # Move to original device if CUDA
    device = data.get("device", "cpu")
    if "cuda" in device:
        tensor = tensor.to(device)

    return tensor


# ---------------------------------------------------------------------------
# Integration with protocol.py
# ---------------------------------------------------------------------------

def is_tensor(obj: Any) -> bool:
    """Check if object is a PyTorch tensor."""
    try:
        import torch
        return isinstance(obj, torch.Tensor)
    except ImportError:
        return False


def can_use_ipc() -> bool:
    """
    Check if IPC tensor sharing is available.

    Requires:
    - PyTorch installed
    - multiprocessing spawn context works
    """
    try:
        import torch
        import torch.multiprocessing as mp
        return True
    except ImportError:
        return False
