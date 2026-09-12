"""Tensor utilities for IPC: the clone-on-foreign-storage rule for CUDA re-share.

There used to be a keeper here too, holding every call's inputs AND results
for 60 s. Review (2026-09-12) found it held nothing that needed holding: the
caller's kwargs own the inputs for the call's duration, torch's CUDA IPC has a
cross-process refcount (CudaIPCSentDataLimbo) that parks an exported block
until the importer releases it, and results are the host's own memory. What it
did do was pin the last call's tensors, a 256 MB result measured, until the
next isolated call, indefinitely while ComfyUI idled. The one keep that is
load-bearing lives at the serialization point in workers/_ipc_parent.py and
is released at end of call.
"""

import logging
from typing import Any

# _ipc_shared is a standalone leaf (imports nothing from comfy_env), so this
# can be a top-level DOWNWARD import rather than a function-body bandage.
from .workers._ipc_shared import _cuda_ipc_metadata_cache

logger = logging.getLogger("comfy_env")


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
    """Recursively prepare tensors for IPC."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return prepare_tensor_for_ipc(obj)
        elif isinstance(obj, list): return [prepare_for_ipc_recursive(x) for x in obj]
        elif isinstance(obj, tuple): return tuple(prepare_for_ipc_recursive(x) for x in obj)
        elif isinstance(obj, dict): return {k: prepare_for_ipc_recursive(v) for k, v in obj.items()}
    except ImportError: pass
    return obj
