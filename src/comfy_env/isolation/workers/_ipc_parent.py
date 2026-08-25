"""Parent-side IPC: socket setup, transport, tensor serialization, keepers.

The worker's half lives in _persistent_worker.py; anything both sides must
agree on lives in _ipc_shared.py.
"""

import base64
import os
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections import deque as _deque
from multiprocessing import shared_memory as shm
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._ipc_shared import (
    TENSOR_KEEPER_TTL,
    SOCKET_ID_LENGTH,
    _memfd_read,
    _PoolPtr,
    _import_pointer,
    _evict_cache_if_needed,
    _cuda_ipc_metadata_cache,
    _deserialize_cuda_ipc,
    _cuda_ipc_cache_tensors,
    _to_shm_generic,
    _decode_np_dtype,
    deserialize_custom,
    loads_or_opaque,
)

# Debug logging -- imported by subprocess.py, passed through here
from ...debug import (
    IPC as _DBG_IPC,
)


# Socket IPC utilities - cross-platform with TCP fallback

def _has_af_unix() -> bool:
    """Check if AF_UNIX sockets are available."""
    return hasattr(socket, 'AF_UNIX')


def _get_socket_dir() -> Path:
    """Get directory for IPC sockets."""
    if sys.platform == 'linux' and os.path.isdir('/dev/shm'):
        return Path('/dev/shm')
    elif sys.platform == 'win32':
        return Path(tempfile.gettempdir())
    else:
        return Path(tempfile.gettempdir())


def _create_server_socket() -> Tuple[socket.socket, str]:
    """
    Create a server socket for IPC.

    Returns:
        Tuple of (socket, address_string).
        Address string is "abstract://name", "unix://path", or "tcp://host:port".
    """
    if _has_af_unix():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if sys.platform == 'linux':
            # Abstract namespace: kernel-only, no filesystem path that can disappear.
            abstract_name = f"\0comfy_worker_{uuid.uuid4().hex[:SOCKET_ID_LENGTH]}"
            sock.bind(abstract_name)
            sock.listen(1)
            return sock, f"abstract://{abstract_name[1:]}"
        else:
            # macOS/other: filesystem sockets (no abstract namespace support).
            # The name carries THIS process's pid: the file is only unlinked at
            # clean shutdown, so a live instance's socket sits on disk for its
            # whole session, and the startup reaper needs a way to tell it from
            # one a crashed instance left behind (pool.py:_cleanup_stale_workers).
            sock_path = (_get_socket_dir() /
                         f"comfy_worker_{os.getpid()}_"
                         f"{uuid.uuid4().hex[:SOCKET_ID_LENGTH]}.sock")
            try:
                sock_path.unlink()
            except FileNotFoundError:
                pass
            sock.bind(str(sock_path))
            sock.listen(1)
            return sock, f"unix://{sock_path}"
    else:
        # TCP localhost fallback (Windows)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 0))  # OS picks free port
        sock.listen(1)
        port = sock.getsockname()[1]
        return sock, f"tcp://127.0.0.1:{port}"




# Tensor lifecycle management (parent side)

class _TensorKeeper:
    """Hold shared tensor references to prevent GC before worker reads them."""
    def __init__(self, retention_seconds=TENSOR_KEEPER_TTL):
        self.retention_seconds = retention_seconds
        self._keeper = _deque()
        self._lock = threading.Lock()

    def keep(self, t):
        now = time.time()
        with self._lock:
            self._keeper.append((now, t))
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                self._keeper.popleft()


_parent_tensor_keeper = _TensorKeeper()


def _serialize_tensor_native_parent(t, registry):
    """Serialize CPU tensor via file_descriptor shared memory (zero-copy to worker).

    Uses share_memory_() with file_descriptor strategy. The fd is kept open on
    the parent side; the worker opens it via /proc/<pid>/fd/<N>. This avoids
    torch's storage manager prematurely unlinking /dev/shm files (torch 2.8 bug).
    """
    import torch.multiprocessing.reductions as reductions

    # Keep tensor alive until worker finishes reading
    _parent_tensor_keeper.keep(t)

    if not t.is_shared():
        t.share_memory_()

    storage = t.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == "rebuild_storage_fd":
        # sargs: (cls, DupFd, size)
        dupfd = sargs[1]
        fd = dupfd.detach()
        # Per-CALL registry, NOT a module global. A global raced exactly like
        # _call_state did (see below): it was cleared in each worker's finally,
        # so worker A finishing closed worker B's in-flight fds when two
        # workers were driven from different threads. _cleanup_shm() already
        # closes int fds and runs on this same registry.
        registry.append(fd)
        return {
            "__type__": "TensorRef",
            "strategy": "file_descriptor",
            "parent_pid": os.getpid(),
            "fd": fd,
            "storage_size": sargs[2],
            "dtype": str(t.dtype),
            "tensor_size": list(t.size()),
            "tensor_stride": list(t.stride()),
            "tensor_offset": t.storage_offset(),
            "requires_grad": t.requires_grad,
        }
    elif sfunc.__name__ == "rebuild_storage_filename":
        # Fallback for platforms where file_descriptor isn't available
        return {
            "__type__": "TensorRef",
            "strategy": "file_system",
            "manager_path": sargs[1].decode("utf-8") if isinstance(sargs[1], bytes) else sargs[1],
            "storage_key": sargs[2].decode("utf-8") if isinstance(sargs[2], bytes) else sargs[2],
            "storage_size": sargs[3],
            "dtype": str(t.dtype),
            "tensor_size": list(t.size()),
            "tensor_stride": list(t.stride()),
            "tensor_offset": t.storage_offset(),
            "requires_grad": t.requires_grad,
        }
    else:
        raise RuntimeError(f"Unexpected reduce function: {sfunc.__name__}")


# CUDA IPC - zero-copy GPU tensor transfer (Linux only)

_cuda_ipc_supported: Optional[bool] = None

# The IPC handle forwarding cache (_cuda_ipc_metadata_cache /
# _cuda_ipc_cache_tensors) is imported from _ipc_shared above -- it lives in
# the standalone leaf so tensor_utils can read it without a cycle, and so it
# sits next to _evict_cache_if_needed / MAX_IPC_CACHE_SIZE that bound it.


def _probe_cuda_ipc() -> bool:
    """Check if CUDA IPC is available (Linux only, requires CUDA)."""
    global _cuda_ipc_supported
    if _cuda_ipc_supported is not None:
        return _cuda_ipc_supported
    if sys.platform != "linux":
        _cuda_ipc_supported = False
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            _cuda_ipc_supported = False
            return False
        torch.cuda.current_device()
        _ = torch.cuda.Event(interprocess=True)
        t = torch.empty(1, device="cuda")
        # Critical: test reduce_tensor() -- fails under cudaMallocAsync
        import torch.multiprocessing.reductions as reductions
        reductions.reduce_tensor(t)
        _cuda_ipc_supported = True
    except Exception:
        _cuda_ipc_supported = False
    return _cuda_ipc_supported


def _serialize_cuda_ipc(t) -> dict:
    """Serialize CUDA tensor via IPC handle (zero-copy, JSON-safe).

    If the tensor was previously received via IPC (from another worker),
    forward the cached IPC handle instead of cloning. This enables true
    zero-copy for multi-hop chains (Worker A -> Parent -> Worker B).
    """
    # Check IPC handle cache -- forward original handle if available
    try:
        storage_id = id(t.untyped_storage())
        cached = _cuda_ipc_metadata_cache.get(storage_id)
        if cached is not None:
            # Same tensor (not a view) -- forward metadata directly
            if (list(t.size()) == cached["tensor_size"]
                    and list(t.stride()) == cached["tensor_stride"]
                    and t.storage_offset() == cached.get("tensor_offset", 0)):
                if _DBG_IPC:
                    print(f"[comfy-env] CUDA IPC cache hit -- forwarding handle (no clone)", file=sys.stderr, flush=True)
                return cached
            # View of the same storage -- forward handle with adjusted shape
            if _DBG_IPC:
                print(f"[comfy-env] CUDA IPC cache hit (view) -- forwarding handle with adjusted shape", file=sys.stderr, flush=True)
            return {**cached, "tensor_size": list(t.size()),
                    "tensor_stride": list(t.stride()),
                    "tensor_offset": t.storage_offset()}
    except Exception:
        pass  # Fall through to standard path

    import torch.multiprocessing.reductions as reductions
    try:
        func, args = reductions.reduce_tensor(t)
    except RuntimeError as e:
        if "received from another process" in str(e):
            # CUDA IPC has no cross-process refcount: the EXPORTER keeps the
            # allocation alive until the importer maps it. We return handles
            # and no tensor, so the keeper is the clone's only reference.
            t = t.clone()
            _parent_tensor_keeper.keep(t)
            func, args = reductions.reduce_tensor(t)
        else:
            raise
    return {
        "__type__": "CudaIPC",
        "tensor_size": list(args[1]),
        "tensor_stride": list(args[2]),
        "tensor_offset": args[3],
        "dtype": str(args[5]),
        "device_idx": args[6],
        "handle": base64.b64encode(args[7]).decode("ascii"),
        "storage_size": args[8],
        "storage_offset": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": base64.b64encode(args[11]).decode("ascii"),
        "ref_counter_offset": args[12],
        "event_handle": base64.b64encode(args[13]).decode("ascii") if args[13] else None,
        "event_sync_required": args[14],
    }


# Pool IPC - shareable CUDA memory pool (cudaMallocAsync-compatible)

_POOL_IPC_ENABLED = os.environ.get("COMFY_ENV_POOL_IPC", "").lower() in ("1", "true", "yes")

# Only the tensors: the parent never reads back pool metadata, it just needs
# a strong ref so the imported allocation stays mapped.
_pool_ipc_cache_tensors: Dict[int, Any] = {}

# Per-CALL state, set by SubprocessWorker around each call. THREAD-LOCAL
# on purpose: module globals here raced when two workers were driven from
# different threads (executor call in one, aiohttp route in another) --
# worker B's call could read worker A's pool handle or demotion flag. The
# RLock serializes per-worker, not globally.
_call_state = threading.local()  # attrs: worker_pool, gpu_demoted


def _get_active_worker_pool():
    return getattr(_call_state, "worker_pool", None)


def _is_gpu_demoted():
    return getattr(_call_state, "gpu_demoted", False)


def _pool_ipc_available() -> bool:
    return _POOL_IPC_ENABLED and sys.platform == "linux"


def _deserialize_pool_ipc(data, source_pool):
    """Deserialize CUDA tensor from pool pointer import (parent side)."""
    import torch
    export_data_bytes = base64.b64decode(data["export_data"])
    imported_ptr = _import_pointer(source_pool, export_data_bytes)
    device_idx = data["device_idx"]
    dtype = getattr(torch, data["dtype"].split(".")[-1])
    storage_size = data["storage_size"]

    raw = torch.as_tensor(_PoolPtr(imported_ptr, storage_size),
                          device=torch.device(f"cuda:{device_idx}"))
    tensor = torch.empty([], dtype=dtype, device=f"cuda:{device_idx}")
    tensor.set_(raw.untyped_storage(), data["tensor_offset"],
                tuple(data["tensor_size"]), tuple(data["tensor_stride"]))
    tensor.requires_grad_(data["requires_grad"])

    # Hold the imported tensor so its allocation stays mapped.
    try:
        _pool_ipc_cache_tensors[id(tensor.untyped_storage())] = tensor
    except Exception:
        pass
    return tensor


def _parent_tensor_serializer(obj, registry, visited):
    """Parent-side tensor serialization strategy.

    Tries (in order): CUDA IPC -> CPU shared memory. GPU zero-copy is
    skipped when the canary handshake demoted it for the current worker
    (thread-local _call_state.gpu_demoted, set by SubprocessWorker around
    each call).
    """
    if obj.is_cuda and not _is_gpu_demoted():
        if _probe_cuda_ipc():
            return _serialize_cuda_ipc(obj)
    tensor = obj.detach().cpu().contiguous()
    return _serialize_tensor_native_parent(tensor, registry)


def _to_shm(obj, registry, visited=None):
    """
    Serialize object to shared memory. Returns JSON-safe metadata.

    Uses the generic shared implementation with parent-specific tensor strategy.
    """
    if visited is None:
        visited = {}
    return _to_shm_generic(obj, registry, visited,
                           tensor_serializer=_parent_tensor_serializer)


# Shared memory deserialization (worker -> parent)

def _deserialize_tensor_ref(data):
    """Deserialize tensor from shared memory (TensorRef format).

    Supports file_descriptor (via /proc/<pid>/fd/<N>) and file_system (legacy).
    """
    import torch

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])
    strategy = data.get("strategy", "file_system")

    if strategy == "file_descriptor":
        import mmap as _mmap
        worker_pid = data["parent_pid"]  # "parent_pid" is the sender's pid
        sender_fd = data["fd"]
        storage_size = data["storage_size"]

        fd = os.open(f"/proc/{worker_pid}/fd/{sender_fd}", os.O_RDWR)
        buf = _mmap.mmap(fd, storage_size, _mmap.MAP_SHARED, _mmap.PROT_READ | _mmap.PROT_WRITE)
        os.close(fd)

        flat = torch.frombuffer(buf, dtype=dtype)
        tensor = flat.view(tuple(data["tensor_size"]))
        tensor._shm_buf = buf
        return tensor
    else:
        import torch.multiprocessing.reductions as reductions

        manager_path = data["manager_path"]
        storage_key = data["storage_key"]
        storage_size = data["storage_size"]

        if isinstance(manager_path, str):
            manager_path = manager_path.encode("utf-8")
        if isinstance(storage_key, str):
            storage_key = storage_key.encode("utf-8")

        rebuilt_storage = reductions.rebuild_storage_filename(
            torch.UntypedStorage, manager_path, storage_key, storage_size
        )

        typed_storage = torch.storage.TypedStorage(
            wrap_storage=rebuilt_storage, dtype=dtype, _internal=True
        )
        metadata = (
            data["tensor_offset"],
            tuple(data["tensor_size"]),
            tuple(data["tensor_stride"]),
            data["requires_grad"],
        )
        tensor = reductions.rebuild_tensor(torch.Tensor, typed_storage, metadata)
        return tensor


def _from_shm(obj, unlink=True):
    """Reconstruct object from shared memory metadata."""
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_from_shm(v, unlink) for v in obj]
        return obj

    # Registered custom type (or OpaquePayload when unknown on this side)
    if "__shm_custom__" in obj:
        return deserialize_custom(obj, lambda v: _from_shm(v, unlink))

    # PoolIPC -> zero-copy CUDA tensor via shareable pool (worker -> parent)
    if obj.get("__type__") == "PoolIPC":
        _pool = _get_active_worker_pool()
        if _pool is not None:
            return _deserialize_pool_ipc(obj, _pool)
        raise RuntimeError("PoolIPC received but no worker pool handle available")

    # CudaIPC -> zero-copy CUDA tensor deserialization
    if obj.get("__type__") == "CudaIPC":
        return _deserialize_cuda_ipc(obj)

    # TensorRef -> use PyTorch's native deserialization (new format)
    if obj.get("__type__") == "TensorRef":
        tensor = _deserialize_tensor_ref(obj)
        # Convert back to numpy if it was originally numpy
        if obj.get("__was_numpy__"):
            return tensor.numpy()
        return tensor

    # numpy array via shared memory (fallback when torch unavailable)
    if "__shm_np__" in obj:
        shape = tuple(obj["shape"])
        dtype = _decode_np_dtype(obj["dtype"])
        if "fd" in obj:
            data = _memfd_read(obj["pid"], obj["fd"], obj["size"])
            return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        else:
            block = shm.SharedMemory(name=obj["__shm_np__"])
            arr = np.ndarray(shape, dtype=dtype, buffer=block.buf).copy()
            block.close()
            if unlink:
                block.unlink()
            return arr

    # SparseTensor -> reconstruct as tagged dict with coords + feats tensors
    if "__shm_sparse_tensor__" in obj:
        import torch
        feats = _from_shm(obj["feats"], unlink)
        # Restore original dtype if metadata available
        feats_dtype = obj.get("feats_dtype")
        if feats_dtype and hasattr(torch, feats_dtype.split(".")[-1]):
            expected = getattr(torch, feats_dtype.split(".")[-1])
            if feats.dtype != expected:
                feats = feats.to(expected)
        return {
            "__sparse_tensor_data__": True,
            "coords": _from_shm(obj["coords"], unlink),
            "feats": feats,
        }

    # generic pickled object (VideoFromFile, etc.). loads_or_opaque holds
    # the bytes as OpaquePickle when this env lacks the class -- the bare
    # host (only comfy-env installed) can hold and forward any pack type.
    if "__shm_pickle__" in obj:
        if "fd" in obj:
            obj_bytes = _memfd_read(obj["pid"], obj["fd"], obj["size"])
        else:
            block = shm.SharedMemory(name=obj["name"])
            obj_bytes = bytes(block.buf[:obj["size"]])
            block.close()
            if unlink:
                block.unlink()
        return loads_or_opaque(obj_bytes)

    # V3 NodeOutput -> reconstruct
    if "__node_output__" in obj:
        from comfy_api.latest import io as _comfy_io
        args = _from_shm(obj["args"], unlink)
        ui = _from_shm(obj["ui"], unlink) if obj.get("ui") is not None else None
        expand = _from_shm(obj["expand"], unlink) if obj.get("expand") is not None else None
        return _comfy_io.NodeOutput(*args, ui=ui, expand=expand, block_execution=obj.get("block_execution"))

    # regular dict - recurse
    return {k: _from_shm(v, unlink) for k, v in obj.items()}


# IPC cache cleanup

def _cleanup_ipc_cache():
    """Remove stale entries and enforce size bounds on IPC forwarding caches."""
    try:
        import torch
        # Legacy CUDA IPC cache
        if _cuda_ipc_cache_tensors:
            dead = [k for k, t in _cuda_ipc_cache_tensors.items()
                    if not isinstance(t, torch.Tensor) or t.storage().size() == 0]
            for k in dead:
                _cuda_ipc_metadata_cache.pop(k, None)
                _cuda_ipc_cache_tensors.pop(k, None)
        # Pool IPC cache
        if _pool_ipc_cache_tensors:
            dead = [k for k, t in _pool_ipc_cache_tensors.items()
                    if not isinstance(t, torch.Tensor) or t.storage().size() == 0]
            for k in dead:
                _pool_ipc_cache_tensors.pop(k, None)
    except Exception:
        pass
    # Enforce size bounds to prevent unbounded growth in long sessions
    _evict_cache_if_needed(_cuda_ipc_metadata_cache)
    _evict_cache_if_needed(_cuda_ipc_cache_tensors)
    _evict_cache_if_needed(_pool_ipc_cache_tensors)


# Legacy serialization helpers (for isolated objects)

def _serialize_for_ipc(obj, visited=None):
    """
    Convert objects with broken __module__ paths to dicts for IPC.

    ComfyUI sets weird __module__ values (file paths) on custom node classes,
    which breaks pickle deserialization in the worker. This converts such
    objects to a serializable dict format.
    """
    if visited is None:
        visited = {}

    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id][1]

    # Handle Path objects - mark for reconstruction
    from pathlib import PurePath
    if isinstance(obj, PurePath):
        return {"__path__": str(obj)}

    # Check if this is a custom object with broken module path
    if (hasattr(obj, '__dict__') and
        hasattr(obj, '__class__') and
        not isinstance(obj, (dict, list, tuple, type)) and
        obj.__class__.__name__ not in ('Tensor', 'ndarray', 'module')):

        cls = obj.__class__
        module = getattr(cls, '__module__', '')

        is_problematic = (
            '/' in module or
            '\\' in module or
            module.startswith('/') or
            'custom_nodes' in module or
            module == '' or
            module == '__main__'
        )
        if is_problematic:
            result = {
                '__isolated_object__': True,
                '__class_name__': cls.__name__,
                '__attrs__': {k: _serialize_for_ipc(v, visited) for k, v in obj.__dict__.items()},
            }
            visited[obj_id] = (obj, result)
            return result

    # Recurse into containers
    if isinstance(obj, dict):
        result = {k: _serialize_for_ipc(v, visited) for k, v in obj.items()}
        visited[obj_id] = (obj, result)
        return result
    elif isinstance(obj, list):
        result = [_serialize_for_ipc(v, visited) for v in obj]
        visited[obj_id] = (obj, result)
        return result
    elif isinstance(obj, tuple):
        result = tuple(_serialize_for_ipc(v, visited) for v in obj)
        visited[obj_id] = (obj, result)
        return result

    # Primitives and other objects - cache and return as-is
    visited[obj_id] = (obj, obj)
    return obj


