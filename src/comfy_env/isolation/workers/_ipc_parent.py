"""
Parent-side IPC code for comfy-env subprocess workers.

This module contains all parent-process IPC infrastructure:
- Socket creation/connection utilities
- SocketTransport (thread-safe, length-prefixed JSON)
- TensorKeeper for shared memory GC prevention
- Tensor serialization (CPU shared memory, CUDA IPC, Pool IPC)
- _to_shm / _from_shm (parent-side serialization/deserialization)
- Legacy serialization helpers for ComfyUI custom objects
"""

import base64
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import uuid
from collections import deque as _deque
from multiprocessing import shared_memory as shm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._ipc_shared import (
    MAX_MESSAGE_SIZE,
    MAX_IPC_CACHE_SIZE,
    TENSOR_KEEPER_TTL,
    SOCKET_ACCEPT_TIMEOUT,
    SOCKET_ID_LENGTH,
    _USE_MEMFD,
    _memfd_write,
    _memfd_read,
    _PoolPtr,
    _import_pointer,
    _export_pointer,
    _prepare_trimesh_for_pickle,
    _cleanup_shm,
    _evict_cache_if_needed,
    _to_shm_generic,
)

# Debug logging -- imported by subprocess.py, passed through here
from ...debug import (
    SERIALIZE as _DBG_SERIALIZE, IPC as _DBG_IPC,
    WORKER as _DBG_WORKER, MODELS as _DBG_MODELS,
)


# =============================================================================
# Socket IPC utilities - cross-platform with TCP fallback
# =============================================================================

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
            # macOS/other: filesystem sockets (no abstract namespace support)
            sock_path = _get_socket_dir() / f"comfy_worker_{uuid.uuid4().hex[:SOCKET_ID_LENGTH]}.sock"
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


def _connect_to_socket(addr: str) -> socket.socket:
    """
    Connect to a server socket.

    Args:
        addr: Address string ("abstract://name", "unix://path", or "tcp://host:port").

    Returns:
        Connected socket.
    """
    if addr.startswith("abstract://"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(f"\0{addr[11:]}")  # Prepend \0 for abstract namespace
        return sock
    elif addr.startswith("unix://"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(addr[7:])  # Strip "unix://"
        return sock
    elif addr.startswith("tcp://"):
        host_port = addr[6:]  # Strip "tcp://"
        host, port = host_port.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        return sock
    else:
        raise ValueError(f"Unknown socket address scheme: {addr}")


class SocketTransport:
    """
    Length-prefixed JSON transport over sockets.

    Message format: [4-byte big-endian length][JSON payload]
    """

    def __init__(self, sock: socket.socket):
        self._sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, obj: dict) -> None:
        """Send a JSON-serializable object."""
        data = json.dumps(obj).encode('utf-8')
        msg = struct.pack('>I', len(data)) + data
        with self._send_lock:
            self._sock.sendall(msg)

    def recv(self, timeout: Optional[float] = None) -> dict:
        """Receive a JSON object. Returns None on timeout."""
        with self._recv_lock:
            if timeout is not None:
                self._sock.settimeout(timeout)
            try:
                # Read 4-byte length header
                raw_len = self._recvall(4)
                if not raw_len:
                    raise ConnectionError("Socket closed")
                msg_len = struct.unpack('>I', raw_len)[0]

                if msg_len > MAX_MESSAGE_SIZE:
                    raise ValueError(f"Message too large: {msg_len} bytes")

                # Read payload
                data = self._recvall(msg_len)
                if len(data) < msg_len:
                    raise ConnectionError(f"Incomplete message: {len(data)}/{msg_len}")

                return json.loads(data.decode('utf-8'))
            except socket.timeout:
                return None
            finally:
                if timeout is not None:
                    self._sock.settimeout(None)

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
        return bytes(data)

    def close(self) -> None:
        """Close the socket."""
        try:
            self._sock.close()
        except OSError:
            pass


# =============================================================================
# Tensor lifecycle management (parent side)
# =============================================================================

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
_parent_fd_registry = []  # Keep fds alive until worker reads them


def _cleanup_parent_fds(registry):
    """Close parent-side fds after worker has read them."""
    for fd in registry:
        try:
            os.close(fd)
        except OSError:
            pass
    registry.clear()


def _serialize_tensor_native_parent(t, registry):
    """Serialize CPU tensor via file_descriptor shared memory (zero-copy to worker).

    Uses share_memory_() with file_descriptor strategy. The fd is kept open on
    the parent side; the worker opens it via /proc/<pid>/fd/<N>. This avoids
    torch's storage manager prematurely unlinking /dev/shm files (torch 2.8 bug).
    """
    import torch
    import torch.multiprocessing as mp
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
        _parent_fd_registry.append(fd)
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


# =============================================================================
# CUDA IPC - zero-copy GPU tensor transfer (Linux only)
# =============================================================================

_cuda_ipc_supported: Optional[bool] = None

# IPC handle forwarding cache: avoids cloning when re-sharing CUDA tensors
# that were received via IPC from another worker. Keyed by id(storage).
_cuda_ipc_metadata_cache: Dict[int, dict] = {}
_cuda_ipc_cache_tensors: Dict[int, Any] = {}  # hold tensor refs to keep storage IDs stable


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
            t = t.clone()
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


def _deserialize_cuda_ipc(data: dict):
    """Deserialize CUDA tensor from IPC handle.

    Caches the IPC metadata so the handle can be forwarded if this tensor
    is later sent to another worker (avoids cloning).
    """
    import torch
    import torch.multiprocessing.reductions as reductions
    dtype = getattr(torch, data["dtype"].split(".")[-1])
    handle = base64.b64decode(data["handle"])
    ref_counter_handle = base64.b64decode(data["ref_counter_handle"])
    event_handle = base64.b64decode(data["event_handle"]) if data["event_handle"] else None
    tensor = reductions.rebuild_cuda_tensor(
        torch.Tensor,
        tuple(data["tensor_size"]),
        tuple(data["tensor_stride"]),
        data["tensor_offset"],
        torch.storage.TypedStorage,
        dtype,
        data["device_idx"],
        handle,
        data["storage_size"],
        data["storage_offset"],
        data["requires_grad"],
        ref_counter_handle,
        data["ref_counter_offset"],
        event_handle,
        data["event_sync_required"],
    )
    # Cache IPC metadata for handle forwarding (zero-copy re-sharing)
    try:
        storage_id = id(tensor.untyped_storage())
        _cuda_ipc_metadata_cache[storage_id] = data
        _cuda_ipc_cache_tensors[storage_id] = tensor
    except Exception:
        pass
    return tensor


# =============================================================================
# Pool IPC - shareable CUDA memory pool (cudaMallocAsync-compatible)
# =============================================================================

_POOL_IPC_ENABLED = os.environ.get("COMFY_ENV_POOL_IPC", "").lower() in ("1", "true", "yes")

_pool_ipc_metadata_cache: Dict[int, dict] = {}
_pool_ipc_cache_tensors: Dict[int, Any] = {}
_active_worker_pool = None  # set per-call before _from_shm
_parent_shareable_pool = None  # set once if PATCH_SHAREABLE_POOL is enabled


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

    # Cache for cross-worker forwarding
    try:
        sid = id(tensor.untyped_storage())
        _pool_ipc_metadata_cache[sid] = data
        _pool_ipc_cache_tensors[sid] = tensor
    except Exception:
        pass
    return tensor


def _serialize_pool_ipc_parent(t):
    """Serialize CUDA tensor via pool pointer export (parent side, zero-copy)."""
    import torch
    # Check cache
    try:
        storage_id = id(t.untyped_storage())
        cached = _pool_ipc_metadata_cache.get(storage_id)
        if cached is not None:
            if (list(t.size()) == cached["tensor_size"]
                    and list(t.stride()) == cached["tensor_stride"]
                    and t.storage_offset() == cached.get("tensor_offset", 0)):
                return cached
            return {**cached, "tensor_size": list(t.size()),
                    "tensor_stride": list(t.stride()),
                    "tensor_offset": t.storage_offset()}
    except Exception:
        pass

    torch.cuda.current_stream().synchronize()
    storage = t.untyped_storage()
    export_data = _export_pointer(storage.data_ptr())

    result = {
        "__type__": "PoolIPC",
        "export_data": base64.b64encode(export_data).decode("ascii"),
        "storage_size": storage.size(),
        "dtype": str(t.dtype),
        "tensor_size": list(t.size()),
        "tensor_stride": list(t.stride()),
        "tensor_offset": t.storage_offset(),
        "device_idx": t.device.index or 0,
        "requires_grad": t.requires_grad,
    }
    try:
        _pool_ipc_metadata_cache[id(t.untyped_storage())] = result
        _pool_ipc_cache_tensors[id(t.untyped_storage())] = t
    except Exception:
        pass
    return result


# =============================================================================
# Shared memory serialization (parent -> worker)
# =============================================================================

def _parent_tensor_serializer(obj, registry, visited):
    """Parent-side tensor serialization strategy.

    Tries (in order): Pool IPC -> CUDA IPC -> CPU shared memory.
    """
    if obj.is_cuda and _parent_shareable_pool is not None:
        return _serialize_pool_ipc_parent(obj)
    if obj.is_cuda and _probe_cuda_ipc():
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


# =============================================================================
# Shared memory deserialization (worker -> parent)
# =============================================================================

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

    # PoolIPC -> zero-copy CUDA tensor via shareable pool (worker -> parent)
    if obj.get("__type__") == "PoolIPC":
        if _active_worker_pool is not None:
            return _deserialize_pool_ipc(obj, _active_worker_pool)
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
        dtype = np.dtype(obj["dtype"])
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

    # trimesh (pickled to preserve visual, metadata, normals)
    if "__shm_trimesh__" in obj:
        import pickle
        if "fd" in obj:
            mesh_bytes = _memfd_read(obj["pid"], obj["fd"], obj["size"])
        else:
            block = shm.SharedMemory(name=obj["name"])
            mesh_bytes = bytes(block.buf[:obj["size"]])
            block.close()
            if unlink:
                block.unlink()
        return pickle.loads(mesh_bytes)

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

    # generic pickled object (VideoFromFile, etc.)
    if "__shm_pickle__" in obj:
        import pickle
        if "fd" in obj:
            obj_bytes = _memfd_read(obj["pid"], obj["fd"], obj["size"])
        else:
            block = shm.SharedMemory(name=obj["name"])
            obj_bytes = bytes(block.buf[:obj["size"]])
            block.close()
            if unlink:
                block.unlink()
        return pickle.loads(obj_bytes)

    # V3 NodeOutput -> reconstruct
    if "__node_output__" in obj:
        from comfy_api.latest import io as _comfy_io
        args = _from_shm(obj["args"], unlink)
        ui = _from_shm(obj["ui"], unlink) if obj.get("ui") is not None else None
        expand = _from_shm(obj["expand"], unlink) if obj.get("expand") is not None else None
        return _comfy_io.NodeOutput(*args, ui=ui, expand=expand, block_execution=obj.get("block_execution"))

    # regular dict - recurse
    return {k: _from_shm(v, unlink) for k, v in obj.items()}


# =============================================================================
# IPC cache cleanup
# =============================================================================

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
                _pool_ipc_metadata_cache.pop(k, None)
                _pool_ipc_cache_tensors.pop(k, None)
    except Exception:
        pass
    # Enforce size bounds to prevent unbounded growth in long sessions
    _evict_cache_if_needed(_cuda_ipc_metadata_cache)
    _evict_cache_if_needed(_cuda_ipc_cache_tensors)
    _evict_cache_if_needed(_pool_ipc_metadata_cache)
    _evict_cache_if_needed(_pool_ipc_cache_tensors)


# =============================================================================
# Legacy serialization helpers (for isolated objects)
# =============================================================================

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
        return visited[obj_id]

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
            visited[obj_id] = result
            return result

    # Recurse into containers
    if isinstance(obj, dict):
        result = {k: _serialize_for_ipc(v, visited) for k, v in obj.items()}
        visited[obj_id] = result
        return result
    elif isinstance(obj, list):
        result = [_serialize_for_ipc(v, visited) for v in obj]
        visited[obj_id] = result
        return result
    elif isinstance(obj, tuple):
        result = tuple(_serialize_for_ipc(v, visited) for v in obj)
        visited[obj_id] = result
        return result

    # Primitives and other objects - cache and return as-is
    visited[obj_id] = obj
    return obj


def _get_shm_dir() -> Path:
    """Get shared memory directory for efficient tensor transfer."""
    if sys.platform == 'linux' and os.path.isdir('/dev/shm'):
        return Path('/dev/shm')
    return Path(tempfile.gettempdir())
