"""
SubprocessWorker - Cross-venv isolation using persistent subprocess + socket IPC.

This worker supports calling functions in a different Python environment:
- Uses a persistent subprocess to avoid spawn overhead
- Socket-based IPC for commands/responses
- Transfers tensors via torch.save/load over socket
- ~50-100ms overhead per call

Use this when you need:
- Different PyTorch version
- Incompatible native library dependencies
- Different Python version

Example:
    worker = SubprocessWorker(
        python="/path/to/other/venv/bin/python",
        working_dir="/path/to/code",
    )

    # Call a method by module path
    result = worker.call_method(
        module_name="my_module",
        class_name="MyClass",
        method_name="process",
        kwargs={"image": my_tensor},
    )
"""

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import Worker, WorkerError
from ...packages.pixi import get_pixi_path
from ...config.types import DEFAULT_HEALTH_CHECK_TIMEOUT

# Debug logging (set COMFY_ENV_DEBUG=1 to enable)
_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")

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
        Address string is "unix://path" or "tcp://host:port".
    """
    if _has_af_unix():
        # Unix domain socket (fast, no port conflicts)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock_path = _get_socket_dir() / f"comfy_worker_{uuid.uuid4().hex[:12]}.sock"
        # Remove stale socket file if exists
        try:
            sock_path.unlink()
        except FileNotFoundError:
            pass
        sock.bind(str(sock_path))
        sock.listen(1)
        return sock, f"unix://{sock_path}"
    else:
        # TCP localhost fallback (works everywhere)
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
        addr: Address string ("unix://path" or "tcp://host:port").

    Returns:
        Connected socket.
    """
    if addr.startswith("unix://"):
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

                # Sanity check
                if msg_len > 100 * 1024 * 1024:  # 100MB limit
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
        except:
            pass


# =============================================================================
# Shared Memory Serialization
# =============================================================================

from multiprocessing import shared_memory as shm
import base64
import numpy as np


# =============================================================================
# CUDA IPC - zero-copy GPU tensor transfer (Linux only)
# =============================================================================

_cuda_ipc_supported: Optional[bool] = None


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
        _ = torch.empty(1, device="cuda")
        _cuda_ipc_supported = True
    except Exception:
        _cuda_ipc_supported = False
    return _cuda_ipc_supported


def _serialize_cuda_ipc(t) -> dict:
    """Serialize CUDA tensor via IPC handle (zero-copy, JSON-safe)."""
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
    """Deserialize CUDA tensor from IPC handle."""
    import torch
    import torch.multiprocessing.reductions as reductions
    dtype = getattr(torch, data["dtype"].split(".")[-1])
    handle = base64.b64decode(data["handle"])
    ref_counter_handle = base64.b64decode(data["ref_counter_handle"])
    event_handle = base64.b64decode(data["event_handle"]) if data["event_handle"] else None
    return reductions.rebuild_cuda_tensor(
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


def _prepare_trimesh_for_pickle(mesh):
    """
    Prepare a trimesh object for cross-Python-version pickling.

    Trimesh attaches helper objects (ray tracer, proximity query) that may use
    native extensions like embreex. These cause import errors when unpickling
    on a system without those extensions. We strip them - they'll be recreated
    lazily when needed.

    Note: Do NOT strip _cache - trimesh needs it to function properly.
    """
    # Make a copy to avoid modifying the original
    mesh = mesh.copy()

    # Remove helper objects that may have unpickleable native code references
    # These are lazily recreated on first access anyway
    # Do NOT remove _cache - it's needed for trimesh to work
    for attr in ('ray', '_ray', 'permutate', 'nearest'):
        try:
            delattr(mesh, attr)
        except AttributeError:
            pass

    return mesh


def _to_shm(obj, registry, visited=None):
    """
    Serialize object to shared memory. Returns JSON-safe metadata.

    Args:
        obj: Object to serialize
        registry: List to track SharedMemory objects for cleanup
        visited: Dict tracking already-serialized objects (cycle detection)
    """
    if visited is None:
        visited = {}

    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]

    t = type(obj).__name__

    # numpy array -> direct shared memory
    if t == 'ndarray':
        arr = np.ascontiguousarray(obj)
        block = shm.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
        registry.append(block)
        result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        visited[obj_id] = result
        return result

    # torch.Tensor -> CUDA IPC (zero-copy) or numpy -> shared memory
    # IMPORTANT: Inline serialization to avoid caching ephemeral numpy array by id().
    # The temp array can be GC'd and its address reused, causing cache collisions.
    if t == 'Tensor':
        # CUDA IPC: zero-copy GPU-to-GPU transfer (Linux only)
        if obj.is_cuda and _probe_cuda_ipc():
            result = _serialize_cuda_ipc(obj)
            visited[obj_id] = result
            return result
        arr = np.ascontiguousarray(obj.detach().cpu().numpy())
        block = shm.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
        registry.append(block)
        result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype), "__was_tensor__": True}
        visited[obj_id] = result  # Cache by tensor id, not temp array id
        return result

    # trimesh.Trimesh -> pickle -> shared memory (preserves visual, metadata, normals)
    if t == 'Trimesh':
        import pickle
        obj = _prepare_trimesh_for_pickle(obj)
        mesh_bytes = pickle.dumps(obj)

        block = shm.SharedMemory(create=True, size=len(mesh_bytes))
        block.buf[:len(mesh_bytes)] = mesh_bytes
        registry.append(block)

        result = {
            "__shm_trimesh__": True,
            "name": block.name,
            "size": len(mesh_bytes),
        }
        visited[obj_id] = result
        return result

    # SparseTensor -> decompose to coords + feats CPU tensors
    if t == 'SparseTensor':
        feats_cpu = obj.feats.detach().cpu().contiguous()
        coords_cpu = obj.coords.detach().cpu().contiguous()
        result = {
            "__shm_sparse_tensor__": True,
            "coords": _to_shm(coords_cpu, registry, visited),
            "feats": _to_shm(feats_cpu, registry, visited),
            "feats_dtype": str(feats_cpu.dtype),
        }
        visited[obj_id] = result
        return result

    # Path -> string
    from pathlib import PurePath
    if isinstance(obj, PurePath):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        result = {k: _to_shm(v, registry, visited) for k, v in obj.items()}
        visited[obj_id] = result
        return result

    # list/tuple
    if isinstance(obj, list):
        result = [_to_shm(v, registry, visited) for v in obj]
        visited[obj_id] = result
        return result
    if isinstance(obj, tuple):
        result = [_to_shm(v, registry, visited) for v in obj]  # JSON doesn't have tuples
        visited[obj_id] = result
        return result

    # primitives pass through (str, int, float, bool, None)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: pickle any remaining object to shared memory
    import pickle
    try:
        obj_bytes = pickle.dumps(obj)
        block = shm.SharedMemory(create=True, size=len(obj_bytes))
        block.buf[:len(obj_bytes)] = obj_bytes
        registry.append(block)
        result = {
            "__shm_pickle__": True,
            "name": block.name,
            "size": len(obj_bytes),
        }
        visited[obj_id] = result
        return result
    except Exception:
        return obj


def _deserialize_tensor_ref(data):
    """Deserialize tensor from PyTorch shared memory (TensorRef format)."""
    import torch
    import torch.multiprocessing.reductions as reductions

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    manager_path = data["manager_path"]
    storage_key = data["storage_key"]
    storage_size = data["storage_size"]

    # Encode to bytes if needed
    if isinstance(manager_path, str):
        manager_path = manager_path.encode("utf-8")
    if isinstance(storage_key, str):
        storage_key = storage_key.encode("utf-8")

    # Rebuild storage from shared memory file
    rebuilt_storage = reductions.rebuild_storage_filename(
        torch.UntypedStorage, manager_path, storage_key, storage_size
    )

    # Wrap in TypedStorage
    typed_storage = torch.storage.TypedStorage(
        wrap_storage=rebuilt_storage, dtype=dtype, _internal=True
    )

    # Rebuild tensor
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

    # numpy array (or tensor that was converted to numpy) - legacy format
    if "__shm_np__" in obj:
        block = shm.SharedMemory(name=obj["__shm_np__"])
        arr = np.ndarray(tuple(obj["shape"]), dtype=np.dtype(obj["dtype"]), buffer=block.buf).copy()
        block.close()
        if unlink:
            block.unlink()
        # Convert back to tensor if it was originally a tensor
        if obj.get("__was_tensor__"):
            try:
                import torch
                return torch.from_numpy(arr)
            except Exception:
                pass
        return arr

    # trimesh (pickled to preserve visual, metadata, normals)
    if "__shm_trimesh__" in obj:
        import pickle
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
        # Restore original dtype if metadata available (guards against shm dtype loss)
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
        block = shm.SharedMemory(name=obj["name"])
        obj_bytes = bytes(block.buf[:obj["size"]])
        block.close()
        if unlink:
            block.unlink()
        return pickle.loads(obj_bytes)

    # regular dict - recurse
    return {k: _from_shm(v, unlink) for k, v in obj.items()}


def _cleanup_shm(registry):
    """Unlink all shared memory blocks in registry."""
    for block in registry:
        try:
            block.close()
            block.unlink()
        except Exception:
            pass
    registry.clear()


# =============================================================================
# Legacy Serialization helpers (for isolated objects)
# =============================================================================


def _serialize_for_ipc(obj, visited=None):
    """
    Convert objects with broken __module__ paths to dicts for IPC.

    ComfyUI sets weird __module__ values (file paths) on custom node classes,
    which breaks pickle deserialization in the worker. This converts such
    objects to a serializable dict format.
    """
    if visited is None:
        visited = {}  # Maps id -> serialized result

    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]  # Return cached serialized result

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

        # Check if module looks like a file path or is problematic for pickling
        # This catches: file paths, custom_nodes imports, and modules starting with /
        is_problematic = (
            '/' in module or
            '\\' in module or
            module.startswith('/') or
            'custom_nodes' in module or
            module == '' or
            module == '__main__'
        )
        if is_problematic:
            # Convert to serializable dict and cache it
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
    # Linux: /dev/shm is RAM-backed tmpfs
    if sys.platform == 'linux' and os.path.isdir('/dev/shm'):
        return Path('/dev/shm')
    # Fallback to regular temp
    return Path(tempfile.gettempdir())


# Persistent worker script - runs as __main__ in the venv Python subprocess
# Uses Unix socket (or TCP localhost) for IPC - completely separate from stdout/stderr
_PERSISTENT_WORKER_SCRIPT = '''
import sys
import os
import json
import socket
import struct
import traceback
import faulthandler
import collections
import time
import importlib
from types import SimpleNamespace

# Enable faulthandler to dump traceback on SIGSEGV/SIGABRT/etc
faulthandler.enable(file=sys.stderr, all_threads=True)

# Also dump to a file so we can see segfaults even if stderr is lost
import tempfile as _fh_tempfile
_faulthandler_log = os.path.join(_fh_tempfile.gettempdir(), "comfy_worker_faulthandler.log")
try:
    _fh_file = open(_faulthandler_log, "a")
    faulthandler.enable(file=_fh_file, all_threads=True)
except Exception:
    pass

# Debug logging (set COMFY_ENV_DEBUG=1 to enable)
_DEBUG = os.environ.get("COMFY_ENV_DEBUG", "").lower() in ("1", "true", "yes")

# Watchdog: dump all thread stacks every 60 seconds to catch hangs
import threading
import tempfile as _tempfile
_watchdog_log = os.path.join(_tempfile.gettempdir(), "comfy_worker_watchdog.log")
def _watchdog():
    import time
    tick = 0
    while True:
        time.sleep(60)
        tick += 1
        # Dump to temp file first (faulthandler needs real file descriptor)
        tmp_path = _watchdog_log + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as tmp:
            faulthandler.dump_traceback(file=tmp, all_threads=True)
        with open(tmp_path, "r", encoding="utf-8") as tmp:
            dump = tmp.read()

        # Write to persistent log
        with open(_watchdog_log, "a", encoding="utf-8") as f:
            f.write(f"\\n=== WATCHDOG TICK {tick} ({time.strftime('%H:%M:%S')}) ===\\n")
            f.write(dump)
            f.write("=== END ===\\n")
            f.flush()

        # Also print (only if debug enabled)
        if _DEBUG:
            print(f"\\n=== WATCHDOG TICK {tick} ===", flush=True)
            print(dump, flush=True)
            print("=== END ===\\n", flush=True)

# Only start watchdog when debugging (still logs to file if needed)
if _DEBUG:
    _watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    _watchdog_thread.start()
if _DEBUG:
    print(f"[worker] Watchdog started, logging to: {_watchdog_log}", flush=True)

# File-based logging for debugging (persists even if stdout/stderr are swallowed)
import tempfile
_worker_log_file = os.path.join(tempfile.gettempdir(), "comfy_worker_debug.log")
def wlog(msg):
    """Log to file only - stdout causes pipe buffer deadlock after many requests."""
    try:
        with open(_worker_log_file, "a", encoding="utf-8") as f:
            import time
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
    # NOTE: Don't print to stdout here! After 50+ requests the pipe buffer
    # fills up and causes deadlock (parent blocked on recv, worker blocked on print)

wlog(f"[worker] === Worker starting, log file: {_worker_log_file} ===")

# Debug: print PATH at startup (only if debug enabled)
if _DEBUG:
    _path_sep = ";" if sys.platform == "win32" else ":"
    _path_parts = os.environ.get("PATH", "").split(_path_sep)
    print(f"[worker] PATH has {len(_path_parts)} entries:", file=sys.stderr, flush=True)
    for _i, _p in enumerate(_path_parts[:15]):
        print(f"[worker]   [{_i}] {_p}", file=sys.stderr, flush=True)
    if len(_path_parts) > 15:
        print(f"[worker]   ... and {len(_path_parts) - 15} more", file=sys.stderr, flush=True)

# On Windows, add host Python's DLL directories so packages like opencv can find VC++ runtime
if sys.platform == "win32":
    _host_python_dir = os.environ.get("COMFYUI_HOST_PYTHON_DIR")
    if _host_python_dir and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_host_python_dir)
            # Also add DLLs subdirectory if it exists
            _dlls_dir = os.path.join(_host_python_dir, "DLLs")
            if os.path.isdir(_dlls_dir):
                os.add_dll_directory(_dlls_dir)
        except Exception:
            pass

    # For pixi environments with MKL, add Library/bin for MKL DLLs
    _pixi_library_bin = os.environ.get("COMFYUI_PIXI_LIBRARY_BIN")
    if _pixi_library_bin and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_pixi_library_bin)
            wlog(f"[worker] Added pixi Library/bin to DLL search: {_pixi_library_bin}")
        except Exception as e:
            wlog(f"[worker] Failed to add pixi Library/bin: {e}")

# =============================================================================
# Shared Memory Serialization
# =============================================================================

from multiprocessing import shared_memory as shm
import numpy as np

# Pin to single CPU core before importing torch to prevent TSC non-monotonicity
# during libc10_cuda.so static initialization (WSL has imprecise per-core TSC sync).
# See: https://github.com/pytorch/pytorch/issues/129992
_affinity_pinned = False
if sys.platform == "linux":
    try:
        os.sched_setaffinity(0, {0})
        _affinity_pinned = True
    except OSError:
        pass

# Set PyTorch to use file_system sharing (uses /dev/shm, no resource_tracker)
try:
    import torch
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    wlog("[worker] PyTorch sharing strategy set to file_system")
except Exception as e:
    wlog(f"[worker] PyTorch not available: {e}")

# Release CPU affinity back to all cores for actual GPU work
if _affinity_pinned:
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except OSError:
        pass


# Tensor keeper - holds tensor references to prevent GC before parent reads shared memory
class TensorKeeper:
    """Keep tensors alive for a retention period to prevent shared memory deletion."""
    def __init__(self, retention_seconds=30.0):
        self.retention_seconds = retention_seconds
        self._keeper = collections.deque()
        self._lock = threading.Lock()

    def keep(self, t):
        now = time.time()
        with self._lock:
            self._keeper.append((now, t))
            # Cleanup old entries
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                self._keeper.popleft()

_tensor_keeper = TensorKeeper()

# CUDA IPC - zero-copy GPU tensor transfer (Linux only)
import base64 as _b64

_cuda_ipc_supported = None

def _probe_cuda_ipc():
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
        _ = torch.empty(1, device="cuda")
        _cuda_ipc_supported = True
        wlog("[worker] CUDA IPC supported")
    except Exception:
        _cuda_ipc_supported = False
        wlog("[worker] CUDA IPC not supported")
    return _cuda_ipc_supported


def _serialize_cuda_ipc(t):
    import torch.multiprocessing.reductions as reductions
    try:
        func, args = reductions.reduce_tensor(t)
    except RuntimeError as e:
        if "received from another process" in str(e):
            t = t.clone()
            func, args = reductions.reduce_tensor(t)
        else:
            raise
    _tensor_keeper.keep(t)
    return {
        "__type__": "CudaIPC",
        "tensor_size": list(args[1]),
        "tensor_stride": list(args[2]),
        "tensor_offset": args[3],
        "dtype": str(args[5]),
        "device_idx": args[6],
        "handle": _b64.b64encode(args[7]).decode("ascii"),
        "storage_size": args[8],
        "storage_offset": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": _b64.b64encode(args[11]).decode("ascii"),
        "ref_counter_offset": args[12],
        "event_handle": _b64.b64encode(args[13]).decode("ascii") if args[13] else None,
        "event_sync_required": args[14],
    }


def _deserialize_cuda_ipc(data):
    import torch
    import torch.multiprocessing.reductions as reductions
    dtype = getattr(torch, data["dtype"].split(".")[-1])
    handle = _b64.b64decode(data["handle"])
    ref_counter_handle = _b64.b64decode(data["ref_counter_handle"])
    event_handle = _b64.b64decode(data["event_handle"]) if data["event_handle"] else None
    return reductions.rebuild_cuda_tensor(
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


def _prepare_trimesh_for_pickle(mesh):
    """
    Prepare a trimesh object for cross-Python-version pickling.
    Strips native extension helpers that cause import errors.
    """
    mesh = mesh.copy()
    for attr in ('ray', '_ray', 'permutate', 'nearest'):
        try:
            delattr(mesh, attr)
        except AttributeError:
            pass
    return mesh


def _serialize_tensor_native(t, registry):
    """Serialize tensor using PyTorch's native shared memory (no resource_tracker)."""
    import torch
    import torch.multiprocessing.reductions as reductions

    # Keep tensor alive until parent reads it
    _tensor_keeper.keep(t)

    # Put tensor in shared memory via PyTorch's manager
    if not t.is_shared():
        t.share_memory_()

    storage = t.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == "rebuild_storage_filename":
        # sargs: (cls, manager_path, storage_key, size)
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
        # Fallback: force file_system strategy
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")
        t.share_memory_()
        return _serialize_tensor_native(t, registry)


def _to_shm(obj, registry, visited=None):
    """Serialize to shared memory. Returns JSON-safe metadata."""
    if visited is None:
        visited = {}
    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]
    t = type(obj).__name__

    # Tensor -> CUDA IPC (zero-copy) or PyTorch native shared memory
    if t == 'Tensor':
        import torch
        # CUDA IPC: zero-copy GPU-to-GPU transfer (Linux only)
        if obj.is_cuda and _probe_cuda_ipc():
            result = _serialize_cuda_ipc(obj)
            visited[obj_id] = result
            return result
        tensor = obj.detach().cpu().contiguous()
        result = _serialize_tensor_native(tensor, registry)
        visited[obj_id] = result
        return result

    # ndarray -> prefer torch native shm, fallback to plain shm
    if t == 'ndarray':
        arr = np.ascontiguousarray(obj)
        try:
            import torch
            tensor = torch.from_numpy(arr)
            result = _serialize_tensor_native(tensor, registry)
            result["__was_numpy__"] = True
            result["numpy_dtype"] = str(arr.dtype)
        except Exception:
            block = shm.SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
            registry.append(block)
            result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype),
                       "__was_tensor__": True}
        visited[obj_id] = result
        return result

    # trimesh.Trimesh -> pickle -> shared memory
    if t == 'Trimesh':
        import pickle
        obj = _prepare_trimesh_for_pickle(obj)
        mesh_bytes = pickle.dumps(obj)

        block = shm.SharedMemory(create=True, size=len(mesh_bytes))
        block.buf[:len(mesh_bytes)] = mesh_bytes
        registry.append(block)

        result = {
            "__shm_trimesh__": True,
            "name": block.name,
            "size": len(mesh_bytes),
        }
        visited[obj_id] = result
        return result

    # SparseTensor -> decompose to coords + feats CPU tensors
    if t == 'SparseTensor':
        result = {
            "__shm_sparse_tensor__": True,
            "coords": _to_shm(obj.coords.cpu(), registry, visited),
            "feats": _to_shm(obj.feats.cpu(), registry, visited),
        }
        visited[obj_id] = result
        return result

    if isinstance(obj, dict):
        result = {k: _to_shm(v, registry, visited) for k, v in obj.items()}
        visited[obj_id] = result
        return result
    if isinstance(obj, (list, tuple)):
        result = [_to_shm(v, registry, visited) for v in obj]
        visited[obj_id] = result
        return result

    # Convert numpy scalars to Python primitives for JSON serialization
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()

    return obj


def _deserialize_tensor_native(data):
    """Deserialize tensor from PyTorch shared memory."""
    import torch
    import torch.multiprocessing.reductions as reductions

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])

    manager_path = data["manager_path"]
    storage_key = data["storage_key"]
    storage_size = data["storage_size"]

    # Encode to bytes if needed
    if isinstance(manager_path, str):
        manager_path = manager_path.encode("utf-8")
    if isinstance(storage_key, str):
        storage_key = storage_key.encode("utf-8")

    # Rebuild storage from shared memory file
    rebuilt_storage = reductions.rebuild_storage_filename(
        torch.UntypedStorage, manager_path, storage_key, storage_size
    )

    # Wrap in TypedStorage
    typed_storage = torch.storage.TypedStorage(
        wrap_storage=rebuilt_storage, dtype=dtype, _internal=True
    )

    # Rebuild tensor
    metadata = (
        data["tensor_offset"],
        tuple(data["tensor_size"]),
        tuple(data["tensor_stride"]),
        data["requires_grad"],
    )
    tensor = reductions.rebuild_tensor(torch.Tensor, typed_storage, metadata)
    return tensor


def _from_shm(obj, _depth=0, _key="root"):
    """Reconstruct from shared memory metadata. Does NOT unlink - caller handles that."""
    if _DEBUG and isinstance(obj, dict) and any(k in obj for k in ("__type__", "__shm_np__", "tensor_size")):
        print(f"[comfy-env] _from_shm got dict with keys: {list(obj.keys())[:5]}", file=sys.stderr, flush=True)
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_from_shm(v, _depth+1, f"{_key}[{i}]") for i, v in enumerate(obj)]
        return obj

    # CudaIPC -> zero-copy CUDA tensor deserialization
    if obj.get("__type__") == "CudaIPC":
        wlog(f"[_from_shm] {_key}: CudaIPC tensor_size={obj.get('tensor_size')}")
        return _deserialize_cuda_ipc(obj)

    # TensorRef -> use PyTorch's native deserialization (new format, worker->parent)
    if obj.get("__type__") == "TensorRef":
        wlog(f"[_from_shm] {_key}: TensorRef tensor_size={obj.get('tensor_size')}")
        if _DEBUG:
            print(f"[comfy-env] DESERIALIZE TensorRef: tensor_size={obj.get('tensor_size')}", file=sys.stderr, flush=True)
        tensor = _deserialize_tensor_native(obj)
        wlog(f"[_from_shm] {_key}: TensorRef deserialized shape={tensor.shape}")
        if _DEBUG:
            print(f"[comfy-env] DESERIALIZED tensor shape: {tensor.shape}", file=sys.stderr, flush=True)
        # Convert back to numpy if it was originally numpy
        if obj.get("__was_numpy__"):
            return tensor.numpy()
        return tensor

    # __shm_np__ -> legacy format (parent->worker, uses Python SharedMemory)
    if "__shm_np__" in obj:
        shm_name = obj["__shm_np__"]
        shape = tuple(obj["shape"])
        dtype = obj["dtype"]
        nbytes = np.prod(shape) * np.dtype(dtype).itemsize
        wlog(f"[_from_shm] {_key}: opening shm '{shm_name}' shape={shape} dtype={dtype} ({nbytes/1e6:.1f} MB)")
        if _DEBUG:
            print(f"[comfy-env] DESERIALIZE __shm_np__: shape={obj.get('shape')}, was_tensor={obj.get('__was_tensor__')}", file=sys.stderr, flush=True)
        block = shm.SharedMemory(name=shm_name)
        wlog(f"[_from_shm] {_key}: shm opened, block.size={block.size}")
        # Unregister from resource_tracker - parent owns these blocks and will clean them up
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(block._name, "shared_memory")
        except Exception:
            pass
        wlog(f"[_from_shm] {_key}: mapping {nbytes/1e6:.1f} MB from shm (zero-copy)")
        arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=block.buf)
        _input_shm_blocks.append(block)  # keep alive -- parent cleans up after we respond
        wlog(f"[_from_shm] {_key}: mapped, arr.shape={arr.shape}")
        if _DEBUG:
            print(f"[comfy-env] DESERIALIZED arr shape: {arr.shape}", file=sys.stderr, flush=True)
        # Convert back to tensor if it was originally a tensor
        if obj.get("__was_tensor__"):
            try:
                import torch
                wlog(f"[_from_shm] {_key}: converting to torch tensor")
                result = torch.from_numpy(arr)
                wlog(f"[_from_shm] {_key}: torch tensor ready shape={result.shape}")
                return result
            except Exception:
                pass
        return arr

    # trimesh (pickled)
    if "__shm_trimesh__" in obj:
        import pickle
        wlog(f"[_from_shm] {_key}: trimesh shm '{obj['name']}' size={obj['size']}")
        block = shm.SharedMemory(name=obj["name"])
        # Unregister from resource_tracker - parent owns these blocks
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(block._name, "shared_memory")
        except Exception:
            pass
        mesh_bytes = bytes(block.buf[:obj["size"]])
        block.close()
        # Don't unlink - parent will clean up
        return pickle.loads(mesh_bytes)

    # SparseTensor -> reconstruct as tagged dict with coords + feats tensors
    if "__shm_sparse_tensor__" in obj:
        wlog(f"[_from_shm] {_key}: SparseTensor")
        import torch
        feats = _from_shm(obj["feats"], _depth+1, f"{_key}.feats")
        # Restore original dtype if metadata available (guards against shm dtype loss)
        feats_dtype = obj.get("feats_dtype")
        if feats_dtype and hasattr(torch, feats_dtype.split(".")[-1]):
            expected = getattr(torch, feats_dtype.split(".")[-1])
            if feats.dtype != expected:
                wlog(f"[_from_shm] {_key}: feats dtype mismatch {feats.dtype} -> {expected}")
                feats = feats.to(expected)
        return {
            "__sparse_tensor_data__": True,
            "coords": _from_shm(obj["coords"], _depth+1, f"{_key}.coords"),
            "feats": feats,
        }

    # generic pickled object (VideoFromFile, etc.)
    if "__shm_pickle__" in obj:
        import pickle
        wlog(f"[_from_shm] {_key}: pickled obj shm '{obj['name']}' size={obj['size']}")
        block = shm.SharedMemory(name=obj["name"])
        try:
            from multiprocessing.resource_tracker import unregister
            unregister(block._name, "shared_memory")
        except Exception:
            pass
        obj_bytes = bytes(block.buf[:obj["size"]])
        block.close()
        return pickle.loads(obj_bytes)

    # Dict - recurse with key names for debugging
    if _depth == 0:
        wlog(f"[_from_shm] top-level keys: {list(obj.keys())}")
    return {k: _from_shm(v, _depth+1, k) for k, v in obj.items()}

def _cleanup_shm(registry):
    for block in registry:
        try:
            block.close()
            block.unlink()
        except Exception:
            pass
    registry.clear()

# Shared memory keeper - holds references to prevent premature GC
class ShmKeeper:
    """Keep shm blocks alive for a retention period to prevent race conditions."""
    def __init__(self, retention_seconds=30.0):
        self.retention_seconds = retention_seconds
        self._keeper = collections.deque()
        self._lock = threading.Lock()

    def keep(self, blocks):
        now = time.time()
        with self._lock:
            self._keeper.append((now, list(blocks)))  # Copy the list
            # Cleanup old entries
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                old_time, old_blocks = self._keeper.popleft()
                _cleanup_shm(old_blocks)

_shm_keeper = ShmKeeper()

_input_shm_blocks = []  # Keep parent->worker shm blocks alive during request processing

# =============================================================================
# Object Reference System - keep complex objects in worker, pass refs to host
# =============================================================================

_object_cache = {}  # Maps ref_id -> object
_object_ids = {}    # Maps id(obj) -> ref_id (for deduplication)
_ref_counter = 0

def _cache_object(obj):
    """Store object in cache, return reference ID. Deduplicates by object id."""
    global _ref_counter
    obj_id = id(obj)

    # Return existing ref if we've seen this object
    if obj_id in _object_ids:
        return _object_ids[obj_id]

    ref_id = f"ref_{_ref_counter:08x}"
    _ref_counter += 1
    _object_cache[ref_id] = obj
    _object_ids[obj_id] = ref_id
    return ref_id

def _resolve_ref(ref_id):
    """Get object from cache by reference ID."""
    return _object_cache.get(ref_id)

def _should_use_reference(obj):
    """Check if object should be passed by reference instead of value."""
    if obj is None:
        return False
    # Primitives - pass by value
    if isinstance(obj, (bool, int, float, str, bytes)):
        return False
    # NumPy scalars - pass by value (convert to Python primitives)
    obj_type = type(obj).__name__
    if obj_type in ('float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64',
                    'uint8', 'uint16', 'uint32', 'uint64', 'bool_'):
        return False
    # NumPy arrays and torch tensors - pass by value (they serialize well)
    if obj_type in ('ndarray', 'Tensor'):
        return False
    # Dicts, lists, tuples - recurse into contents (don't ref the container)
    if isinstance(obj, (dict, list, tuple)):
        return False
    # Trimesh - pass by value but needs special handling (see _prepare_trimesh_for_pickle)
    if obj_type == 'Trimesh':
        return False
    # Everything else (custom classes) - pass by reference
    return True


def _serialize_result(obj, visited=None):
    """Convert result for IPC - complex objects become references."""
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited:
        # Circular reference - use existing ref or create one
        if obj_id in _object_ids:
            return {"__comfy_ref__": _object_ids[obj_id], "__class__": type(obj).__name__}
        return None  # Skip circular refs to primitives

    if _should_use_reference(obj):
        ref_id = _cache_object(obj)
        return {"__comfy_ref__": ref_id, "__class__": type(obj).__name__}

    visited.add(obj_id)

    # Handle trimesh objects specially - strip unpickleable native extensions
    obj_type = type(obj).__name__
    if obj_type == 'Trimesh':
        return _prepare_trimesh_for_pickle(obj)

    if isinstance(obj, dict):
        return {k: _serialize_result(v, visited) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_result(v, visited) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_serialize_result(v, visited) for v in obj)

    # Convert numpy scalars to Python primitives for JSON serialization
    if obj_type in ('float16', 'float32', 'float64'):
        return float(obj)
    if obj_type in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
        return int(obj)
    if obj_type == 'bool_':
        return bool(obj)

    return obj

def _deserialize_input(obj):
    """Convert input from IPC - references become real objects."""
    if isinstance(obj, dict):
        if "__comfy_ref__" in obj:
            ref_id = obj["__comfy_ref__"]
            real_obj = _resolve_ref(ref_id)
            if real_obj is None:
                raise ValueError(f"Object reference not found: {ref_id}")
            return real_obj
        return {k: _deserialize_input(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_input(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_deserialize_input(v) for v in obj)
    return obj


class SocketTransport:
    """Length-prefixed JSON transport."""
    def __init__(self, sock):
        self._sock = sock

    def send(self, obj):
        data = json.dumps(obj).encode("utf-8")
        msg = struct.pack(">I", len(data)) + data
        self._sock.sendall(msg)

    def recv(self):
        raw_len = self._recvall(4)
        if not raw_len:
            return None
        msg_len = struct.unpack(">I", raw_len)[0]
        data = self._recvall(msg_len)
        return json.loads(data.decode("utf-8"))

    def _recvall(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
        return bytes(data)

    def close(self):
        try:
            self._sock.close()
        except:
            pass


def _connect(addr):
    """Connect to server socket (unix:// or tcp://)."""
    if addr.startswith("unix://"):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(addr[7:])
        return sock
    elif addr.startswith("tcp://"):
        host_port = addr[6:]
        host, port = host_port.rsplit(":", 1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, int(port)))
        return sock
    else:
        raise ValueError(f"Unknown socket scheme: {addr}")


def _deserialize_isolated_objects(obj):
    """Reconstruct objects serialized with __isolated_object__ marker."""
    if isinstance(obj, dict):
        if obj.get("__path__"):
            from pathlib import Path
            return Path(obj["__path__"])
        if obj.get("__isolated_object__"):
            attrs = {k: _deserialize_isolated_objects(v) for k, v in obj.get("__attrs__", {}).items()}
            ns = SimpleNamespace(**attrs)
            ns.__class_name__ = obj.get("__class_name__", "Unknown")
            return ns
        return {k: _deserialize_isolated_objects(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deserialize_isolated_objects(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deserialize_isolated_objects(v) for v in obj)
    return obj


def main():
    wlog("[worker] Starting...")
    # Get socket address from command line
    if len(sys.argv) < 2:
        wlog("Usage: worker.py <socket_addr>")
        sys.exit(1)
    socket_addr = sys.argv[1]
    wlog(f"[worker] Connecting to {socket_addr}...")

    # Connect to host process
    sock = _connect(socket_addr)
    transport = SocketTransport(sock)
    wlog("[worker] Connected, waiting for config...")

    # Read config as first message
    config = transport.recv()
    if not config:
        wlog("[worker] No config received, exiting")
        return
    wlog("[worker] Got config, setting up paths...")

    # Setup sys.path
    for p in config.get("sys_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Try to import torch (optional - not all isolated envs need it)
    _HAS_TORCH = False
    try:
        import torch
        _HAS_TORCH = True
        wlog(f"[worker] Torch imported: {torch.__version__}")
    except Exception as e:
        wlog(f"[worker] Torch not available: {e}")

    # Setup log forwarding to host
    # This makes print() and logging statements in node code visible to the user
    import builtins
    import logging
    _original_print = builtins.print

    def _forwarded_print(*args, **kwargs):
        """Forward print() calls to host via socket."""
        # Build message from args
        sep = kwargs.get('sep', ' ')
        message = sep.join(str(a) for a in args)
        # Send to host
        try:
            transport.send({"type": "log", "message": message})
        except Exception:
            pass  # Don't fail if transport is closed
        # Also log locally for debugging
        wlog(f"[print] {message}")

    builtins.print = _forwarded_print

    # Also forward logging module output
    class SocketLogHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                transport.send({"type": "log", "message": msg})
                wlog(f"[log] {msg}")
            except Exception:
                pass

    # Add our handler to the root logger
    _socket_handler = SocketLogHandler()
    _socket_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.root.addHandler(_socket_handler)

    wlog("[worker] Print and logging forwarding enabled")

    # ---------------------------------------------------------------
    # Model registry — tracks nn.Module instances on CUDA so the main
    # process can command device moves via IPC for VRAM management.
    #
    # Auto-detection: hooks Module.to() and .cuda() to catch any
    # module that lands on CUDA.  No manual registration needed.
    # ---------------------------------------------------------------
    _model_registry = {}          # model_id -> nn.Module
    _model_registry_meta = {}     # model_id -> {"size": int, "kind": str}
    _model_id_by_obj = {}         # id(module) -> model_id  (dedup)
    _model_counter = [0]          # mutable counter in list for closure access
    _new_models_this_call = []    # populated during call, sent in response
    _loading_via_shim = [False]   # suppress auto-detection during shimmed load_models_gpu

    def _compute_model_size(model):
        """Compute size in bytes: parameters + buffers."""
        size = 0
        if hasattr(model, "parameters"):
            size += sum(p.numel() * p.element_size() for p in model.parameters())
        if hasattr(model, "buffers"):
            size += sum(b.numel() * b.element_size() for b in model.buffers())
        return size

    def _register_model(model_id, model, kind="other"):
        """Register a model explicitly (optional — auto-hook handles most cases)."""
        _model_registry[model_id] = model
        _model_id_by_obj[id(model)] = model_id
        size = _compute_model_size(model)
        _model_registry_meta[model_id] = {"size": size, "kind": kind}
        _new_models_this_call.append({"id": model_id, "size": size, "kind": kind})
        wlog(f"[worker] Registered model '{model_id}': {size / 1e9:.2f} GB, kind={kind}")
        return size

    def _auto_register_if_cuda(module):
        """Auto-register an nn.Module if it just landed on CUDA."""
        if _loading_via_shim[0]:
            return  # Parent already coordinates VRAM during shimmed loads
        obj_id = id(module)
        if obj_id in _model_id_by_obj:
            return  # Already registered
        try:
            first_param = next(module.parameters(), None)
            if first_param is None or first_param.device.type != "cuda":
                return
        except Exception:
            return
        _model_counter[0] += 1
        model_id = f"{module.__class__.__name__}_{_model_counter[0]}"
        size = _compute_model_size(module)
        _model_registry[model_id] = module
        _model_registry_meta[model_id] = {"size": size, "kind": "other"}
        _model_id_by_obj[obj_id] = model_id
        _new_models_this_call.append({"id": model_id, "size": size, "kind": "other"})
        wlog(f"[worker] Auto-registered '{model_id}': {size / 1e9:.2f} GB")

    # Install hooks on Module.to() and .cuda()
    # Module.to() only fires for the outermost call — PyTorch recurses
    # through children via _apply(), not .to(), so we naturally catch
    # only top-level models.
    try:
        import torch as _torch
        _orig_module_to = _torch.nn.Module.to
        _orig_module_cuda = _torch.nn.Module.cuda

        def _hooked_to(self, *args, **kwargs):
            result = _orig_module_to(self, *args, **kwargs)
            _auto_register_if_cuda(self)
            return result

        def _hooked_cuda(self, *args, **kwargs):
            result = _orig_module_cuda(self, *args, **kwargs)
            _auto_register_if_cuda(self)
            return result

        _torch.nn.Module.to = _hooked_to
        _torch.nn.Module.cuda = _hooked_cuda
        wlog("[worker] Installed Module.to()/cuda() auto-registration hooks")
    except ImportError:
        wlog("[worker] torch not available, skipping auto-registration hooks")

    # ---------------------------------------------------------------
    # Bidirectional RPC — call parent methods during execution
    # ---------------------------------------------------------------
    def _call_parent(method, **params):
        """Call a method on the parent process and wait for result.

        Can only be called during method execution (while transport is active).
        The parent handles the callback and sends back a response.
        """
        transport.send({"type": "callback", "method": method, **params})
        response = transport.recv()
        if response is None:
            raise RuntimeError("Parent disconnected during callback")
        if response.get("type") != "callback_response":
            raise RuntimeError(f"Expected callback_response, got {response.get('type')}")
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Callback failed"))
        return response.get("result")

    # ---------------------------------------------------------------
    # Auto-enable fastest attention backend before comfy modules are
    # imported.  In the subprocess, comfy.cli_args.args is parsed with
    # empty argv so --use-sage-attention / --use-flash-attention are
    # False.  Setting them here lets comfy.ldm.modules.attention pick
    # up sage/flash when it is first imported below.
    # ---------------------------------------------------------------
    try:
        import torch as _torch_check
        if _torch_check.cuda.is_available() and _torch_check.cuda.get_device_capability()[0] >= 8:
            from comfy.cli_args import args as _cli_args
            try:
                import sageattention  # noqa: F401
                _cli_args.use_sage_attention = True
                wlog("[worker] Auto-enabled sage attention")
            except ImportError:
                pass
            try:
                import flash_attn  # noqa: F401
                _cli_args.use_flash_attention = True
                wlog("[worker] Auto-enabled flash attention")
            except ImportError:
                pass
    except Exception:
        pass

    # ---------------------------------------------------------------
    # Shim comfy.model_management.load_models_gpu — tell parent to
    # make room first, then let the real load_models_gpu handle the
    # actual loading (it already calculates lowvram_model_memory from
    # get_free_memory internally).
    # This eliminates dual VRAM management (subprocess + parent).
    # ---------------------------------------------------------------
    try:
        import comfy.model_management as _cmm
        _original_load_models_gpu = _cmm.load_models_gpu

        def _shimmed_load_models_gpu(models, *args, **kwargs):
            """Ask parent to free VRAM, then run real load_models_gpu."""
            _loading_via_shim[0] = True
            try:
                model_info = []
                for m in models:
                    size = m.model_size() if hasattr(m, 'model_size') else 0
                    model_info.append({"size": size, "key": str(id(m))})

                total_size = sum(mi["size"] for mi in model_info)
                wlog(f"[worker] load_models_gpu shim: {len(models)} models, {total_size / 1e9:.2f} GB total")

                # Ask parent to evict its models and make room
                result = _call_parent("request_vram_budget",
                             model_info=model_info,
                             total_size=total_size)

                # Propagate parent's VRAM constraints to subprocess
                if result:
                    extra_reserved = result.get("extra_reserved_vram")
                    if extra_reserved is not None:
                        _cmm.EXTRA_RESERVED_VRAM = extra_reserved
                        wlog(f"[worker] Set EXTRA_RESERVED_VRAM = {extra_reserved / 1e9:.2f} GB")

                    parent_vram_state = result.get("vram_state")
                    if parent_vram_state:
                        try:
                            _cmm.vram_state = _cmm.VRAMState[parent_vram_state]
                            wlog(f"[worker] Set vram_state = {parent_vram_state}")
                        except (KeyError, AttributeError):
                            pass

                # Now run the real load_models_gpu — it calls get_free_memory()
                # which uses EXTRA_RESERVED_VRAM via minimum_inference_memory(),
                # so it will calculate lowvram_model_memory correctly.
                _original_load_models_gpu(models, *args, **kwargs)
                wlog(f"[worker] Models loaded via real load_models_gpu")
            finally:
                _loading_via_shim[0] = False

        _cmm.load_models_gpu = _shimmed_load_models_gpu
        wlog("[worker] Installed load_models_gpu shim (budget-based)")
    except ImportError:
        wlog("[worker] comfy.model_management not available, skipping load_models_gpu shim")

    # Set up progress bar forwarding to parent process.
    # The subprocess's comfy.utils.PROGRESS_BAR_HOOK is None (server.py never ran here).
    # Setting it lets any ProgressBar created in subprocess code (e.g. stages.py)
    # automatically forward updates to the parent, which relays to the ComfyUI frontend.
    try:
        import comfy.utils as _cu
        class _InterruptedError(RuntimeError):
            """Raised when the user cancels the current run."""
            pass
        def _progress_hook(value, total, preview=None, node_id=None):
            try:
                _call_parent("report_progress", value=value, total=total)
            except RuntimeError as e:
                if "interrupted" in str(e).lower():
                    raise _InterruptedError(str(e))
            except Exception:
                pass
        _cu.set_progress_bar_global_hook(_progress_hook)
        wlog("[worker] Installed progress bar hook (forwards to parent)")
    except ImportError:
        wlog("[worker] comfy.utils not available, skipping progress hook")

    # Expose explicit API as comfy_worker module (optional override)
    import types as _types
    _comfy_worker = _types.ModuleType("comfy_worker")
    _comfy_worker.__doc__ = "Helper for registering models with the comfy-env worker."
    _comfy_worker.register_model = _register_model
    _comfy_worker.call_parent = _call_parent
    sys.modules["comfy_worker"] = _comfy_worker

    # Signal ready
    transport.send({"status": "ready"})
    wlog("[worker] Ready, entering request loop...")

    # Process requests
    request_num = 0
    while True:
        request_num += 1
        wlog(f"[worker] Waiting for request #{request_num}...")
        try:
            request = transport.recv()
            if not request:
                wlog("[worker] Empty request received, exiting loop")
                break
        except Exception as e:
            wlog(f"[worker] Exception receiving request: {e}")
            break

        if request.get("method") == "shutdown":
            wlog("[worker] Shutdown requested")
            break

        if request.get("method") == "ping":
            # Health check - respond immediately
            transport.send({"status": "pong"})
            continue

        if request.get("method") == "model_to_device":
            # Move a registered model to a device (cuda/cpu)
            _mid = request.get("model_id")
            _target = request.get("device", "cpu")
            _model = _model_registry.get(_mid)
            if _model is None:
                transport.send({"status": "error",
                                "error": f"Model '{_mid}' not registered"})
                continue
            try:
                import torch as _torch
                _target_dev = _torch.device(_target)
                # Check if already on target device — idempotent
                _current_dev = None
                try:
                    _first_param = next(_model.parameters(), None)
                    if _first_param is not None:
                        _current_dev = _first_param.device
                except Exception:
                    pass
                if _current_dev is not None and _current_dev == _target_dev:
                    wlog(f"[worker] model_to_device: '{_mid}' already on {_target}")
                    transport.send({"status": "ok", "device": _target, "moved": False})
                    continue
                _was_cuda = _current_dev is not None and _current_dev.type == "cuda"
                wlog(f"[worker] model_to_device: '{_mid}' -> {_target}")
                _model.to(_target_dev)
                # Only empty cache if we actually freed CUDA tensors
                if _was_cuda and _target_dev.type == "cpu":
                    _torch.cuda.empty_cache()
                transport.send({"status": "ok", "device": _target, "moved": True})
            except Exception as _e:
                wlog(f"[worker] model_to_device error: {_e}")
                transport.send({"status": "error", "error": str(_e)})
            continue

        if request.get("method") == "list_models":
            # Return registered model metadata
            transport.send({"status": "ok", "models": _model_registry_meta})
            continue

        # Release input shm blocks from previous request
        for _old_block in _input_shm_blocks:
            try:
                _old_block.close()
            except Exception:
                pass
        _input_shm_blocks.clear()

        # Clear new-models tracker for this call
        _new_models_this_call.clear()

        shm_registry = []
        try:
            request_type = request.get("type", "call_module")
            module_name = request["module"]
            wlog(f"[worker] Request: {request_type} {module_name}")

            # Load inputs from shared memory
            kwargs_meta = request.get("kwargs")
            if kwargs_meta:
                wlog(f"[worker] Reconstructing inputs from shm...")
                inputs = _from_shm(kwargs_meta)
                inputs = _deserialize_isolated_objects(inputs)
                inputs = _deserialize_input(inputs)
                wlog(f"[worker] Inputs ready: {list(inputs.keys()) if isinstance(inputs, dict) else type(inputs)}")
                # Debug: log tensor shapes
                if isinstance(inputs, dict):
                    for k, v in inputs.items():
                        if hasattr(v, 'shape'):
                            wlog(f"[worker] Input '{k}' shape: {v.shape}")
            else:
                inputs = {}

            # Import module
            wlog(f"[worker] Importing module {module_name}...")
            module = importlib.import_module(module_name)
            wlog(f"[worker] Module imported")

            if request_type == "call_method":
                class_name = request["class_name"]
                method_name = request["method_name"]
                self_state = request.get("self_state")
                wlog(f"[worker] Getting class {class_name}...")

                cls = getattr(module, class_name)
                wlog(f"[worker] Creating instance...")
                instance = object.__new__(cls)
                if self_state:
                    self_state = _deserialize_isolated_objects(self_state)
                    instance.__dict__.update(self_state)
                wlog(f"[worker] Calling {method_name}...")
                method = getattr(instance, method_name)
                result = method(**inputs)
                wlog(f"[worker] Method returned")
            else:
                func_name = request["func"]
                func = getattr(module, func_name)
                result = func(**inputs)

            # Serialize result to shared memory
            wlog(f"[worker] Serializing result to shm...")
            result_meta = _to_shm(result, shm_registry)
            wlog(f"[worker] Created {len(shm_registry)} shm blocks for result")

            response = {"status": "ok", "result": result_meta}
            if _new_models_this_call:
                response["_new_models"] = list(_new_models_this_call)
            transport.send(response)
            _shm_keeper.keep(shm_registry)  # Keep alive for 30s until host reads

        except Exception as e:
            # Cleanup shm on error since host won't read it
            _cleanup_shm(shm_registry)
            transport.send({
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    transport.close()

if __name__ == "__main__":
    main()
'''


class SubprocessWorker(Worker):
    """
    Cross-venv worker using persistent subprocess + socket IPC.

    Uses Unix domain sockets (or TCP localhost on older Windows) for IPC.
    This completely separates IPC from stdout/stderr, so C libraries
    printing to stdout (like Blender) won't corrupt the protocol.

    Benefits:
    - Works on Windows with different venv Python (full isolation)
    - Compiled CUDA extensions load correctly in the venv
    - ~50-100ms per call (persistent subprocess avoids spawn overhead)
    - Tensor transfer via shared memory files
    - Immune to stdout pollution from C libraries

    Use this for calls to isolated venvs with different Python/dependencies.
    """

    def __init__(
        self,
        python: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
        sys_path: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        share_torch: bool = True,  # Kept for API compatibility
        health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT,
    ):
        """
        Initialize persistent worker.

        Args:
            python: Path to Python executable in target venv.
            working_dir: Working directory for subprocess.
            sys_path: Additional paths to add to sys.path.
            env: Additional environment variables.
            name: Optional name for logging.
            share_torch: Ignored (kept for API compatibility).
            health_check_timeout: Timeout in seconds for worker health checks.
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"SubprocessWorker({self.python.parent.parent.name})"
        self.health_check_timeout = health_check_timeout

        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_pvenv_'))
        self._shm_dir = _get_shm_dir()
        self._process: Optional[subprocess.Popen] = None
        self._shutdown = False
        self._lock = threading.Lock()
        self._last_new_models = []  # Auto-detected models from last call
        self._callback_handlers: Dict[str, Callable] = {}  # Bidirectional RPC callbacks

        # Socket IPC
        self._server_socket: Optional[socket.socket] = None
        self._socket_addr: Optional[str] = None
        self._transport: Optional[SocketTransport] = None

        # Stderr inherits from parent (no pipe -- avoids tqdm/\r deadlock)

        # Write worker script to temp file
        self._worker_script = self._temp_dir / "persistent_worker.py"
        self._worker_script.write_text(_PERSISTENT_WORKER_SCRIPT, encoding="utf-8")

    def _find_comfyui_base(self) -> Optional[Path]:
        """Find ComfyUI base directory."""
        # Use folder_paths.base_path (canonical source) if available
        try:
            import folder_paths
            return Path(folder_paths.base_path)
        except ImportError:
            pass

        # Fallback: Check common child directories (for test environments)
        for base in [self.working_dir, self.working_dir.parent]:
            for child in [".comfy-test-env/ComfyUI", "ComfyUI"]:
                candidate = base / child
                if (candidate / "main.py").exists() and (candidate / "comfy").exists():
                    return candidate

        # Fallback: Walk up from working_dir (standard ComfyUI custom_nodes layout)
        current = self.working_dir.resolve()
        for _ in range(10):
            if (current / "main.py").exists() and (current / "comfy").exists():
                return current
            current = current.parent
        return None

    def _check_socket_health(self) -> bool:
        """Check if socket connection is healthy using a quick ping."""
        if not self._transport:
            print(f"[{self.name}] Health check: no transport", file=sys.stderr, flush=True)
            return False
        try:
            # Send a ping request with configurable timeout
            print(f"[{self.name}] Health check: ping (timeout={self.health_check_timeout}s)...", file=sys.stderr, flush=True)
            self._transport.send({"method": "ping"})
            response = self._transport.recv(timeout=self.health_check_timeout)
            ok = response is not None and response.get("status") == "pong"
            print(f"[{self.name}] Health check: {'ok' if ok else 'failed'}", file=sys.stderr, flush=True)
            return ok
        except Exception as e:
            print(f"[{self.name}] Socket health check exception: {e}", file=sys.stderr, flush=True)
            return False

    def _kill_worker(self) -> None:
        """Kill the worker process and clean up resources."""
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except:
                pass
            self._process = None
        if self._transport:
            try:
                self._transport.close()
            except:
                pass
            self._transport = None
        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
            self._server_socket = None

    def _ensure_started(self):
        """Start persistent worker subprocess if not running."""
        if self._shutdown:
            raise RuntimeError(f"{self.name}: Worker has been shut down")

        if self._process is not None and self._process.poll() is None:
            # Process is running, but check if socket is healthy
            if self._transport and self._check_socket_health():
                return  # All good
            # Socket is dead/unhealthy - restart worker
            print(f"[{self.name}] Socket unhealthy, restarting worker...", file=sys.stderr, flush=True)
            self._kill_worker()

        # Clean up any previous socket
        if self._transport:
            self._transport.close()
            self._transport = None
        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

        # Create server socket for IPC
        self._server_socket, self._socket_addr = _create_server_socket()

        # Set up environment (shared with metadata scan)
        from ..wrap import build_isolation_env
        env = build_isolation_env(self.python, self.extra_env)

        # Find ComfyUI base and add to sys_path for real folder_paths/comfy modules
        # This works because comfy.options.args_parsing=False by default, so folder_paths
        # auto-detects its base directory from __file__ location
        comfyui_base = self._find_comfyui_base()
        if comfyui_base:
            env["COMFYUI_BASE"] = str(comfyui_base)  # Keep for fallback/debugging

        # Build sys_path: ComfyUI first (for real modules), then working_dir, then extras
        all_sys_path = []
        if comfyui_base:
            all_sys_path.append(str(comfyui_base))
        all_sys_path.append(str(self.working_dir))
        all_sys_path.extend(self.sys_path)

        print(f"[{self.name}] python: {self.python}", flush=True)
        print(f"[{self.name}] sys_path sent to worker: {all_sys_path}", flush=True)

        # Launch subprocess with the venv Python, passing socket address
        # For pixi environments, use "pixi run python" to get proper environment activation
        # (CONDA_PREFIX, Library paths, etc.) which fixes DLL loading issues with bpy
        is_pixi = '.pixi' in str(self.python)
        if _DEBUG:
            print(f"[SubprocessWorker] is_pixi={is_pixi}, python={self.python}", flush=True)
        if is_pixi:
            # Find pixi project root (parent of .pixi directory)
            pixi_project = self.python
            while pixi_project.name != '.pixi' and pixi_project.parent != pixi_project:
                pixi_project = pixi_project.parent
            pixi_project = pixi_project.parent  # Go up from .pixi to project root
            pixi_toml = pixi_project / "pixi.toml"
            if _DEBUG:
                print(f"[SubprocessWorker] pixi_toml={pixi_toml}, exists={pixi_toml.exists()}", flush=True)

            if pixi_toml.exists():
                pixi_exe = get_pixi_path()
                if pixi_exe is None:
                    raise WorkerError("pixi not found - required for isolated environment execution")
                cmd = [str(pixi_exe), "run", "--manifest-path", str(pixi_toml),
                       "python", str(self._worker_script), self._socket_addr]
                # Clean PATH to remove ct-env entries that have conflicting DLLs
                # Pixi will add its own environment paths
                path_sep = ";" if sys.platform == "win32" else ":"
                current_path = env.get("PATH", "")
                # Filter out ct-envs and conda/mamba paths that could conflict
                clean_path_parts = [
                    p for p in current_path.split(path_sep)
                    if not any(x in p.lower() for x in (".ct-envs", "conda", "mamba", "miniforge", "miniconda", "anaconda"))
                ]
                env["PATH"] = path_sep.join(clean_path_parts)
                launch_env = env
            else:
                cmd = [str(self.python), str(self._worker_script), self._socket_addr]
                launch_env = env
        else:
            cmd = [str(self.python), str(self._worker_script), self._socket_addr]
            launch_env = env

        if _DEBUG:
            print(f"[SubprocessWorker] launching cmd={cmd[:3]}...", flush=True)
            if launch_env:
                path_sep = ";" if sys.platform == "win32" else ":"
                path_parts = launch_env.get("PATH", "").split(path_sep)
                print(f"[SubprocessWorker] PATH has {len(path_parts)} entries:", flush=True)
                for i, p in enumerate(path_parts[:10]):  # Show first 10
                    print(f"[SubprocessWorker]   [{i}] {p}", flush=True)
                if len(path_parts) > 10:
                    print(f"[SubprocessWorker]   ... and {len(path_parts) - 10} more", flush=True)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,  # Inherit parent stderr (avoids pipe deadlock with tqdm)
            cwd=str(self.working_dir),
            env=launch_env,
        )

        # Accept connection from worker with timeout
        self._server_socket.settimeout(60)
        try:
            client_sock, _ = self._server_socket.accept()
        except socket.timeout:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except:
                pass
            raise RuntimeError(f"{self.name}: Worker failed to connect (timeout). Check stderr output above.")
        finally:
            self._server_socket.settimeout(None)

        self._transport = SocketTransport(client_sock)

        # Send config
        config = {
            "sys_paths": all_sys_path,
        }
        self._transport.send(config)

        # Wait for ready signal (skip any log messages from worker startup)
        while True:
            msg = self._transport.recv(timeout=60)
            if not msg:
                raise RuntimeError(f"{self.name}: Worker failed to send ready signal")
            if msg.get("type") == "log":
                # Worker sends log messages during startup (e.g. comfy imports)
                print(f"[worker:{self.name}] {msg.get('message', '')}", file=sys.stderr, flush=True)
                continue
            break

        if msg.get("status") != "ready":
            raise RuntimeError(f"{self.name}: Unexpected ready message: {msg}")

    def call(
        self,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Not supported - use call_module()."""
        raise NotImplementedError(
            f"{self.name}: Use call_module(module='...', func='...') instead."
        )

    def register_callback(self, method: str, handler: Callable) -> None:
        """Register a handler for worker callbacks (bidirectional RPC).

        When the worker calls _call_parent(method, ...) during execution,
        the parent dispatches to the registered handler and sends the result back.

        Args:
            method: Callback method name (e.g., "request_vram_budget").
            handler: Function(request_dict) -> result_dict.
        """
        self._callback_handlers[method] = handler

    def _handle_callback(self, request: dict) -> dict:
        """Execute a callback request from the worker."""
        method = request.get("method")
        handler = self._callback_handlers.get(method)
        if not handler:
            return {"type": "callback_response", "status": "error",
                    "error": f"Unknown callback method: {method}"}
        try:
            result = handler(request)
            return {"type": "callback_response", "status": "ok", "result": result}
        except Exception as e:
            return {"type": "callback_response", "status": "error", "error": str(e)}

    def _send_request(self, request: dict, timeout: float) -> dict:
        """Send request via socket and read response with timeout."""
        if not self._transport:
            raise RuntimeError(f"{self.name}: Transport not initialized")

        # Send request
        self._transport.send(request)

        # Read response with timeout, handling log/callback messages along the way
        try:
            while True:
                response = self._transport.recv(timeout=timeout)
                if response is None:
                    break  # Timeout

                # Handle log messages from worker
                if response.get("type") == "log":
                    msg = response.get("message", "")
                    print(f"[worker:{self.name}] {msg}", file=sys.stderr, flush=True)
                    continue  # Keep waiting for actual response

                # Handle callback from worker (bidirectional RPC)
                if response.get("type") == "callback":
                    callback_response = self._handle_callback(response)
                    self._transport.send(callback_response)
                    continue  # Keep waiting for actual response

                # Got a real response
                break
        except ConnectionError as e:
            # Socket closed - check if worker process died
            self._shutdown = True
            exit_code = None
            if self._process:
                exit_code = self._process.poll()

            if exit_code is not None:
                raise RuntimeError(
                    f"{self.name}: Worker process died with exit code {exit_code}. "
                    f"This usually indicates a crash in native code (CGAL, pymeshlab, etc.). "
                    f"Check stderr output above."
                ) from e
            else:
                raise RuntimeError(
                    f"{self.name}: Socket closed but worker process still running. "
                    f"This may indicate a protocol error or worker bug."
                ) from e

        if response is None:
            # Timeout - kill process
            try:
                self._process.kill()
            except:
                pass
            self._shutdown = True
            raise TimeoutError(f"{self.name}: Call timed out after {timeout}s")

        return response

    def call_method(
        self,
        module_name: str,
        class_name: str,
        method_name: str,
        self_state: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Call a class method by module/class/method path.

        Args:
            module_name: Module containing the class (e.g., "depth_estimate").
            class_name: Class name (e.g., "SAM3D_DepthEstimate").
            method_name: Method name (e.g., "estimate_depth").
            self_state: Optional dict to populate instance __dict__.
            kwargs: Keyword arguments for the method.
            timeout: Timeout in seconds.

        Returns:
            Return value of the method.
        """
        import sys
        if _DEBUG:
            print(f"[SubprocessWorker] call_method: {module_name}.{class_name}.{method_name}", file=sys.stderr, flush=True)

        with self._lock:
            if _DEBUG:
                print(f"[SubprocessWorker] acquired lock, ensuring started...", file=sys.stderr, flush=True)
            self._ensure_started()
            if _DEBUG:
                print(f"[SubprocessWorker] worker started/confirmed", file=sys.stderr, flush=True)

            timeout = timeout or 600.0
            shm_registry = []

            try:
                # Serialize kwargs to shared memory
                if kwargs:
                    if _DEBUG:
                        for k, v in kwargs.items():
                            if hasattr(v, 'shape'):
                                print(f"[comfy-env] PRE-SERIALIZE '{k}' shape: {v.shape}", file=sys.stderr, flush=True)
                    if _DEBUG:
                        print(f"[SubprocessWorker] serializing kwargs to shm...", file=sys.stderr, flush=True)
                    kwargs_meta = _to_shm(kwargs, shm_registry)
                    if _DEBUG:
                        print(f"[SubprocessWorker] created {len(shm_registry)} shm blocks", file=sys.stderr, flush=True)
                else:
                    kwargs_meta = None

                # Send request with shared memory metadata
                request = {
                    "type": "call_method",
                    "module": module_name,
                    "class_name": class_name,
                    "method_name": method_name,
                    "self_state": _serialize_for_ipc(self_state) if self_state else None,
                    "kwargs": kwargs_meta,
                }
                if _DEBUG:
                    print(f"[SubprocessWorker] sending request via socket...", file=sys.stderr, flush=True)
                response = self._send_request(request, timeout)
                if _DEBUG:
                    print(f"[SubprocessWorker] got response: {response.get('status')}", file=sys.stderr, flush=True)

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                # Store newly auto-registered models (for proxy to create patchers)
                self._last_new_models = response.get("_new_models", [])

                # Reconstruct result from shared memory
                result_meta = response.get("result")
                if result_meta is not None:
                    return _from_shm(result_meta)
                return None

            finally:
                _cleanup_shm(shm_registry)

    def call_module(
        self,
        module: str,
        func: str,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """Call a function by module path."""
        with self._lock:
            self._ensure_started()

            timeout = timeout or 600.0
            shm_registry = []

            try:
                kwargs_meta = _to_shm(kwargs, shm_registry) if kwargs else None

                request = {
                    "type": "call_module",
                    "module": module,
                    "func": func,
                    "kwargs": kwargs_meta,
                }
                response = self._send_request(request, timeout)

                if response.get("status") == "error":
                    raise WorkerError(
                        response.get("error", "Unknown"),
                        traceback=response.get("traceback"),
                    )

                result_meta = response.get("result")
                if result_meta is not None:
                    return _from_shm(result_meta)
                return None

            finally:
                _cleanup_shm(shm_registry)

    def send_command(self, method, **params):
        """Send a management command to the worker (model device moves, etc.).

        Uses the same socket transport as call_method but expects a simple
        JSON response (no shared-memory result).
        """
        with self._lock:
            self._ensure_started()
            request = {"method": method, **params}
            response = self._send_request(request, timeout=60.0)
            if response.get("status") == "error":
                raise RuntimeError(
                    f"{self.name}: Command '{method}' failed: "
                    f"{response.get('error', 'Unknown error')}"
                )
            return response

    def shutdown(self) -> None:
        """Shut down the persistent worker."""
        if self._shutdown:
            return
        self._shutdown = True

        # Send shutdown signal via socket
        if self._transport and self._process and self._process.poll() is None:
            try:
                self._transport.send({"method": "shutdown"})
            except:
                pass

        # Close transport and socket
        if self._transport:
            self._transport.close()
            self._transport = None

        if self._server_socket:
            try:
                self._server_socket.close()
            except:
                pass
            # Clean up unix socket file
            if self._socket_addr and self._socket_addr.startswith("unix://"):
                try:
                    Path(self._socket_addr[7:]).unlink()
                except:
                    pass
            self._server_socket = None

        # Wait for process to exit
        if self._process and self._process.poll() is None:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass

        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def is_alive(self) -> bool:
        if self._shutdown:
            return False
        if self._process is None:
            return False
        return self._process.poll() is None

    def __repr__(self):
        status = "alive" if self.is_alive() else "stopped"
        return f"<SubprocessWorker name={self.name!r} status={status}>"
