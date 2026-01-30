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
from ..pixi import get_pixi_path

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
import numpy as np


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

    # numpy array → direct shared memory
    if t == 'ndarray':
        arr = np.ascontiguousarray(obj)
        block = shm.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
        registry.append(block)
        result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        visited[obj_id] = result
        return result

    # torch.Tensor → convert to numpy → shared memory (with marker to restore type)
    if t == 'Tensor':
        arr = obj.detach().cpu().numpy()
        result = _to_shm(arr, registry, visited)
        result["__was_tensor__"] = True
        return result

    # trimesh.Trimesh → pickle → shared memory (preserves visual, metadata, normals)
    if t == 'Trimesh':
        import pickle
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

    # Path → string
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

    # primitives pass through
    return obj


def _from_shm(obj, unlink=True):
    """Reconstruct object from shared memory metadata."""
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_from_shm(v, unlink) for v in obj]
        return obj

    # numpy array (or tensor that was converted to numpy)
    if "__shm_np__" in obj:
        block = shm.SharedMemory(name=obj["__shm_np__"])
        arr = np.ndarray(tuple(obj["shape"]), dtype=np.dtype(obj["dtype"]), buffer=block.buf).copy()
        block.close()
        if unlink:
            block.unlink()
        # Convert back to tensor if it was originally a tensor
        if obj.get("__was_tensor__"):
            import torch
            return torch.from_numpy(arr)
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

    # Handle Path objects - convert to string for JSON serialization
    from pathlib import PurePath
    if isinstance(obj, PurePath):
        return str(obj)

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
from types import SimpleNamespace

# Enable faulthandler to dump traceback on SIGSEGV/SIGABRT/etc
faulthandler.enable(file=sys.stderr, all_threads=True)

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

        # Also print
        print(f"\\n=== WATCHDOG TICK {tick} ===", flush=True)
        print(dump, flush=True)
        print("=== END ===\\n", flush=True)

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

def _to_shm(obj, registry, visited=None):
    """Serialize to shared memory. Returns JSON-safe metadata."""
    if visited is None:
        visited = {}
    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]
    t = type(obj).__name__

    if t == 'ndarray':
        arr = np.ascontiguousarray(obj)
        block = shm.SharedMemory(create=True, size=arr.nbytes)
        np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
        registry.append(block)
        result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        visited[obj_id] = result
        return result

    if t == 'Tensor':
        arr = obj.detach().cpu().numpy()
        result = _to_shm(arr, registry, visited)
        result["__was_tensor__"] = True
        return result

    # trimesh.Trimesh → pickle → shared memory (preserves visual, metadata, normals)
    if t == 'Trimesh':
        import pickle
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

    if isinstance(obj, dict):
        result = {k: _to_shm(v, registry, visited) for k, v in obj.items()}
        visited[obj_id] = result
        return result
    if isinstance(obj, (list, tuple)):
        result = [_to_shm(v, registry, visited) for v in obj]
        visited[obj_id] = result
        return result

    return obj

def _from_shm(obj):
    """Reconstruct from shared memory metadata. Does NOT unlink - caller handles that."""
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_from_shm(v) for v in obj]
        return obj
    if "__shm_np__" in obj:
        block = shm.SharedMemory(name=obj["__shm_np__"])
        arr = np.ndarray(tuple(obj["shape"]), dtype=np.dtype(obj["dtype"]), buffer=block.buf).copy()
        block.close()
        # Convert back to tensor if it was originally a tensor
        if obj.get("__was_tensor__"):
            import torch
            return torch.from_numpy(arr)
        return arr
    # trimesh (pickled to preserve visual, metadata, normals)
    if "__shm_trimesh__" in obj:
        import pickle
        block = shm.SharedMemory(name=obj["name"])
        mesh_bytes = bytes(block.buf[:obj["size"]])
        block.close()
        return pickle.loads(mesh_bytes)
    return {k: _from_shm(v) for k, v in obj.items()}

def _cleanup_shm(registry):
    for block in registry:
        try:
            block.close()
            block.unlink()
        except Exception:
            pass
    registry.clear()

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
    except ImportError:
        wlog("[worker] Torch not available, using pickle for serialization")

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
            else:
                inputs = {}

            # Import module
            wlog(f"[worker] Importing module {module_name}...")
            module = __import__(module_name, fromlist=[""])
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

            transport.send({"status": "ok", "result": result_meta})
            # Note: don't cleanup shm_registry here - host needs to read it

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
        """
        self.python = Path(python)
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.sys_path = sys_path or []
        self.extra_env = env or {}
        self.name = name or f"SubprocessWorker({self.python.parent.parent.name})"

        if not self.python.exists():
            raise FileNotFoundError(f"Python not found: {self.python}")

        self._temp_dir = Path(tempfile.mkdtemp(prefix='comfyui_pvenv_'))
        self._shm_dir = _get_shm_dir()
        self._process: Optional[subprocess.Popen] = None
        self._shutdown = False
        self._lock = threading.Lock()

        # Socket IPC
        self._server_socket: Optional[socket.socket] = None
        self._socket_addr: Optional[str] = None
        self._transport: Optional[SocketTransport] = None

        # Stderr buffer for crash diagnostics
        self._stderr_buffer: List[str] = []
        self._stderr_lock = threading.Lock()

        # Write worker script to temp file
        self._worker_script = self._temp_dir / "persistent_worker.py"
        self._worker_script.write_text(_PERSISTENT_WORKER_SCRIPT)

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
            return False
        try:
            # Send a ping request with short timeout
            self._transport.send({"method": "ping"})
            response = self._transport.recv(timeout=2.0)
            return response is not None and response.get("status") == "pong"
        except Exception as e:
            print(f"[{self.name}] Socket health check failed: {e}", file=sys.stderr, flush=True)
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

        # Set up environment
        env = os.environ.copy()
        env.update(self.extra_env)
        env["COMFYUI_ISOLATION_WORKER"] = "1"

        # For conda/pixi environments, add lib dir to LD_LIBRARY_PATH
        # This ensures libraries like libstdc++ from the env are used
        lib_dir = self.python.parent.parent / "lib"
        if lib_dir.is_dir():
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else str(lib_dir)

        # On Windows, pass host Python directory so worker can add it via os.add_dll_directory()
        # This fixes "DLL load failed" errors for packages like opencv-python-headless
        if sys.platform == "win32":
            env["COMFYUI_HOST_PYTHON_DIR"] = str(Path(sys.executable).parent)

            # For pixi environments with MKL, add Library/bin to PATH for DLL loading
            # MKL DLLs are in .pixi/envs/default/Library/bin/
            # Pixi has python.exe directly in env dir, not in Scripts/
            env_dir = self.python.parent
            library_bin = env_dir / "Library" / "bin"

            # COMPLETE DLL ISOLATION: Build minimal PATH from scratch
            # Only include Windows system directories + pixi environment
            # This prevents DLL conflicts from mingw, conda, etc.
            windir = os.environ.get("WINDIR", r"C:\Windows")
            minimal_path_parts = [
                str(env_dir),  # Pixi env (python.exe location)
                str(env_dir / "Scripts"),  # Pixi Scripts
                str(env_dir / "Lib" / "site-packages" / "bpy"),  # bpy DLLs
                f"{windir}\\System32",  # Core Windows DLLs
                f"{windir}",  # Windows directory
                f"{windir}\\System32\\Wbem",  # WMI tools
            ]
            if library_bin.is_dir():
                minimal_path_parts.insert(1, str(library_bin))  # MKL DLLs

            env["PATH"] = ";".join(minimal_path_parts)
            env["COMFYUI_PIXI_LIBRARY_BIN"] = str(library_bin) if library_bin.is_dir() else ""
            # Allow duplicate OpenMP libraries (MKL's libiomp5md.dll + PyTorch's libomp.dll)
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            # Use UTF-8 encoding for stdout/stderr to handle Unicode symbols
            env["PYTHONIOENCODING"] = "utf-8"

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
            stdout=subprocess.DEVNULL,  # DEVNULL to prevent pipe buffer deadlock
            stderr=subprocess.PIPE,  # Capture stderr separately for crash diagnostics
            cwd=str(self.working_dir),
            env=launch_env,
        )

        # Clear stderr buffer for new process
        with self._stderr_lock:
            self._stderr_buffer.clear()

        # Start stderr capture thread (buffer for crash diagnostics)
        def capture_stderr():
            try:
                for line in self._process.stderr:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='replace')
                    # Print to terminal AND buffer for crash reporting
                    sys.stderr.write(f"  [stderr] {line}")
                    sys.stderr.flush()
                    with self._stderr_lock:
                        self._stderr_buffer.append(line.rstrip())
                        # Keep last 50 lines
                        if len(self._stderr_buffer) > 50:
                            self._stderr_buffer.pop(0)
            except:
                pass
        self._stderr_thread = threading.Thread(target=capture_stderr, daemon=True)
        self._stderr_thread.start()

        # Accept connection from worker with timeout
        self._server_socket.settimeout(60)
        try:
            client_sock, _ = self._server_socket.accept()
        except socket.timeout:
            # Collect stderr from buffer
            time.sleep(0.2)  # Give stderr thread time to capture
            with self._stderr_lock:
                stderr = "\n".join(self._stderr_buffer) if self._stderr_buffer else "(no stderr captured)"
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except:
                pass
            raise RuntimeError(f"{self.name}: Worker failed to connect (timeout).\nStderr:\n{stderr}")
        finally:
            self._server_socket.settimeout(None)

        self._transport = SocketTransport(client_sock)

        # Send config
        config = {
            "sys_paths": all_sys_path,
        }
        self._transport.send(config)

        # Wait for ready signal
        msg = self._transport.recv(timeout=60)
        if not msg:
            raise RuntimeError(f"{self.name}: Worker failed to send ready signal")

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

    def _send_request(self, request: dict, timeout: float) -> dict:
        """Send request via socket and read response with timeout."""
        if not self._transport:
            raise RuntimeError(f"{self.name}: Transport not initialized")

        # Send request
        self._transport.send(request)

        # Read response with timeout, handling log messages along the way
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

                # Got a real response
                break
        except ConnectionError as e:
            # Socket closed - check if worker process died
            self._shutdown = True
            time.sleep(0.2)  # Give process time to fully exit and stderr to flush
            exit_code = None
            if self._process:
                exit_code = self._process.poll()

            # Get captured stderr
            with self._stderr_lock:
                stderr_output = "\n".join(self._stderr_buffer) if self._stderr_buffer else "(no stderr captured)"

            if exit_code is not None:
                raise RuntimeError(
                    f"{self.name}: Worker process died with exit code {exit_code}. "
                    f"This usually indicates a crash in native code (CGAL, pymeshlab, etc.).\n"
                    f"Stderr:\n{stderr_output}"
                ) from e
            else:
                # Process still alive but socket closed - something weird
                raise RuntimeError(
                    f"{self.name}: Socket closed but worker process still running. "
                    f"This may indicate a protocol error or worker bug.\n"
                    f"Stderr:\n{stderr_output}"
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
                self._process.kill()
                self._process.wait(timeout=2)

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
