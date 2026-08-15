"""
Shared IPC utilities for comfy-env subprocess workers.

This module contains serialization functions shared between the parent process
and isolated worker subprocesses. It is intentionally standalone — no imports
from comfy_env — so it can be copied alongside the worker script and imported
directly in the isolated venv.

At worker startup, SubprocessWorker writes this file to the temp directory
next to persistent_worker.py so the worker can `import _ipc_shared`.
"""

import ctypes
import ctypes.util
import mmap as _mmap_mod
import os
import socket
import sys

# =============================================================================
# Constants
# =============================================================================

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB message size limit

# CUDA memory pool constants
CUDA_MEM_HANDLE_TYPE_POSIX_FD = 1
CUDA_MEM_ALLOCATION_TYPE_PINNED = 1
CUDA_MEM_LOCATION_TYPE_DEVICE = 1
CUDA_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 3
CUDA_MEMPOOL_ATTR_USED_MEM_CURRENT = 5

# Worker faulthandler dump file (basename under tempdir). The worker writes
# it; the parent's crash diagnostic reads it. MUST match on both sides --
# they drifted once (.log vs .txt) and crash dumps were silently never found.
WORKER_FAULTHANDLER_BASENAME = "comfy_worker_faulthandler.log"

# Timing constants (single source of truth -- the worker imports this
# module at its top; its own directory is sys.path[0], and the file is
# always copied alongside by SubprocessWorker)
TENSOR_KEEPER_TTL = 60.0        # seconds to hold shared tensors before GC
WATCHDOG_INTERVAL = 60          # seconds between watchdog thread dumps
VRAM_POLL_THRESHOLD = 200 * 1024 * 1024  # 200MB change triggers log
VRAM_POLL_INTERVAL = 0.1        # 100ms between VRAM polls
VRAM_LOG_COOLDOWN = 1.0         # 1 second between VRAM log messages
SOCKET_ACCEPT_TIMEOUT = 60      # seconds to wait for worker to connect
SOCKET_ID_LENGTH = 12           # hex chars in socket name uuid

# Cache limits
MAX_IPC_CACHE_SIZE = 256        # max entries in IPC handle forwarding caches


# =============================================================================
# Anonymous shared memory via memfd_create (Linux)
# =============================================================================

_USE_MEMFD = sys.platform == "linux"
_libc = None


def _memfd_write(data):
    """Create anonymous shared memory, write data. Returns (fd, size)."""
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    fd = _libc.memfd_create(b"comfy_ipc", 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create failed")
    size = len(data)
    os.ftruncate(fd, size)
    buf = _mmap_mod.mmap(fd, size, _mmap_mod.MAP_SHARED, _mmap_mod.PROT_WRITE)
    buf[:size] = data
    buf.close()
    return fd, size


def _memfd_read(pid, fd, size):
    """Read data from another process's memfd via procfs."""
    local_fd = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDONLY)
    try:
        buf = _mmap_mod.mmap(local_fd, size, _mmap_mod.MAP_SHARED, _mmap_mod.PROT_READ)
        data = bytes(buf[:size])
        buf.close()
        return data
    finally:
        os.close(local_fd)


# =============================================================================
# CUDA memory pool ctypes bindings
# =============================================================================

class _CudaMemPoolPtrExportData(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


class _CudaMemPoolProps(ctypes.Structure):
    _fields_ = [
        ("allocType", ctypes.c_int),
        ("handleTypes", ctypes.c_int),
        ("location_type", ctypes.c_int),
        ("location_id", ctypes.c_int),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("maxSize", ctypes.c_size_t),
        ("reserved", ctypes.c_ubyte * 56),
    ]


_cudart_lib = None


def _get_cudart():
    """Load and cache the CUDA runtime library."""
    global _cudart_lib
    if _cudart_lib is not None:
        return _cudart_lib
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            _cudart_lib = ctypes.CDLL(name)
            return _cudart_lib
        except OSError:
            continue
    lib_name = ctypes.util.find_library("cudart")
    if lib_name:
        _cudart_lib = ctypes.CDLL(lib_name)
        return _cudart_lib
    return None


def _cuda_check(err, name):
    """Raise RuntimeError if CUDA call returned non-zero."""
    if err != 0:
        raise RuntimeError(f"{name} returned {err}")


def _create_shareable_pool(device=0):
    """Create a CUDA memory pool with POSIX FD shareable handles."""
    cudart = _get_cudart()
    if not cudart:
        raise RuntimeError("libcudart not found")
    props = _CudaMemPoolProps()
    ctypes.memset(ctypes.addressof(props), 0, ctypes.sizeof(props))
    props.allocType = CUDA_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = CUDA_MEM_HANDLE_TYPE_POSIX_FD
    props.location_type = CUDA_MEM_LOCATION_TYPE_DEVICE
    props.location_id = device
    pool = ctypes.c_void_p()
    _cuda_check(cudart.cudaMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)),
                "cudaMemPoolCreate")
    return pool


def _export_pool_fd(pool):
    """Export a CUDA memory pool as a POSIX file descriptor."""
    cudart = _get_cudart()
    fd = ctypes.c_int()
    _cuda_check(cudart.cudaMemPoolExportToShareableHandle(
        ctypes.byref(fd), pool,
        ctypes.c_int(CUDA_MEM_HANDLE_TYPE_POSIX_FD), ctypes.c_uint(0)),
        "cudaMemPoolExportToShareableHandle")
    return fd.value


def _import_pool_from_fd(fd):
    """Import a CUDA memory pool from a POSIX file descriptor."""
    cudart = _get_cudart()
    pool = ctypes.c_void_p()
    fd_val = ctypes.c_int(fd)
    _cuda_check(cudart.cudaMemPoolImportFromShareableHandle(
        ctypes.byref(pool), ctypes.byref(fd_val),
        ctypes.c_int(CUDA_MEM_HANDLE_TYPE_POSIX_FD), ctypes.c_uint(0)),
        "cudaMemPoolImportFromShareableHandle")
    return pool


def _set_device_pool(device, pool):
    """Set the current CUDA memory pool for a device."""
    cudart = _get_cudart()
    _cuda_check(cudart.cudaDeviceSetMemPool(ctypes.c_int(device), pool),
                "cudaDeviceSetMemPool")


def _export_pointer(ptr):
    """Export a CUDA pool pointer to opaque bytes for cross-process transfer."""
    cudart = _get_cudart()
    export_data = _CudaMemPoolPtrExportData()
    _cuda_check(cudart.cudaMemPoolExportPointer(
        ctypes.byref(export_data), ctypes.c_void_p(ptr)),
        "cudaMemPoolExportPointer")
    return bytes(export_data)


def _import_pointer(pool, export_data_bytes):
    """Import a CUDA pool pointer from opaque bytes."""
    cudart = _get_cudart()
    export_data = _CudaMemPoolPtrExportData.from_buffer_copy(export_data_bytes)
    ptr = ctypes.c_void_p()
    _cuda_check(cudart.cudaMemPoolImportPointer(
        ctypes.byref(ptr), pool, ctypes.byref(export_data)),
        "cudaMemPoolImportPointer")
    return ptr.value


def _trim_pool(pool, min_bytes=0):
    """Trim a CUDA memory pool to release unused memory."""
    cudart = _get_cudart()
    _cuda_check(cudart.cudaMemPoolTrimTo(pool, ctypes.c_size_t(min_bytes)),
                "cudaMemPoolTrimTo")


def _get_pool_mem_stats(pool):
    """Query reserved and active bytes from a CUDA memory pool."""
    cudart = _get_cudart()
    if not cudart or not pool:
        return 0, 0
    reserved = ctypes.c_size_t(0)
    active = ctypes.c_size_t(0)
    cudart.cudaMemPoolGetAttribute(
        pool, ctypes.c_int(CUDA_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
        ctypes.byref(reserved))
    cudart.cudaMemPoolGetAttribute(
        pool, ctypes.c_int(CUDA_MEMPOOL_ATTR_USED_MEM_CURRENT),
        ctypes.byref(active))
    return reserved.value, active.value


# =============================================================================
# FD passing (SCM_RIGHTS) over Unix domain sockets
# =============================================================================

def _send_fd(sock, fd):
    """Send a file descriptor over a Unix domain socket via SCM_RIGHTS."""
    import array as _array
    sock.sendmsg([b'\x00'],
                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, _array.array('i', [fd]))])


def _recv_fd(sock, timeout=10.0):
    """Receive a file descriptor from a Unix domain socket via SCM_RIGHTS."""
    import array as _array
    sock.settimeout(timeout)
    try:
        msg, ancdata, flags, addr = sock.recvmsg(1, socket.CMSG_LEN(4))
        for level, type_, data in ancdata:
            if level == socket.SOL_SOCKET and type_ == socket.SCM_RIGHTS:
                fds = _array.array('i')
                fds.frombytes(data[:fds.itemsize])
                return fds[0]
        raise RuntimeError("No FD in ancillary data")
    finally:
        sock.settimeout(None)


# =============================================================================
# Pool pointer wrapper
# =============================================================================

class _PoolPtr:
    """Wrap imported CUDA pointer for __cuda_array_interface__."""
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            'shape': (nbytes,), 'typestr': '|u1',
            'data': (ptr, False), 'version': 3,
        }


# =============================================================================
# Shared memory registry cleanup
# =============================================================================

def _cleanup_shm(registry):
    """Close all shared memory in registry (memfd fds or SharedMemory blocks)."""
    for item in registry:
        try:
            if isinstance(item, int):
                os.close(item)  # memfd fd
            else:
                item.close()
                item.unlink()
        except OSError:
            pass
    registry.clear()


# =============================================================================
# IPC cache management
# =============================================================================

def _evict_cache_if_needed(cache_dict):
    """Evict oldest half of cache if it exceeds MAX_IPC_CACHE_SIZE."""
    if len(cache_dict) > MAX_IPC_CACHE_SIZE:
        to_remove = list(cache_dict.keys())[:len(cache_dict) // 2]
        for k in to_remove:
            del cache_dict[k]


# =============================================================================
# Serializer registry (custom data types)
# =============================================================================
#
# Node packs can teach the transport about their own types. Both sides of the
# process boundary carry THIS module (the parent imports it from comfy_env;
# the worker imports the copy placed next to it), so a serializer module
# imported on both sides registers identical rules by construction.
#
# A pack declares its wire types in comfy-env-root.toml (ADR-0015):
#
#     [types]
#     TRIMESH = "custom"      # code in <pack>/serialization.py
#     SKELETON = "builtin"    # automatic transport, listed for humans/tests
#
# <pack>/serialization.py is loaded by FILE PATH under a mangled per-pack
# module name on BOTH sides (parent + worker), must keep its top-level
# imports to stdlib/numpy/comfy_env (heavy deps inside the functions), and
# looks like:
#
#     try:  # parent process
#         from comfy_env.isolation.workers import _ipc_shared as ipc
#     except ImportError:  # worker process (module copied next to the worker)
#         import _ipc_shared as ipc
#
#     def _ser(obj, recurse):
#         return {"verts": recurse(obj.vertices), "id": obj.id}
#
#     def _deser(payload, recurse):
#         from mymeshlib import MyMesh
#         return MyMesh(recurse(payload["verts"]), payload["id"])
#
#     try:  # register deserialize only where the library exists; a side
#         import mymeshlib  # noqa: F401  -- without it holds materialized
#         _DESER = _deser   #                OpaquePayload receipts instead
#     except ImportError:
#         _DESER = None
#
#     ipc.register_serializer("MyMesh", _ser, _DESER, tag="mymeshlib.MyMesh")
#
# Tag convention (ADR-0015): shared library types tag by TYPE IDENTITY
# ("mymeshlib.MyMesh") so independent packs interoperate; pack-private
# types tag with a pack prefix.
#
# `recurse` routes nested values through the normal transport, so tensors
# inside a custom payload keep their zero-copy paths.
#
# If the RECEIVING side has no deserializer for a tag (typical for the parent,
# whose env deliberately lacks the pack's classes), the value arrives as a
# MATERIALIZED OpaquePayload -- receiver-owned objects that re-serialize to
# fresh frames. Custom objects therefore survive worker -> parent -> worker
# round trips, worker restarts included, without the parent understanding
# them.


class SerializerRegistry:
    """Per-process registry: type name -> (tag, serialize); tag -> deserialize."""

    def __init__(self):
        self._by_type = {}    # type __name__ -> (tag, serialize_fn)
        self._by_tag = {}     # tag -> deserialize_fn

    def register(self, type_name, serialize, deserialize=None, tag=None):
        tag = tag or type_name
        self._by_type[type_name] = (tag, serialize)
        if deserialize is not None:
            self._by_tag[tag] = deserialize

    def lookup_serializer(self, obj):
        """Match by exact class name, then by MRO (base class names)."""
        entry = self._by_type.get(type(obj).__name__)
        if entry is not None:
            return entry
        for base in type(obj).__mro__[1:-1]:  # skip cls itself and object
            entry = self._by_type.get(base.__name__)
            if entry is not None:
                return entry
        return None

    def lookup_deserializer(self, tag):
        return self._by_tag.get(tag)


REGISTRY = SerializerRegistry()


def register_serializer(type_name, serialize, deserialize=None, tag=None):
    """Register a custom type with the transport (see module comment above).

    Args:
        type_name: class __name__ to match (base-class names match via MRO).
        serialize: callable(obj, recurse) -> JSON-safe payload.
        deserialize: callable(payload, recurse) -> obj. Optional on sides
            that only forward the type (they get OpaquePayload instead).
        tag: wire tag; defaults to type_name.
    """
    REGISTRY.register(type_name, serialize, deserialize, tag)


class OpaquePayload:
    """A custom-tagged value this process cannot reconstruct.

    The payload is MATERIALIZED on receipt (see deserialize_custom): every
    frame is deserialized into objects this process owns, so the held value
    no longer references the sender's shared memory. Re-serializing walks
    those objects and emits fresh frames backed by this process's memory.
    """

    def __init__(self, tag, payload):
        self.tag = tag
        self.payload = payload

    def __repr__(self):
        return f"OpaquePayload(tag={self.tag!r})"


class OpaquePickle:
    """Pickled bytes this process cannot unpickle (class's module not
    installed here -- e.g. a bare ComfyUI host env that installs only
    comfy-env). Owns the bytes, so the value survives sender restarts;
    re-serializing writes a fresh __shm_pickle__ block for the next hop.
    """

    def __init__(self, data):
        self.data = bytes(data)

    def __repr__(self):
        return f"OpaquePickle({len(self.data)} bytes)"


def loads_or_opaque(obj_bytes):
    """pickle.loads that degrades to OpaquePickle when this side lacks the
    class. AttributeError covers version-skewed modules missing the name;
    ModuleNotFoundError subclasses ImportError. Either way the value stays
    forwardable instead of killing the request."""
    import pickle
    try:
        return pickle.loads(obj_bytes)
    except (ImportError, AttributeError):
        return OpaquePickle(obj_bytes)


def _own_tree(value):
    """Ensure no node of a materialized payload borrows the sender's
    buffers: numpy arrays that are views (the worker-side _from_shm maps
    arrays directly into the sender's blocks) are copied. Tensors from
    TensorRef own their storage mapping and survive the sender."""
    if type(value).__name__ == "ndarray" and getattr(value, "base", None) is not None:
        return value.copy()
    if isinstance(value, dict):
        return {k: _own_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_own_tree(v) for v in value]
    return value


def deserialize_custom(obj, recurse):
    """Handle a {"__shm_custom__": tag, "payload": ...} frame.

    Called by both sides' _from_shm. With a registered deserializer the
    payload is passed RAW -- its own `recurse` calls decide which nested
    parts to reconstruct (so it can skip or transform sections).

    Without one (this side lacks the pack's libs), every frame in the
    payload is materialized into objects THIS process owns before wrapping
    in OpaquePayload: a held receipt must not reference the sender's shared
    memory, which the sender frees on ack/TTL/restart. Holding raw frames
    was the root cause of FileNotFoundError on dead shm names when a
    lib-less host forwarded a mesh after its worker restarted.
    """
    tag = obj["__shm_custom__"]
    deser = REGISTRY.lookup_deserializer(tag)
    if deser is None:
        return OpaquePayload(tag, _own_tree(recurse(obj["payload"])))
    return deser(obj["payload"], recurse)


def _cuda_device_uuid():
    """UUID string of torch's CURRENT CUDA device, or None if no CUDA.

    Identity-not-index: `device_idx` in the zero-copy frames means nothing
    across the boundary if the two sides enumerate GPUs differently (a pack
    env with its own CUDA_VISIBLE_DEVICES). The parent compares this against
    its own current-device UUID and demotes zero-copy on mismatch rather
    than importing a handle onto the wrong physical GPU. Best-effort: any
    failure returns None (treated as "cannot verify", not "mismatch").
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return str(torch.cuda.get_device_properties(
            torch.cuda.current_device()).uuid)
    except Exception:
        return None


def registration_count():
    """Number of registered type serializers (validation hook, ADR-0015)."""
    return len(REGISTRY._by_type)


def load_serializer_files(spec, log=None):
    """Load comma-separated serializer FILES (COMFY_ENV_SERIALIZER_FILES).

    Each file executes under a mangled per-pack module name, so every pack
    can ship a plain `serialization.py` without colliding in sys.modules
    (plain module-name loading did collide: first import won and the second
    pack's serializers silently never registered -- ADR-0015).

    Serializer files must be self-contained: top-level imports limited to
    stdlib/numpy/comfy_env, heavy deps imported inside the functions.
    Import errors are reported, not raised, here -- the parent applies the
    loud [types] validation separately.
    """
    import importlib.util
    import os as _os
    import re as _re
    import sys as _sys
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        pack = _os.path.basename(_os.path.dirname(raw)) or "pack"
        stem = _os.path.splitext(_os.path.basename(raw))[0]
        mod_name = "_comfy_env_ser__" + _re.sub(r"\W+", "_", f"{pack}_{stem}")
        if mod_name in _sys.modules:
            continue  # already loaded (same pack, several envs)
        try:
            file_spec = importlib.util.spec_from_file_location(mod_name, raw)
            module = importlib.util.module_from_spec(file_spec)
            _sys.modules[mod_name] = module
            file_spec.loader.exec_module(module)
            if log:
                log(f"[comfy-env] serializers loaded from {raw}")
        except Exception as e:
            _sys.modules.pop(mod_name, None)
            if log:
                log(f"[comfy-env] serializer file {raw} not importable "
                    f"here ({e}); its types are held opaquely")


# =============================================================================
# Generic shared memory serialization (_to_shm)
# =============================================================================

def _encode_np_dtype(dt):
    """JSON round-trippable numpy dtype.

    Simple dtypes -> their type string (e.g. '<f4'): small, readable, and
    what legacy frames already carried. Structured / subarray dtypes -> a
    pickled blob, because their field descr does NOT survive a JSON round-trip
    -- JSON collapses the tuple-vs-list distinction the descr format depends
    on, and numpy's descr_to_dtype then misreads it (the failure even varies
    by numpy version: a PLY record dtype crashed on np.dtype() one way and on
    descr_to_dtype another). A dtype pickles safely -- numpy only, no arbitrary
    code -- and it is a few bytes of metadata, consistent with the transport's
    existing pickle rung."""
    import base64
    import pickle
    if dt.fields is None and dt.subdtype is None:
        return dt.str
    return {"__pickle_dtype__": base64.b64encode(pickle.dumps(dt)).decode("ascii")}


def _decode_np_dtype(enc):
    """Inverse of _encode_np_dtype. A dict is a pickled structured dtype; a
    string is a simple dtype (also accepts a legacy str(dtype) for simple
    dtypes, which np.dtype parses)."""
    import numpy as np
    if isinstance(enc, dict):
        import base64
        import pickle
        return pickle.loads(base64.b64decode(enc["__pickle_dtype__"]))
    return np.dtype(enc)


def _to_shm_generic(obj, registry, visited, *, tensor_serializer, node_output_serializer=None):
    """
    Serialize object to shared memory. Returns JSON-safe metadata.

    This is the shared implementation used by both parent and worker. The
    tensor_serializer callback handles the Tensor branch (which differs
    between parent and worker due to different pool/IPC strategies).

    Args:
        obj: Object to serialize
        registry: List to track SharedMemory objects for cleanup
        visited: Dict tracking already-serialized objects (cycle detection)
        tensor_serializer: Callable(tensor, registry, visited) -> dict metadata
        node_output_serializer: Optional callable for NodeOutput objects
    """
    from pathlib import PurePath

    obj_id = id(obj)
    if obj_id in visited:
        return visited[obj_id]

    t = type(obj).__name__

    def _recurse(v):
        return _to_shm_generic(v, registry, visited,
                               tensor_serializer=tensor_serializer,
                               node_output_serializer=node_output_serializer)

    # Opaque re-emit: the payload was materialized into owned objects on
    # receipt, so serialize those afresh -- the next hop gets frames backed
    # by THIS process's memory, not the (possibly dead) original sender's.
    if isinstance(obj, OpaquePayload):
        return {"__shm_custom__": obj.tag, "payload": _recurse(obj.payload)}

    # Held pickled bytes (class not importable here) -> fresh pickle block.
    if isinstance(obj, OpaquePickle):
        result = _pickle_frame(obj.data, registry)
        visited[obj_id] = result
        return result

    # Registered custom types take precedence over the built-in branches so
    # a pack may deliberately override handling of a named type.
    entry = REGISTRY.lookup_serializer(obj)
    if entry is not None:
        tag, serialize = entry
        result = {"__shm_custom__": tag, "payload": serialize(obj, _recurse)}
        visited[obj_id] = result
        return result

    # torch.Tensor -> delegate to caller-provided strategy
    if t == 'Tensor':
        result = tensor_serializer(obj, registry, visited)
        visited[obj_id] = result
        return result

    # numpy array -> PyTorch native shared memory (zero-copy), fallback to shm copy
    if t == 'ndarray':
        import numpy as np
        arr = np.ascontiguousarray(obj)
        try:
            import torch
            tensor = torch.from_numpy(arr)
            result = tensor_serializer(tensor, registry, visited)
            result["__was_numpy__"] = True
            result["numpy_dtype"] = str(arr.dtype)
        except Exception:
            arr_bytes = arr.tobytes()
            if _USE_MEMFD:
                fd, size = _memfd_write(arr_bytes)
                registry.append(fd)
                result = {"__shm_np__": True, "fd": fd, "pid": os.getpid(),
                          "shape": list(arr.shape), "dtype": _encode_np_dtype(arr.dtype), "size": size}
            else:
                from multiprocessing import shared_memory as shm
                block = shm.SharedMemory(create=True, size=arr.nbytes)
                np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
                registry.append(block)
                result = {"__shm_np__": block.name, "shape": list(arr.shape),
                          "dtype": _encode_np_dtype(arr.dtype)}
        visited[obj_id] = result
        return result

    # NOTE: trimesh has no builtin branch (removed 2026-08, no backcompat):
    # packs register their own mesh serializers via the registry (ADR-0014;
    # ComfyUI-GeometryPack's geometrypack_wire_types is the reference).
    # Unregistered meshes fall to the generic pickle rung below.

    # SparseTensor -> decompose to coords + feats CPU tensors
    if t == 'SparseTensor':
        feats_cpu = obj.feats.detach().cpu().contiguous()
        coords_cpu = obj.coords.detach().cpu().contiguous()
        result = {
            "__shm_sparse_tensor__": True,
            "coords": _to_shm_generic(coords_cpu, registry, visited,
                                       tensor_serializer=tensor_serializer,
                                       node_output_serializer=node_output_serializer),
            "feats": _to_shm_generic(feats_cpu, registry, visited,
                                      tensor_serializer=tensor_serializer,
                                      node_output_serializer=node_output_serializer),
            "feats_dtype": str(feats_cpu.dtype),
        }
        visited[obj_id] = result
        return result

    # V3 NodeOutput -> delegate to caller if provided
    if t == 'NodeOutput' and node_output_serializer is not None:
        result = node_output_serializer(obj, registry, visited)
        visited[obj_id] = result
        return result

    # Path -> string
    if isinstance(obj, PurePath):
        return str(obj)

    # dict
    if isinstance(obj, dict):
        result = {k: _to_shm_generic(v, registry, visited,
                                       tensor_serializer=tensor_serializer,
                                       node_output_serializer=node_output_serializer)
                  for k, v in obj.items()}
        visited[obj_id] = result
        return result

    # list/tuple
    if isinstance(obj, (list, tuple)):
        result = [_to_shm_generic(v, registry, visited,
                                   tensor_serializer=tensor_serializer,
                                   node_output_serializer=node_output_serializer)
                  for v in obj]
        visited[obj_id] = result
        return result

    # Convert numpy scalars to Python primitives for JSON serialization
    try:
        import numpy as np
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
    except ImportError:
        pass

    # primitives pass through (str, int, float, bool, None)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: pickle any remaining object to shared memory. A pickle
    # failure is a LOUD error naming the type and cause -- the old
    # `return obj` fallback leaked the raw object into the JSON message
    # and crashed two layers away with "not JSON serializable"
    # (demonstrated: pickling a textured Trimesh imports PIL at
    # serialize time; without PIL the bytes are never produced).
    import pickle
    try:
        obj_bytes = pickle.dumps(obj)
    except Exception as e:
        raise TypeError(
            f"comfy-env transport cannot serialize "
            f"{type(obj).__name__!r}: pickle failed ({e}). Register a "
            f"serializer for this type in your pack's serialization.py "
            f"([types] socket = \"custom\", ADR-0015), or install the "
            f"missing dependency in this environment."
        ) from e
    result = _pickle_frame(obj_bytes, registry)
    visited[obj_id] = result
    return result


def _pickle_frame(obj_bytes, registry):
    """Write pickled bytes to a shm block/memfd; return the wire frame."""
    if _USE_MEMFD:
        fd, size = _memfd_write(obj_bytes)
        registry.append(fd)
        return {"__shm_pickle__": True, "fd": fd, "pid": os.getpid(), "size": size}
    from multiprocessing import shared_memory as shm
    block = shm.SharedMemory(create=True, size=len(obj_bytes))
    block.buf[:len(obj_bytes)] = obj_bytes
    registry.append(block)
    return {"__shm_pickle__": True, "name": block.name, "size": len(obj_bytes)}
