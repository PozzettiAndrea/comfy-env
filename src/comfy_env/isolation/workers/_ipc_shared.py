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

# Timing constants
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
# Trimesh preparation for cross-process pickle
# =============================================================================

def _prepare_trimesh_for_pickle(mesh):
    """
    Prepare a trimesh object for cross-Python-version pickling.

    Trimesh attaches helper objects (ray tracer, proximity query) that may use
    native extensions like embreex. These cause import errors when unpickling
    on a system without those extensions. We strip them - they'll be recreated
    lazily when needed.

    Note: Do NOT strip _cache - trimesh needs it to function properly.
    """
    mesh = mesh.copy()
    for attr in ('ray', '_ray', 'permutate', 'nearest'):
        try:
            delattr(mesh, attr)
        except AttributeError:
            pass
    return mesh


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
# Generic shared memory serialization (_to_shm)
# =============================================================================

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
                          "shape": list(arr.shape), "dtype": str(arr.dtype), "size": size}
            else:
                from multiprocessing import shared_memory as shm
                block = shm.SharedMemory(create=True, size=arr.nbytes)
                np.ndarray(arr.shape, arr.dtype, buffer=block.buf)[:] = arr
                registry.append(block)
                result = {"__shm_np__": block.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        visited[obj_id] = result
        return result

    # trimesh.Trimesh -> pickle -> shared memory
    if t == 'Trimesh':
        import pickle
        obj = _prepare_trimesh_for_pickle(obj)
        mesh_bytes = pickle.dumps(obj)

        if _USE_MEMFD:
            fd, size = _memfd_write(mesh_bytes)
            registry.append(fd)
            result = {"__shm_trimesh__": True, "fd": fd, "pid": os.getpid(), "size": size}
        else:
            from multiprocessing import shared_memory as shm
            block = shm.SharedMemory(create=True, size=len(mesh_bytes))
            block.buf[:len(mesh_bytes)] = mesh_bytes
            registry.append(block)
            result = {"__shm_trimesh__": True, "name": block.name, "size": len(mesh_bytes)}

        visited[obj_id] = result
        return result

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

    # Fallback: pickle any remaining object to shared memory
    import pickle
    try:
        obj_bytes = pickle.dumps(obj)
        if _USE_MEMFD:
            fd, size = _memfd_write(obj_bytes)
            registry.append(fd)
            result = {"__shm_pickle__": True, "fd": fd, "pid": os.getpid(), "size": size}
        else:
            from multiprocessing import shared_memory as shm
            block = shm.SharedMemory(create=True, size=len(obj_bytes))
            block.buf[:len(obj_bytes)] = obj_bytes
            registry.append(block)
            result = {"__shm_pickle__": True, "name": block.name, "size": len(obj_bytes)}
        visited[obj_id] = result
        return result
    except Exception:
        return obj
