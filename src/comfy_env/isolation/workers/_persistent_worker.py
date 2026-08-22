
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

# _ipc_shared.py is always copied next to this script by SubprocessWorker,
# and the script's own directory is sys.path[0] -- so shared constants are
# importable even this early. One source of truth; no hand-synced literals.
import _ipc_shared

# Also dump to a file so we can see segfaults even if stderr is lost
import tempfile as _fh_tempfile
_faulthandler_log = os.path.join(
    _fh_tempfile.gettempdir(), _ipc_shared.WORKER_FAULTHANDLER_BASENAME)
try:
    _fh_file = open(_faulthandler_log, "a")
    faulthandler.enable(file=_fh_file, all_threads=True)
except Exception:
    pass

# Debug logging -- granular categories (env vars propagate from parent)
def _dbg_on(var):
    return os.environ.get(var, "").lower() in ("1", "true", "yes")
_DBG_ALL = _dbg_on("COMFY_ENV_DEBUG")
_DBG_SERIALIZE = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_SERIALIZE")
_DBG_IPC = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_IPC")
_DBG_WORKER = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_WORKER")
_DBG_MODELS = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_MODELS")
_DBG_STACKTRACE = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_STACKTRACE")
_DBG_VRAM = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_VRAM")
_DBG_WATCHDOG = _DBG_ALL or _dbg_on("COMFY_ENV_DEBUG_WATCHDOG")
_DEBUG = any((_DBG_SERIALIZE, _DBG_IPC, _DBG_WORKER, _DBG_MODELS))

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
            f.write(f"\n=== WATCHDOG TICK {tick} ({time.strftime('%H:%M:%S')}) ===\n")
            f.write(dump)
            f.write("=== END ===\n")
            f.flush()

        # Also print (only if watchdog debug enabled)
        if _DBG_WATCHDOG:
            print(f"\n=== WATCHDOG TICK {tick} ===", flush=True)
            print(dump, flush=True)
            print("=== END ===\n", flush=True)

# Start watchdog when its own flag or any debug is on (always logs to file, only prints if _DBG_WATCHDOG)
if _DBG_WATCHDOG or _DEBUG:
    _watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    _watchdog_thread.start()
if _DBG_WATCHDOG:
    print(f"[worker] Watchdog started, logging to: {_watchdog_log}", flush=True)

# File-based logging for debugging (persists even if stdout/stderr are swallowed)
import tempfile
_worker_log_file = os.path.join(tempfile.gettempdir(), "comfy_worker_debug.log")
def wlog(msg):
    """Log to file only - stdout causes pipe buffer deadlock after many requests."""
    try:
        with open(_worker_log_file, "a", encoding="utf-8") as f:
            import time
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
    # NOTE: Don't print to stdout here! After 50+ requests the pipe buffer
    # fills up and causes deadlock (parent blocked on recv, worker blocked on print)

wlog(f"[worker] === Worker starting, log file: {_worker_log_file} ===")

# VRAM poller: background thread that detects GPU memory changes
_vram_poll_transport = None  # set in main() after transport is available
if _DBG_VRAM:
    def _vram_poller():
        import time as _vt
        threshold = 200 * 1024 * 1024  # 200MB -- ignore attention transients
        min_interval = 1.0              # max 1 log/sec
        last_alloc = 0
        last_log_time = 0.0
        peak_alloc = 0
        _torch = None
        while True:
            _vt.sleep(0.1)
            try:
                if _torch is None:
                    import torch as _torch
                    if not _torch.cuda.is_available():
                        return
                alloc = _torch.cuda.memory_allocated()
                if alloc > peak_alloc:
                    peak_alloc = alloc
                delta = alloc - last_alloc
                now = _vt.time()
                if abs(delta) >= threshold and (now - last_log_time) >= min_interval:
                    alloc_mb = alloc // (1024 * 1024)
                    reserved_mb = _torch.cuda.memory_reserved() // (1024 * 1024)
                    sign = "+" if delta > 0 else ""
                    delta_mb = delta // (1024 * 1024)
                    peak_mb = peak_alloc // (1024 * 1024)
                    msg = f"[VRAM] {sign}{delta_mb}MB (now {alloc_mb}MB) reserved={reserved_mb}MB peak={peak_mb}MB"
                    wlog(msg)
                    if _vram_poll_transport is not None:
                        try:
                            _vram_poll_transport.send({"type": "log", "message": msg})
                        except Exception:
                            pass
                    last_alloc = alloc
                    last_log_time = now
            except ImportError:
                pass  # torch not yet imported, retry next tick
            except Exception:
                pass
    _vram_thread = threading.Thread(target=_vram_poller, daemon=True)
    _vram_thread.start()
    wlog("[worker] VRAM poller started (200MB threshold, 100ms poll, 1s cooldown)")

# =============================================================================
# Shared Memory Serialization
# =============================================================================

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

# Import torch BEFORE numpy on Windows. conda-forge's numpy is MKL-linked, and
# loading numpy first pulls in libiomp5md.dll from <env>/Library/bin -- once that
# OMP runtime is in the process, torch's bundled libiomp5md (in torch/lib/) can't
# load alongside it and fbgemm.dll's delay-loaded deps fail with WinError 127.
# Order matters: torch first ensures torch/lib's DLLs win the address-space race.
# Use default sharing strategy (file_descriptor on Linux).
# Do NOT force file_system -- its torch_shm_manager prematurely unlinks files in torch 2.8.
try:
    import torch
    import torch.multiprocessing as mp
    wlog(f"[worker] PyTorch sharing strategy: {mp.get_sharing_strategy()}")
except Exception as e:
    wlog(f"[worker] PyTorch not available: {e}")

from multiprocessing import shared_memory as shm
import mmap as _mmap_mod  # noqa: F401 -- kept for worker-local mmap users
import numpy as np

# The shared IPC module is copied next to this script by SubprocessWorker
# (subprocess.py) -- import it instead of duplicating its contents. It is
# stdlib-only at module scope, so this import is safe w.r.t. the torch/numpy
# DLL-ordering constraints above.
import _ipc_shared
# Alias the local copy under its package name: a serializer module that does
# `from comfy_env.isolation.workers import _ipc_shared` (the parent-side
# spelling) inside a worker whose env happens to have comfy_env installed
# must land on THIS instance -- two module instances would mean two
# registries and silently unregistered types.
sys.modules.setdefault("comfy_env.isolation.workers._ipc_shared", _ipc_shared)
# Same defence for the SHORT spelling `from comfy_env import register_serializer`,
# which is the documented one from comfy-env 1.0. Only when comfy_env is not
# already importable here -- a real package must win, and we must never shadow
# it with a stub, because node code may import comfy_env for other reasons.
if "comfy_env" not in sys.modules:
    try:
        import comfy_env as _real_ce  # noqa: F401 -- probe only
    except ImportError:
        import types as _types
        _ce_stub = _types.ModuleType("comfy_env")
        _ce_stub.register_serializer = _ipc_shared.register_serializer
        _ce_stub.__all__ = ["register_serializer"]
        sys.modules["comfy_env"] = _ce_stub
from _ipc_shared import (  # noqa: F401 -- re-exported names used below
    _USE_MEMFD,
    _memfd_write,
    _memfd_read,
    _create_shareable_pool,
    _export_pool_fd,
    _import_pool_from_fd,
    _set_device_pool,
    _export_pointer,
    _import_pointer,
    _trim_pool,
    _get_pool_mem_stats,
    _send_fd,
    _recv_fd,
    _PoolPtr,
)

# Release CPU affinity back to all cores for actual GPU work
if _affinity_pinned:
    try:
        os.sched_setaffinity(0, set(range(os.cpu_count() or 1)))
    except OSError:
        pass


# Call id of the response currently being serialized. Set by the main loop
# before _to_shm(result, ...) so keeper entries can be released the moment
# the parent acks that call with {"type": "consumed"} instead of waiting
# out the TTL. The TTL survives only as the fallback for a parent that
# died before acking.
_serializing_call_id = None


# Tensor keeper - holds tensor references to prevent GC before parent reads shared memory
class TensorKeeper:
    """Keep tensors alive until the parent acks the call (release()), with
    a TTL sweep as the crash fallback. A timer alone is a guess: parent
    sleeps/suspends, nested calls mid-read, or slow multi-output
    serialization can all outlive any fixed window.
    """
    def __init__(self, retention_seconds=_ipc_shared.TENSOR_KEEPER_TTL):
        self.retention_seconds = retention_seconds
        self._keeper = collections.deque()  # (time, call_id, tensor)
        self._lock = threading.Lock()

    def keep(self, t):
        now = time.time()
        with self._lock:
            self._keeper.append((now, _serializing_call_id, t))
            # TTL sweep (fallback only -- release() is the real path)
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                self._keeper.popleft()

    def release(self, call_id):
        """Drop every entry kept for `call_id` -- the parent confirmed it
        has read (or now owns) all frames of that call's response."""
        if call_id is None:
            return
        with self._lock:
            kept = [e for e in self._keeper if e[1] != call_id]
            self._keeper.clear()
            self._keeper.extend(kept)

    def count(self):
        with self._lock:
            return len(self._keeper)

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
    # An IPC handle is only useful if the other end can import it, and the two
    # ends often run different allocators -- ComfyUI adds
    # backend:cudaMallocAsync, which cannot import, while this worker's env may
    # allow export. Exporting anyway means the failure lands on the parent's
    # rebuild, after the node has finished its work. The parent tells us here.
    if os.environ.get("COMFY_ENV_PARENT_CUDA_IPC") == "0":
        _cuda_ipc_supported = False
        wlog("[worker] CUDA IPC disabled: parent cannot import handles "
             "(likely cudaMallocAsync); falling back to shared-memory copy")
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
        wlog("[worker] CUDA IPC supported (legacy)")
    except Exception as e:
        _cuda_ipc_supported = False
        wlog(f"[worker] CUDA IPC not supported: {e}")
    return _cuda_ipc_supported

# IPC handle forwarding cache (worker-side, for passthrough tensors)
_cuda_ipc_metadata_cache = {}
_cuda_ipc_cache_tensors = {}

def _serialize_cuda_ipc(t):
    import torch.multiprocessing.reductions as reductions
    # Check IPC handle cache -- forward original handle if available
    try:
        storage_id = id(t.untyped_storage())
        cached = _cuda_ipc_metadata_cache.get(storage_id)
        if cached is not None:
            if (list(t.size()) == cached["tensor_size"]
                    and list(t.stride()) == cached["tensor_stride"]
                    and t.storage_offset() == cached.get("tensor_offset", 0)):
                wlog("[worker] CUDA IPC cache hit -- forwarding handle (no clone)")
                return cached
            wlog("[worker] CUDA IPC cache hit (view) -- forwarding with adjusted shape")
            return {**cached, "tensor_size": list(t.size()),
                    "tensor_stride": list(t.stride()),
                    "tensor_offset": t.storage_offset()}
    except Exception:
        pass
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
    # Cache IPC metadata for handle forwarding (zero-copy passthrough)
    try:
        storage_id = id(tensor.untyped_storage())
        _cuda_ipc_metadata_cache[storage_id] = data
        _cuda_ipc_cache_tensors[storage_id] = tensor
    except Exception:
        pass
    return tensor


# =============================================================================
# Pool IPC - shareable CUDA memory pool (worker side)
# =============================================================================

_POOL_IPC_ENABLED = os.environ.get("COMFY_ENV_POOL_IPC", "").lower() in ("1", "true", "yes")
_pool_ipc_ok = False
_our_pool = None
# Parent's shareable pool (parent->worker zero-copy); imported in main()'s
# handshake, read by the module-level _from_shm().
_parent_pool = None
_pool_ipc_metadata_cache = {}
_pool_ipc_cache_tensors = {}

# Pool ctypes primitives and _PoolPtr come from _ipc_shared (imported at the
# top of this file). The duplicated definitions that lived here drifted from
# the shared copies before being deleted -- do not reintroduce them.

def _deserialize_pool_ipc(data, source_pool):
    import torch
    export_data_bytes = _b64.b64decode(data["export_data"])
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
    return tensor

def _serialize_pool_ipc(t):
    """Serialize CUDA tensor via pool pointer export (zero-copy)."""
    import torch
    # Check forwarding cache
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
        "export_data": _b64.b64encode(export_data).decode("ascii"),
        "storage_size": storage.size(),
        "dtype": str(t.dtype),
        "tensor_size": list(t.size()),
        "tensor_stride": list(t.stride()),
        "tensor_offset": t.storage_offset(),
        "device_idx": t.device.index or 0,
        "requires_grad": t.requires_grad,
    }
    # Cache for future forwarding
    try:
        _pool_ipc_metadata_cache[id(t.untyped_storage())] = result
        _pool_ipc_cache_tensors[id(t.untyped_storage())] = t
    except Exception:
        pass
    return result




def _serialize_tensor_native(t, registry):
    """Serialize tensor using file_descriptor shared memory (zero-copy to parent)."""
    import torch
    import torch.multiprocessing.reductions as reductions

    # Keep tensor alive until parent reads it
    _tensor_keeper.keep(t)

    if not t.is_shared():
        t.share_memory_()

    storage = t.untyped_storage()
    sfunc, sargs = reductions.reduce_storage(storage)

    if sfunc.__name__ == "rebuild_storage_fd":
        dupfd = sargs[1]
        fd = dupfd.detach()
        _worker_fd_registry.append(fd)
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


def _worker_tensor_serializer(t, registry, visited):
    """Worker-side Tensor strategy: Pool IPC -> legacy CUDA IPC -> CPU shm."""
    import torch  # noqa: F401 -- ensures torch is importable before use
    if t.is_cuda:
        # Pool IPC: zero-copy via shareable pool (cudaMallocAsync-safe)
        if _pool_ipc_ok and _our_pool is not None:
            try:
                return _serialize_pool_ipc(t)
            except Exception as e:
                wlog(f"[worker] Pool IPC serialize failed: {e}, falling back")
        # Legacy CUDA IPC (only works without cudaMallocAsync)
        if _probe_cuda_ipc():
            return _serialize_cuda_ipc(t)
    tensor = t.detach().cpu().contiguous()
    return _serialize_tensor_native(tensor, registry)


def _worker_node_output_serializer(obj, registry, visited):
    """V3 NodeOutput -> tagged dict for IPC serialization."""
    ui_val = obj.ui
    if hasattr(ui_val, 'as_dict'):
        ui_val = ui_val.as_dict()
    return {
        "__node_output__": True,
        "args": _to_shm(list(obj.args), registry, visited),
        "ui": _to_shm(ui_val, registry, visited) if ui_val is not None else None,
        "expand": _to_shm(obj.expand, registry, visited) if obj.expand is not None else None,
        "block_execution": obj.block_execution,
    }


def _to_shm(obj, registry, visited=None):
    """Serialize to shared memory. Returns JSON-safe metadata.

    Thin wrapper over the SHARED walker in _ipc_shared -- only the tensor
    strategy and NodeOutput handling are worker-specific. This replaced a
    full local reimplementation of the walker that had already drifted from
    the shared copy.
    """
    if visited is None:
        visited = {}
    return _ipc_shared._to_shm_generic(
        obj, registry, visited,
        tensor_serializer=_worker_tensor_serializer,
        node_output_serializer=_worker_node_output_serializer,
    )


def _deserialize_tensor_native(data):
    """Deserialize tensor from parent's shared memory.

    Supports two strategies:
    - file_descriptor: opens parent's fd via /proc/<pid>/fd/<N>, mmaps it,
      wraps with torch.frombuffer. No torch storage manager involvement.
    - file_system: legacy fallback using rebuild_storage_filename.
    """
    import torch

    dtype_str = data["dtype"]
    dtype = getattr(torch, dtype_str.split(".")[-1])
    strategy = data.get("strategy", "file_system")

    if strategy == "file_descriptor":
        import mmap as _mmap
        parent_pid = data["parent_pid"]
        parent_fd = data["fd"]
        storage_size = data["storage_size"]

        # Open the parent's fd via /proc -- zero-copy mmap
        fd = os.open(f"/proc/{parent_pid}/fd/{parent_fd}", os.O_RDWR)
        buf = _mmap.mmap(fd, storage_size, _mmap.MAP_SHARED, _mmap.PROT_READ | _mmap.PROT_WRITE)
        os.close(fd)  # mmap holds its own reference

        # Wrap the mmap as a tensor -- zero-copy
        flat = torch.frombuffer(buf, dtype=dtype)
        tensor = flat.view(tuple(data["tensor_size"]))
        # Keep mmap alive as long as tensor is in use
        tensor._shm_buf = buf
        return tensor
    else:
        # Legacy file_system fallback
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

        # Prevent worker from unlinking parent-owned shm file on GC
        rebuilt_storage._shared_incref()
        _input_torch_storages.append(rebuilt_storage)

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


def _from_shm(obj, _depth=0, _key="root"):
    """Reconstruct from shared memory metadata. Does NOT unlink - caller handles that."""
    if _DBG_SERIALIZE and isinstance(obj, dict) and any(k in obj for k in ("__type__", "__shm_np__", "tensor_size")):
        print(f"[comfy-env] _from_shm got dict with keys: {list(obj.keys())[:5]}", file=sys.stderr, flush=True)
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [_from_shm(v, _depth+1, f"{_key}[{i}]") for i, v in enumerate(obj)]
        return obj

    # Registered custom type (or OpaquePayload when unknown on this side)
    if "__shm_custom__" in obj:
        return _ipc_shared.deserialize_custom(
            obj, lambda v: _from_shm(v, _depth + 1, f"{_key}.custom"))

    # PoolIPC -> zero-copy CUDA tensor via shareable pool (parent -> worker)
    if obj.get("__type__") == "PoolIPC":
        wlog(f"[_from_shm] {_key}: PoolIPC tensor_size={obj.get('tensor_size')}")
        if _parent_pool is not None:
            return _deserialize_pool_ipc(obj, _parent_pool)
        wlog(f"[_from_shm] {_key}: PoolIPC but no parent pool, falling back to error")
        raise RuntimeError("PoolIPC received but no parent pool handle available")

    # CudaIPC -> zero-copy CUDA tensor deserialization
    if obj.get("__type__") == "CudaIPC":
        wlog(f"[_from_shm] {_key}: CudaIPC tensor_size={obj.get('tensor_size')}")
        return _deserialize_cuda_ipc(obj)

    # TensorRef -> use PyTorch's native deserialization (both directions)
    if obj.get("__type__") == "TensorRef":
        wlog(f"[_from_shm] {_key}: TensorRef tensor_size={obj.get('tensor_size')}")
        if _DBG_SERIALIZE:
            print(f"[comfy-env] DESERIALIZE TensorRef: tensor_size={obj.get('tensor_size')}", file=sys.stderr, flush=True)
        tensor = _deserialize_tensor_native(obj)
        wlog(f"[_from_shm] {_key}: TensorRef deserialized shape={tensor.shape}")
        if _DBG_SERIALIZE:
            print(f"[comfy-env] DESERIALIZED tensor shape: {tensor.shape}", file=sys.stderr, flush=True)
        # Convert back to numpy if it was originally numpy
        if obj.get("__was_numpy__"):
            return tensor.numpy()
        return tensor

    # __shm_np__ -> numpy array via shared memory (fallback when torch unavailable)
    if "__shm_np__" in obj:
        shape = tuple(obj["shape"])
        dtype = _ipc_shared._decode_np_dtype(obj["dtype"])
        if "fd" in obj:
            wlog(f"[_from_shm] {_key}: numpy memfd pid={obj['pid']} fd={obj['fd']} shape={shape}")
            data = _memfd_read(obj["pid"], obj["fd"], obj["size"])
            arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        else:
            shm_name = obj["__shm_np__"]
            wlog(f"[_from_shm] {_key}: opening shm '{shm_name}' shape={shape} dtype={dtype}")
            block = shm.SharedMemory(name=shm_name)
            try:
                from multiprocessing.resource_tracker import unregister
                unregister(block._name, "shared_memory")
            except Exception:
                pass
            arr = np.ndarray(shape, dtype=dtype, buffer=block.buf)
            _input_shm_blocks.append(block)
        wlog(f"[_from_shm] {_key}: mapped arr shape={arr.shape}")
        return arr

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
        if "fd" in obj:
            wlog(f"[_from_shm] {_key}: pickled memfd pid={obj['pid']} fd={obj['fd']} size={obj['size']}")
            obj_bytes = _memfd_read(obj["pid"], obj["fd"], obj["size"])
        else:
            wlog(f"[_from_shm] {_key}: pickled obj shm '{obj['name']}' size={obj['size']}")
            block = shm.SharedMemory(name=obj["name"])
            try:
                from multiprocessing.resource_tracker import unregister
                unregister(block._name, "shared_memory")
            except Exception:
                pass
            obj_bytes = bytes(block.buf[:obj["size"]])
            block.close()
        # Degrades to OpaquePickle when this env lacks the class (e.g. a
        # cross-pack type this worker only forwards) -- see _ipc_shared.
        return _ipc_shared.loads_or_opaque(obj_bytes)

    # Dict - recurse with key names for debugging
    if _depth == 0:
        wlog(f"[_from_shm] top-level keys: {list(obj.keys())}")
    return {k: _from_shm(v, _depth+1, k) for k, v in obj.items()}

def _cleanup_shm(registry):
    for item in registry:
        try:
            if isinstance(item, int):
                os.close(item)  # memfd fd
            else:
                item.close()
                item.unlink()
        except Exception:
            pass
    registry.clear()

# Shared memory keeper - holds references to prevent premature GC
class ShmKeeper:
    """Keep shm blocks alive until the parent acks the call (release()),
    with a TTL sweep as the crash fallback (see TensorKeeper).
    """
    def __init__(self, retention_seconds=_ipc_shared.TENSOR_KEEPER_TTL):
        self.retention_seconds = retention_seconds
        self._keeper = collections.deque()  # (time, call_id, blocks)
        self._lock = threading.Lock()

    def keep(self, blocks, call_id=None):
        now = time.time()
        with self._lock:
            self._keeper.append((now, call_id, list(blocks)))  # Copy the list
            # TTL sweep (fallback only -- release() is the real path)
            while self._keeper and now - self._keeper[0][0] > self.retention_seconds:
                _old_time, _old_id, old_blocks = self._keeper.popleft()
                _cleanup_shm(old_blocks)

    def release(self, call_id):
        """Free every block kept for `call_id` -- parent acked the call."""
        if call_id is None:
            return
        to_free = []
        with self._lock:
            kept = []
            for entry in self._keeper:
                if entry[1] == call_id:
                    to_free.append(entry[2])
                else:
                    kept.append(entry)
            self._keeper.clear()
            self._keeper.extend(kept)
        for blocks in to_free:
            _cleanup_shm(blocks)

    def count(self):
        with self._lock:
            return len(self._keeper)

_shm_keeper = ShmKeeper()


def _release_consumed(call_id):
    """Parent sent {"type": "consumed", "call_id": N}: every frame of call
    N's response has been read or copied into parent-owned memory. Free
    that call's keeper entries now instead of waiting out the TTL."""
    _tensor_keeper.release(call_id)
    _shm_keeper.release(call_id)

_input_shm_blocks = []  # Keep parent->worker shm blocks alive during request processing
_input_torch_storages = []  # Track parent-owned torch storages to balance _shared_incref
_worker_fd_registry = []  # Keep worker fds alive for worker->parent tensor transfer

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

    obj_type = type(obj).__name__  # noqa: F841

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
    """Length-prefixed JSON transport.

    send() MUST be locked: the worker routes print()/logging into
    transport.send, and pack code prints from threads it spawns -- an
    unlocked send interleaves partial writes from two threads and
    permanently desyncs the length-prefixed stream. Mirrors the parent's
    SocketTransport (_ipc_parent.py), including the MAX_MESSAGE_SIZE
    check the worker copy previously lacked.
    """
    def __init__(self, sock):
        self._sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    def send(self, obj):
        data = json.dumps(obj).encode("utf-8")
        msg = struct.pack(">I", len(data)) + data
        with self._send_lock:
            self._sock.sendall(msg)

    def recv(self, timeout=None):
        with self._recv_lock:
            if timeout is not None:
                self._sock.settimeout(timeout)
            try:
                raw_len = self._recvall(4)
                if not raw_len:
                    return None
                msg_len = struct.unpack(">I", raw_len)[0]
                if msg_len > _ipc_shared.MAX_MESSAGE_SIZE:
                    raise ValueError(f"Message too large: {msg_len} bytes")
                data = self._recvall(msg_len)
                return json.loads(data.decode("utf-8"))
            finally:
                if timeout is not None:
                    self._sock.settimeout(None)

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
        except OSError:
            pass


def _connect(addr):
    """Connect to server socket (abstract://, unix://, or tcp://)."""
    if addr.startswith("abstract://"):
        # Abstract Unix socket (Linux) — kernel namespace, no filesystem path
        name = f"\0{addr[11:]}"
        if _DBG_WORKER:
            wlog(f"[worker] abstract socket name={addr[11:]}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(name)
        return sock
    elif addr.startswith("unix://"):
        path = addr[7:]
        if _DBG_WORKER:
            wlog(f"[worker] socket path={path} exists={os.path.exists(path)} dir_exists={os.path.isdir(os.path.dirname(path))}")
            wlog(f"[worker] pid={os.getpid()} ppid={os.getppid()} cwd={os.getcwd()}")
            wlog(f"[worker] sys.argv={sys.argv}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
        except FileNotFoundError:
            # Always log this to stderr — worker is about to crash and wlog file may be lost
            print(f"[worker] FATAL: socket not found: path={path} exists={os.path.exists(path)} "
                  f"dir={os.path.dirname(path)} dir_exists={os.path.isdir(os.path.dirname(path))} "
                  f"argv={sys.argv}", file=sys.stderr, flush=True)
            raise
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
    # Socket address and auth token arrive via the ENVIRONMENT, never argv
    # (argv is world-readable through /proc/<pid>/cmdline).
    socket_addr = os.environ.get("COMFY_ENV_IPC_ADDR")
    authkey = os.environ.get("COMFY_ENV_IPC_AUTHKEY", "")
    if not socket_addr:
        wlog("[worker] COMFY_ENV_IPC_ADDR not set, exiting")
        sys.exit(1)
    wlog(f"[worker] Connecting to {socket_addr}...")

    # Connect to host process
    sock = _connect(socket_addr)
    transport = SocketTransport(sock)
    # First frame MUST be the auth token: the parent refuses to speak the
    # protocol (which carries pickled payloads) to an unauthenticated peer.
    transport.send({"authkey": authkey})
    # Give the VRAM poller access to transport for sending log messages to parent
    global _vram_poll_transport, _serializing_call_id
    _vram_poll_transport = transport
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

    # Load pack-declared custom serializers ([types] "custom" entries in
    # comfy-env-root.toml, forwarded as serialization.py file paths --
    # ADR-0015). Runs after sys.path setup so lazy imports inside the
    # functions resolve; failures are non-fatal (types stay opaque).
    _ipc_shared.load_serializer_files(
        os.environ.get("COMFY_ENV_SERIALIZER_FILES"), log=wlog)

    # Apply the parent process's folder_paths state so this worker's
    # folder_paths module (a separate import in this subprocess) resolves
    # input/output/temp/user to the SAME dirs ComfyUI actually uses. The
    # parent snapshotted them from its own `import folder_paths` before
    # sending config, so this covers every override mechanism (Comfy
    # Desktop inputDir, --input-directory, --base-directory,
    # extra_model_paths.yaml, etc.) without special-casing per host.
    _fps = config.get("folder_paths_state") or {}
    if _fps:
        try:
            import folder_paths
            if _fps.get("base_path"):
                folder_paths.base_path = _fps["base_path"]
            if _fps.get("input_directory"):
                folder_paths.set_input_directory(_fps["input_directory"])
            if _fps.get("output_directory"):
                folder_paths.set_output_directory(_fps["output_directory"])
            if _fps.get("temp_directory"):
                folder_paths.set_temp_directory(_fps["temp_directory"])
            if _fps.get("user_directory"):
                folder_paths.set_user_directory(_fps["user_directory"])
            # Rebuild folder_names_and_paths — the models search-paths
            # registry. Extensions were serialized as sorted list; rebuild
            # as set to match folder_paths' expected shape.
            _fnap = _fps.get("folder_names_and_paths") or {}
            if _fnap:
                folder_paths.folder_names_and_paths = {
                    k: (list(v[0]), set(v[1]))
                    for k, v in _fnap.items()
                    if isinstance(v, (list, tuple)) and len(v) == 2
                }
            wlog(f"[worker] folder_paths applied from parent: "
                 f"input={_fps.get('input_directory')} "
                 f"output={_fps.get('output_directory')} "
                 f"model_categories={len(_fnap)}")
        except ImportError:
            pass
    else:
        # Legacy fallback: honor COMFYUI_USER_DIR env var for older
        # invocations that don't pass folder_paths_state in config.
        _user_dir = os.environ.get("COMFYUI_USER_DIR")
        if _user_dir:
            try:
                import folder_paths
                folder_paths.base_path = _user_dir
                folder_paths.output_directory = os.path.join(_user_dir, "output")
                folder_paths.input_directory = os.path.join(_user_dir, "input")
                folder_paths.user_directory = os.path.join(_user_dir, "user")
                wlog(f"[worker] folder_paths redirected to {_user_dir} (legacy env-var path)")
            except ImportError:
                pass

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
    # Make the forwarder reachable as a module attribute of __main__. numba's
    # @infer_global(print) does getattr(sys.modules[print.__module__], print.__name__)
    # at import; without this, that lookup raises AttributeError and silently breaks
    # `import numba` in node code (forcing slow fallbacks). Harmless for everything else.
    globals()["_forwarded_print"] = _forwarded_print

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
    # Model registry -- tracks nn.Module instances on CUDA so the main
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
        """Register a model explicitly (optional -- auto-hook handles most cases)."""
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

    def _register_if_cuda(module):
        """Register an nn.Module with parent if it's on CUDA.

        Like _auto_register_if_cuda but bypasses the _loading_via_shim guard.
        Called after shimmed load_models_gpu to ensure the parent can evict
        models that were loaded inside the shim.
        """
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
        wlog(f"[worker] Post-shim registered '{model_id}': {size / 1e9:.2f} GB")

    # Install hooks on Module.to() and .cuda()
    # Module.to() only fires for the outermost call -- PyTorch recurses
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
    # Bidirectional RPC -- call parent methods during execution
    # ---------------------------------------------------------------
    _current_call_id = None  # Tracks call_id of the request being processed

    def _find_loaded_model(_model):
        """The worker's own ComfyUI LoadedModel wrapping this module, if any."""
        try:
            import comfy.model_management as _cmm_f
            for _lm in list(_cmm_f.current_loaded_models):
                if _lm.model is not None and _lm.model.model is _model:
                    return _lm
        except Exception:
            pass
        return None

    def _handle_model_partial(request):
        """Byte-quantized partial load/unload against the REAL ModelPatcher.

        The parent's proxy has no weights and cannot do partial residency, so
        it asks for bytes and we answer with bytes ACTUALLY moved -- the parent
        must never guess. Falls back to a whole-model move when this module is
        not under a ComfyUI patcher (a plain nn.Module a pack moved itself).
        """
        _mid = request.get("model_id")
        _req_call_id = request.get("call_id", _current_call_id)
        _partial_unload = request.get("method") == "model_partial_unload"
        _model = _model_registry.get(_mid)
        if _model is None:
            transport.send({"status": "error", "call_id": _req_call_id,
                            "error": f"Model '{_mid}' not registered"})
            return
        try:
            import torch as _torch
            _lm = _find_loaded_model(_model)
            _resident_of = (lambda: int(_lm.model.loaded_size())) if _lm is not None else None

            if _lm is not None:
                _before = _resident_of()
                if _partial_unload:
                    _want = int(request.get("bytes_to_free", 0)) or _before
                    _lm.model.partially_unload(_lm.model.offload_device, _want)
                    # MANDATORY: bytes must reach the driver, or the parent
                    # (which decides admission from device-wide free) cannot
                    # see them and will keep evicting.
                    if _torch.cuda.is_available():
                        _torch.cuda.empty_cache()
                    _after = _resident_of()
                    _moved = max(0, _before - _after)
                else:
                    _extra = int(request.get("extra_bytes", 0))
                    _lm.model.partially_load(_lm.model.load_device, _extra)
                    _after = _resident_of()
                    _moved = max(0, _after - _before)
                transport.send({"status": "ok", "call_id": _req_call_id,
                                "freed" if _partial_unload else "loaded": _moved,
                                "resident": _after})
                return

            # No patcher: plain module. Whole-model move, report honestly.
            _meta = _model_registry_meta.get(_mid, {})
            _size = int(_meta.get("size", 0))
            _cur = next(_model.parameters(), None)
            _on_cuda = _cur is not None and _cur.device.type == "cuda"
            if _partial_unload:
                if not _on_cuda:
                    transport.send({"status": "ok", "call_id": _req_call_id,
                                    "freed": 0, "resident": 0})
                    return
                _model.to(_torch.device("cpu"))
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                transport.send({"status": "ok", "call_id": _req_call_id,
                                "freed": _size, "resident": 0})
            else:
                if _on_cuda:
                    transport.send({"status": "ok", "call_id": _req_call_id,
                                    "loaded": 0, "resident": _size})
                    return
                _model.to(_torch.device(request.get("device", "cuda")))
                transport.send({"status": "ok", "call_id": _req_call_id,
                                "loaded": _size, "resident": _size})
        except Exception as _e:
            wlog(f"[worker] model_partial error: {_e}")
            transport.send({"status": "error", "call_id": _req_call_id, "error": str(_e)})

    def _handle_model_to_device(request):
        """Handle a model_to_device command. Can be called from main loop or _call_parent."""
        _mid = request.get("model_id")
        _target = request.get("device", "cpu")
        _req_call_id = request.get("call_id", _current_call_id)
        _model = _model_registry.get(_mid)
        if _model is None:
            transport.send({"status": "error", "call_id": _req_call_id,
                            "error": f"Model '{_mid}' not registered"})
            return
        try:
            import torch as _torch
            _target_dev = _torch.device(_target)
            _current_dev = None
            try:
                _first_param = next(_model.parameters(), None)
                if _first_param is not None:
                    _current_dev = _first_param.device
            except Exception:
                pass
            if _current_dev is not None and _current_dev == _target_dev:
                wlog(f"[worker] model_to_device: '{_mid}' already on {_target}")
                transport.send({"status": "ok", "call_id": _req_call_id, "device": _target, "moved": False})
                return
            _was_cuda = _current_dev is not None and _current_dev.type == "cuda"
            wlog(f"[worker] model_to_device: '{_mid}' -> {_target}")
            _used_patcher = False
            try:
                import comfy.model_management as _cmm_move
                for _lm in list(_cmm_move.current_loaded_models):
                    if _lm.model is not None and _lm.model.model is _model:
                        if _target_dev.type == "cpu":
                            _lm.model_unload()
                            # Remove from current_loaded_models to avoid
                            # zombie entries.  model_unload() sets
                            # real_model = None which makes is_dead() crash
                            # (TypeError: 'NoneType' is not callable)
                            # because cleanup_models_gc() expects real_model
                            # to be either a live weakref or absent.
                            try:
                                _cmm_move.current_loaded_models.remove(_lm)
                            except ValueError:
                                pass
                        else:
                            _lm.model_load()
                        _used_patcher = True
                        break
            except Exception as _pe:
                wlog(f"[worker] model_to_device: patcher path failed ({_pe}), falling back to .to()")
            if not _used_patcher:
                _model.to(_target_dev)
            if _was_cuda and _target_dev.type == "cpu":
                _torch.cuda.empty_cache()
            transport.send({"status": "ok", "call_id": _req_call_id, "device": _target, "moved": True})
        except Exception as _e:
            wlog(f"[worker] model_to_device error: {_e}")
            transport.send({"status": "error", "call_id": _req_call_id, "error": str(_e)})

    def _call_parent(method, **params):
        """Call a method on the parent process and wait for result.

        Can only be called during method execution (while transport is active).
        The parent handles the callback and sends back a response.
        Handles interleaved management commands (model_to_device, ping, etc.)
        that may arrive while waiting for the callback_response.
        """
        transport.send({"type": "callback", "method": method, "call_id": _current_call_id, **params})
        while True:
            response = transport.recv()
            if response is None:
                raise RuntimeError("Parent disconnected during callback")
            # Handle interleaved management commands
            if response.get("method") == "model_to_device":
                _handle_model_to_device(response)
                continue
            if response.get("method") in ("model_partial_unload", "model_partial_load"):
                _handle_model_partial(response)
                continue
            if response.get("method") == "ping":
                transport.send({"status": "pong", "call_id": response.get("call_id")})
                continue
            if response.get("method") == "list_models":
                transport.send({"status": "ok", "call_id": response.get("call_id"), "models": _model_registry_meta})
                continue
            if response.get("type") == "consumed":
                # Ack for an earlier call arriving while we wait -- handle
                # here too so it is never mistaken for a callback_response.
                _release_consumed(response.get("call_id"))
                continue
            if response.get("method") == "shutdown":
                raise RuntimeError("Shutdown requested during callback")
            # Check for actual callback_response
            if response.get("type") == "callback_response":
                if response.get("status") == "error":
                    raise RuntimeError(response.get("error", "Callback failed"))
                return response.get("result")
            # Unknown message — log and skip
            wlog(f"[worker] _call_parent: unexpected message type={response.get('type')}, keys={list(response.keys())}")

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
    # Propagate --cpu flag from parent process.  When the parent is
    # started with --cpu, it sets COMFY_CPU=1 in our env.  We mirror
    # that into comfy.cli_args so comfy.model_management sets
    # cpu_state = CPUState.CPU and get_torch_device() returns cpu.
    # This MUST run before comfy.model_management is imported below.
    # ---------------------------------------------------------------
    if os.environ.get("COMFY_CPU") == "1":
        try:
            from comfy.cli_args import args as _cli_args
            _cli_args.cpu = True
            wlog("[worker] Set args.cpu=True (COMFY_CPU=1)")
        except Exception:
            pass

    # ---------------------------------------------------------------
    # Shim comfy.model_management.load_models_gpu -- tell parent to
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

                    # Correct OUR OWN blindness. get_free_memory() here reports
                    # this process's budget on WDDM -- it cannot see the parent
                    # or sibling workers, so the real load_models_gpu below
                    # would size lowvram_model_memory against a card it thinks
                    # is empty and OVER-load into driver sysmem fallback (the
                    # unexplained 10x slowdowns). The parent measured true
                    # device-wide free; the difference is exactly what everyone
                    # else holds, so reserving it makes our view honest.
                    _dev_free = result.get("device_free_bytes")
                    if _dev_free is not None:
                        try:
                            _my_blind = _cmm.get_free_memory(_cmm.get_torch_device())
                            _others = max(0, int(_my_blind) - int(_dev_free))
                            extra_reserved = max(int(extra_reserved or 0), _others)
                            wlog(f"[worker] blindness correction: my_free="
                                 f"{_my_blind / 1e9:.2f}GB device_free="
                                 f"{_dev_free / 1e9:.2f}GB -> reserve "
                                 f"{_others / 1e9:.2f}GB for other processes")
                        except Exception as _be:
                            wlog(f"[worker] blindness correction skipped: {_be}")

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

                # Now run the real load_models_gpu -- it calls get_free_memory()
                # which uses EXTRA_RESERVED_VRAM via minimum_inference_memory(),
                # so it will calculate lowvram_model_memory correctly.
                _original_load_models_gpu(models, *args, **kwargs)
                wlog(f"[worker] Models loaded via real load_models_gpu")

                # Register loaded models with parent so they participate in
                # cross-process VRAM eviction.  The auto-hook was suppressed
                # during the shim (_loading_via_shim=True), so the parent
                # doesn't know about these models yet.  Without this, the
                # parent's free_memory() can't evict them when another
                # subprocess needs VRAM.
                for m in models:
                    model_obj = getattr(m, 'model', None)
                    if model_obj is not None and hasattr(model_obj, 'parameters'):
                        _register_if_cuda(model_obj)
            finally:
                _loading_via_shim[0] = False

        _cmm.load_models_gpu = _shimmed_load_models_gpu
        wlog("[worker] Installed load_models_gpu shim (budget-based)")
    except Exception as e:
        wlog(f"[worker] comfy.model_management not available ({type(e).__name__}: {e}), skipping load_models_gpu shim")

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
    except Exception as e:
        wlog(f"[worker] comfy.utils not available ({type(e).__name__}: {e}), skipping progress hook")

    # Expose explicit API as comfy_worker module (optional override)
    import types as _types
    _comfy_worker = _types.ModuleType("comfy_worker")
    _comfy_worker.__doc__ = "Helper for registering models with the comfy-env worker."
    _comfy_worker.register_model = _register_model
    _comfy_worker.call_parent = _call_parent
    sys.modules["comfy_worker"] = _comfy_worker

    # Signal ready
    transport.send({"status": "ready"})
    wlog("[worker] Ready")

    # --- Pool IPC handshake: create shareable pool and send FD to parent ---
    global _pool_ipc_ok, _our_pool, _parent_pool
    if _POOL_IPC_ENABLED and sys.platform == "linux":
        try:
            import torch as _pt
            if _pt.cuda.is_available():
                device = _pt.cuda.current_device()
                _our_pool = _create_shareable_pool(device)
                _set_device_pool(device, _our_pool)
                wlog(f"[worker] Pool IPC: created shareable pool on device {device}")

                # Patch empty_cache to also trim our pool
                _orig_empty_cache = _pt.cuda.empty_cache
                def _patched_empty_cache():
                    _orig_empty_cache()
                    try:
                        if _our_pool is not None:
                            _trim_pool(_our_pool, 0)
                    except Exception:
                        pass
                _pt.cuda.empty_cache = _patched_empty_cache

                # Send pool FD to parent
                pool_fd = _export_pool_fd(_our_pool)
                _send_fd(sock, pool_fd)
                os.close(pool_fd)
                transport.send({"type": "pool_fd_sent", "device": device})
                _pool_ipc_ok = True
                wlog("[worker] Pool IPC: handshake complete")
                print(
                    "[comfy-env] WARNING: Pool IPC is EXPERIMENTAL and has "
                    "known lifetime hazards (imported pointers are never "
                    "freed; exporter-side cache eviction can free memory "
                    "under a live parent alias; no cross-process sync "
                    "protocol). Known-unsound until the pluggable-allocator "
                    "redesign (ADR-0010 v2 item 6) -- do not enable outside "
                    "experiments.", file=sys.stderr, flush=True)
        except Exception as e:
            wlog(f"[worker] Pool IPC setup failed: {e}, using CPU shm fallback")
            _pool_ipc_ok = False
            _our_pool = None

    # --- Receive parent's shareable pool FD (for parent->worker zero-copy) ---
    _parent_pool = None
    try:
        msg = transport.recv(timeout=5)
        if msg and msg.get("type") == "parent_pool_fd_sent":
            parent_fd = _recv_fd(sock, timeout=5)
            _parent_pool = _import_pool_from_fd(parent_fd)
            os.close(parent_fd)
            wlog("[worker] Pool IPC: imported parent pool for parent->worker zero-copy")
        else:
            wlog("[worker] No parent shareable pool (parent->worker uses CPU shm)")
    except Exception as e:
        wlog(f"[worker] Parent pool import skipped: {e}")
        _parent_pool = None

    wlog("[worker] Entering request loop...")

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

        _current_call_id = request.get("call_id")

        if request.get("method") == "shutdown":
            wlog("[worker] Shutdown requested")
            break

        if request.get("method") == "ping":
            # Health check - respond immediately. Keeper counts ride along
            # for tests/doctor (how many un-acked calls still pin memory).
            transport.send({"status": "pong", "call_id": _current_call_id,
                            "keepers": {"tensors": _tensor_keeper.count(),
                                        "shm": _shm_keeper.count()}})
            continue

        if request.get("type") == "consumed":
            # One-way ack from the parent: it finished reading (or copied)
            # every frame of that call's response. No reply expected.
            _release_consumed(request.get("call_id"))
            continue

        if request.get("method") == "model_to_device":
            _handle_model_to_device(request)
            continue

        if request.get("method") in ("model_partial_unload", "model_partial_load"):
            _handle_model_partial(request)
            continue

        if request.get("method") == "list_models":
            # Return registered model metadata
            transport.send({"status": "ok", "call_id": _current_call_id, "models": _model_registry_meta})
            continue

        # Release input shm blocks from previous request
        for _old_block in _input_shm_blocks:
            try:
                _old_block.close()
            except Exception:
                pass
        _input_shm_blocks.clear()

        # Balance _shared_incref for parent-owned torch storages
        for _old_storage in _input_torch_storages:
            try:
                _old_storage._shared_decref()
            except Exception:
                pass
        _input_torch_storages.clear()

        # Close fds from previous worker->parent result transfer
        for _old_fd in _worker_fd_registry:
            try:
                os.close(_old_fd)
            except OSError:
                pass
        _worker_fd_registry.clear()

        # Clear new-models tracker for this call
        _new_models_this_call.clear()

        # Defensive: skip stale callback_responses or unknown messages
        if request.get("type") == "callback_response":
            wlog(f"[worker] Ignoring stale callback_response in main loop")
            continue

        # Transport canary: round-trip the payload through the PRODUCTION
        # serialization path (_from_shm -> _to_shm) so the parent can verify
        # each transport tier actually works for this parent/worker pair.
        # MUST use the same code path as real calls -- a parallel test
        # serializer would validate nothing.
        if request.get("type") == "echo":
            shm_registry = []
            try:
                payload = _from_shm(request.get("kwargs") or {})
                payload = _deserialize_input(payload)
                _serializing_call_id = _current_call_id
                result_meta = _to_shm(payload, shm_registry)
                try:
                    import torch as _echo_torch
                    _echo_tv = getattr(_echo_torch, "__version__", None)
                except ImportError:
                    _echo_tv = None
                # CUDA device UUID of the worker's CURRENT device -- lets the
                # parent detect enumeration skew (a pack env that set its own
                # CUDA_VISIBLE_DEVICES makes device_idx=0 a DIFFERENT physical
                # GPU on each side, which would be a silent wrong-device
                # import on the zero-copy tiers). None when no CUDA.
                _echo_uuid = _ipc_shared._cuda_device_uuid()
                transport.send({"status": "ok", "call_id": _current_call_id,
                                "result": result_meta, "torch_version": _echo_tv,
                                "cuda_device_uuid": _echo_uuid})
                _shm_keeper.keep(shm_registry, _current_call_id)
            except Exception as e:
                _cleanup_shm(shm_registry)
                transport.send({"status": "error", "call_id": _current_call_id,
                                "error": str(e),
                                "traceback": traceback.format_exc()})
            continue

        if "module" not in request:
            wlog(f"[worker] Ignoring unknown request format: {list(request.keys())}")
            continue

        shm_registry = []
        try:
            request_type = request.get("type", "call_module")
            module_name = request["module"]
            wlog(f"[worker] Request: {request_type} {module_name} call_id={_current_call_id}")

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

            try:
                import torch as _torch_worker
                _infer_mode = _torch_worker.inference_mode
            except ImportError:
                import contextlib as _contextlib_worker
                _infer_mode = _contextlib_worker.nullcontext
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
                with _infer_mode():
                    result = method(**inputs)
                wlog(f"[worker] Method returned")
            else:
                func_name = request["func"]
                func = getattr(module, func_name)
                with _infer_mode():
                    result = func(**inputs)

            # Serialize result to shared memory
            wlog(f"[worker] Serializing result to shm...")
            _serializing_call_id = _current_call_id
            result_meta = _to_shm(result, shm_registry)
            wlog(f"[worker] Created {len(shm_registry)} shm blocks for result")

            response = {"status": "ok", "call_id": _current_call_id, "result": result_meta}
            if _new_models_this_call:
                # Resolve actual device at response time.  Models are
                # auto-detected when they land on CUDA, but the subprocess
                # may have moved them back to CPU before the call finished.
                for _nme in _new_models_this_call:
                    _nm_model = _model_registry.get(_nme["id"])
                    if _nm_model is not None:
                        try:
                            _nm_p = next(_nm_model.parameters(), None)
                            _nme["device"] = str(_nm_p.device) if _nm_p is not None else "cpu"
                        except Exception:
                            _nme["device"] = "cpu"
                    else:
                        _nme["device"] = "cpu"
                response["_new_models"] = list(_new_models_this_call)
            transport.send(response)
            # Kept until the parent's "consumed" ack (TTL is the fallback)
            _shm_keeper.keep(shm_registry, _current_call_id)

        except Exception as e:
            # Cleanup shm on error since host won't read it
            _cleanup_shm(shm_registry)
            transport.send({
                "status": "error",
                "call_id": _current_call_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    transport.close()

if __name__ == "__main__":
    main()
