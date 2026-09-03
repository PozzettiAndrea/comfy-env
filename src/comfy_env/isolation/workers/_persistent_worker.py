
import sys
import json
import os
import socket
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
            # No os.fsync: it cost 2.78 ms per line vs 0.02 ms without (measured,
            # ext4). With 92 call sites -- 13 of them inside _from_shm's
            # per-node recursion -- that was 50-100 ms of pure fsync on a
            # typical call, dwarfing the transport itself. flush() already
            # survives a Python-level crash; only a kernel panic loses the tail,
            # and a kernel panic loses the worker anyway.
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

# Shared Memory Serialization

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
# Staged next to this script by subprocess.py, same reason as _ipc_shared: the
# worker must be able to report which memory manager it resolved to even when
# comfy_env itself is not importable in its environment.
try:
    import memory_manager as _memmgr
except Exception:  # pragma: no cover - the worker must start without it
    _memmgr = None
try:
    import state_sync as _state_sync
except Exception:  # pragma: no cover - the worker must start without it
    _state_sync = None
try:
    import mirrored_args as _mirrored_args
except Exception:  # pragma: no cover - the worker must start without it
    _mirrored_args = None
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

        # input_files twin: packs call `from comfy_env import input_files` in
        # their nodes modules, which import HERE (no comfy_env installed --
        # ADR-0006). The worker never needs the provenance tag (nothing scans
        # options in this process), so a plain live listing suffices. Keep in
        # step with isolation/provided.py's _list_sources semantics.
        def _ce_input_files(sources, exts=None, placeholder=None):
            import os as _os
            if isinstance(sources, str):
                sources = [sources]
            norm = []
            for _s in sources:
                if isinstance(_s, str):
                    norm.append({"dir": _s, "recursive": False,
                                 "rel_to_input": False})
                else:
                    _d = dict(_s)
                    _d.setdefault("dir", "")
                    _d.setdefault("recursive", False)
                    _d.setdefault("rel_to_input", False)
                    norm.append(_d)
            _ex = set(str(_e).lower() for _e in (exts or []))
            names, seen = [], set()
            try:
                import folder_paths as _fp
                base = _fp.get_input_directory()
            except Exception:
                base = None
            if base is not None:
                for _src in norm:
                    _sub = _src.get("dir", "") or ""
                    _root = _os.path.join(base, _sub) if _sub else base
                    try:
                        if _src.get("recursive"):
                            for _r, _dd, _ff in _os.walk(_root):
                                for _fn in _ff:
                                    if _ex and _os.path.splitext(_fn)[1].lower() not in _ex:
                                        continue
                                    _rel = _os.path.relpath(
                                        _os.path.join(_r, _fn),
                                        base if _src.get("rel_to_input") else _root)
                                    _v = _rel.replace(_os.sep, "/")
                                    if _v not in seen:
                                        seen.add(_v)
                                        names.append(_v)
                        else:
                            for _fn in _os.listdir(_root):
                                if not _os.path.isfile(_os.path.join(_root, _fn)):
                                    continue
                                if _ex and _os.path.splitext(_fn)[1].lower() not in _ex:
                                    continue
                                _v = (_os.path.join(_sub, _fn).replace(_os.sep, "/")
                                      if _src.get("rel_to_input") and _sub else _fn)
                                if _v not in seen:
                                    seen.add(_v)
                                    names.append(_v)
                    except Exception:
                        continue
            names.sort()
            if not names and placeholder is not None:
                names = [placeholder]
            return names

        _ce_stub.input_files = _ce_input_files
        _ce_stub.__all__ = ["register_serializer", "input_files"]
        sys.modules["comfy_env"] = _ce_stub
from _ipc_shared import (
    _cleanup_shm,
    _memfd_read,
    _create_shareable_pool,
    _export_pool_fd,
    _set_device_pool,
    _export_pointer,
    _trim_pool,
    _send_fd,
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

def _serialize_cuda_ipc(t):
    import torch.multiprocessing.reductions as reductions
    # Check IPC handle cache -- forward original handle if available
    try:
        storage_id = id(t.untyped_storage())
        cached = _ipc_shared._cuda_ipc_metadata_cache.get(storage_id)
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


# Pool IPC - shareable CUDA memory pool (worker side)

_POOL_IPC_ENABLED = os.environ.get("COMFY_ENV_POOL_IPC", "").lower() in ("1", "true", "yes")
_pool_ipc_ok = False
_our_pool = None
# Parent's shareable pool (parent->worker zero-copy); imported in main()'s
# handshake, read by the module-level _from_shm().
_pool_ipc_metadata_cache = {}
_pool_ipc_cache_tensors = {}

# Pool ctypes primitives come from _ipc_shared (imported at the
# top of this file). The duplicated definitions that lived here drifted from
# the shared copies before being deleted -- do not reintroduce them.

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

    # CudaIPC -> zero-copy CUDA tensor deserialization
    if obj.get("__type__") == "CudaIPC":
        wlog(f"[_from_shm] {_key}: CudaIPC tensor_size={obj.get('tensor_size')}")
        return _ipc_shared._deserialize_cuda_ipc(obj)

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
    transport = _ipc_shared.SocketTransport(sock)
    # First frame MUST be the auth token: the parent refuses to speak the
    # protocol (which carries pickled payloads) to an unauthenticated peer.
    transport.send({"authkey": authkey})
    # Give the VRAM poller access to transport for sending log messages to parent
    global _vram_poll_transport, _serializing_call_id
    _vram_poll_transport = transport
    wlog("[worker] Connected, waiting for config...")

    # Read config as first message. recv() raises on a closed socket now, and
    # a parent that dies before sending config is a routine shutdown, not a
    # fault -- main() is called unguarded, so an escape here is a traceback.
    try:
        config = transport.recv()
    except (ConnectionError, OSError) as e:
        wlog(f"[worker] Parent closed before sending config ({e}), exiting")
        return
    if not config:
        wlog("[worker] No config received, exiting")
        return
    wlog("[worker] Got config, setting up paths...")

    # Setup sys.path
    for p in config.get("sys_paths", []):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Mirror the host's CLI args BEFORE anything imports comfy. This must sit
    # here, in main's direct body: `import folder_paths` below transitively
    # runs `from comfy.cli_args import args`, and the memory-relevant reads
    # (DISABLE_SMART_MEMORY, NUM_STREAMS, the MAX_PINNED gate, DISABLE_MMAP)
    # execute ONCE at comfy.model_management/comfy.utils import, so a later
    # apply is unrecoverable. Earlier is impossible: comfy.cli_args is not
    # importable until the parent's sys_paths (just above) land. The ast
    # guard in tests/test_mirrored_args.py pins this ordering.
    _mirror_payload = None
    _mirror_report = {"applied": [], "skipped": []}
    if _mirrored_args is not None:
        try:
            _mp_raw = os.environ.get(_mirrored_args.MIRROR_ENV_VAR)
            if _mp_raw:
                _mirror_payload = json.loads(_mp_raw)
                from comfy.cli_args import args as _host_args
                _mirror_report = _mirrored_args.apply_host_args(
                    _host_args, _mirror_payload, log=wlog)
        except Exception as _me:
            wlog(f"[worker] host args mirror failed: {_me}")

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
    # Per-model residency sequence. Bumped at every byte-moving site so the
    # parent can order frame censuses against command echoes and drop stale
    # ones: the only defence against a census resurrecting freed bytes.
    _residency_seq = {}

    # ------------- node state overflow tier (state_sync design) -------------
    # Values that cannot cross the wire (device resident, unpicklable, over
    # cap) stay HERE, keyed by a monotonic handle, represented parent-side by
    # a marker. _STATE_GEN changes on every worker start so a marker from a
    # dead worker raises a pointed error instead of a silent fresh default.
    import uuid as _uuid
    _STATE_GEN = _uuid.uuid4().hex[:8]
    _overflow_store = {}          # handle -> (owner_state_id, value)
    _overflow_counter = [0]

    def _mint_handle():
        _overflow_counter[0] += 1
        return _overflow_counter[0]

    # the state_out computed by the current call, attached to whichever frame
    # goes out (ok or error), then cleared at the next request
    _pending_state_out = [None]

    def _bump_seq(mid):
        _residency_seq[mid] = _residency_seq.get(mid, 0) + 1
        return _residency_seq[mid]

    # ----------------- prompt-epoch pin marks (current_prompt) -------------
    # A worker runs no executor, so upstream's PromptModelTracker never marks
    # its models and pin-eviction tier 1 (cp False, active NOT consulted)
    # evicts this prompt's warm models like stale leftovers, including a
    # model mid-load whose unregistered staging pages can be decommitted
    # under an in-flight async copy. The parent forwards a prompt epoch on
    # every request; marks clear at epoch change (preamble, before the
    # method runs) and are set in the shim BEFORE the load (the pressure
    # fires during the load itself). No token means the sticky-with-decay
    # fallback, never dark.
    _prompt_marks = [{}]
    _mark_call_n = [0]
    _active_prompt_gen = [None]
    _resident_at_call_start = [set()]
    _pin_marks_enabled = os.environ.get(
        "COMFY_ENV_PIN_MARKS", "1").strip().lower() not in ("0", "false", "off")

    def _dynamic_patcher_registry():
        """key -> dynamic patcher, from the worker's own ledger. Keyed by the
        inner model object's id (stable across patcher clones, the same key
        space upstream's tracker uses)."""
        out = {}
        try:
            import comfy.model_management as _mpm
            for _lm in list(_mpm.current_loaded_models):
                _pt = getattr(_lm, "model", None)
                if _pt is None:
                    continue
                try:
                    if _pt.is_dynamic():
                        out[str(id(_pt.model))] = _pt
                except Exception:
                    continue
        except Exception:
            pass
        return out

    def _prompt_marks_preamble(request):
        """Clear the previous prompt's marks BEFORE this call's method (and
        therefore before any of its loads) runs."""
        if not _pin_marks_enabled or _state_sync is None or _memmgr is None:
            return
        try:
            _mark_call_n[0] += 1
            _gen = request.get("prompt_gen")
            if _gen is not None and _gen != _active_prompt_gen[0]:
                _memmgr.reset_pin_churn_epoch()
            _active_prompt_gen[0] = _gen
            registry = _dynamic_patcher_registry()
            # Snapshot residency so the node-end sweep marks only models that
            # became resident DURING this call (a shim-bypassing load), never
            # survivors of a previous call: re-marking survivors would mark
            # everything resident to the current prompt, the blanket-marking
            # hazard the contract forbids.
            _resident_at_call_start[0] = set(registry.keys())
            new_marks, to_clear = _state_sync.clear_on_epoch_change(
                _prompt_marks[0], _gen, _mark_call_n[0], registry.keys())
            _prompt_marks[0] = new_marks
            if to_clear:
                _memmgr.apply_prompt_marks(registry, [], to_clear, log=wlog)
        except Exception as _pme:
            wlog(f"[worker] prompt mark preamble failed: {_pme}")

    def _prompt_marks_on_load(models):
        """Mark the models a shimmed load is about to touch, before the real
        load_models_gpu runs. Sweeps patches and nested additional models,
        mirroring upstream tracker.add."""
        if not _pin_marks_enabled or _state_sync is None or _memmgr is None:
            return
        try:
            registry = {}
            for _m in models:
                sweep = [_m]
                try:
                    sweep.extend(_m.model_patches_models())
                except Exception:
                    pass
                try:
                    sweep.extend(_m.get_nested_additional_models())
                except Exception:
                    pass
                for _sm in sweep:
                    try:
                        if _sm.is_dynamic():
                            registry[str(id(_sm.model))] = _sm
                    except Exception:
                        continue
            if not registry:
                return
            new_marks, to_set = _state_sync.mark_on_load(
                _prompt_marks[0], _active_prompt_gen[0], _mark_call_n[0],
                registry.keys())
            _prompt_marks[0] = new_marks
            if to_set:
                _memmgr.apply_prompt_marks(registry, to_set, [], log=wlog)
        except Exception as _pme:
            wlog(f"[worker] prompt mark on load failed: {_pme}")

    def _prompt_marks_sweep():
        """Node-end catch-up: mark dynamic models that became resident THIS
        call outside the shim (a load that bypassed load_models_gpu). Scoped
        to models newly present since the preamble snapshot, so a survivor of
        a previous call is never re-marked to the current prompt (that would
        be blanket marking; a status call that loads nothing marks nothing)."""
        if not _pin_marks_enabled or _state_sync is None or _memmgr is None:
            return
        try:
            registry = _dynamic_patcher_registry()
            newly = set(registry.keys()) - _resident_at_call_start[0]
            missing = [k for k in newly if k not in _prompt_marks[0]]
            if not missing:
                return
            new_marks, to_set = _state_sync.mark_on_load(
                _prompt_marks[0], _active_prompt_gen[0], _mark_call_n[0],
                missing)
            _prompt_marks[0] = new_marks
            if to_set:
                _memmgr.apply_prompt_marks(registry, to_set, [], log=wlog)
        except Exception:
            pass
    _model_registry_meta = {}     # model_id -> {"size": int, "kind": str}
    _model_id_by_obj = {}         # id(module) -> model_id  (dedup)
    _model_counter = [0]          # mutable counter in list for closure access
    # Serialises the registry writes below. _hooked_to/_hooked_cuda replace
    # torch.nn.Module.to/.cuda GLOBALLY, so they fire on whatever thread a
    # pack uses. `_model_counter[0] += 1` is a read-modify-write: two models
    # reaching CUDA concurrently could mint the SAME model_id, leaving the
    # loser GPU-resident with no ledger entry -- permanently un-evictable.
    _registry_lock = threading.Lock()
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
        _bump_seq(model_id)
        wlog(f"[worker] Registered model '{model_id}': {size / 1e9:.2f} GB, kind={kind}")
        return size

    def _register_cuda_module(module, label):
        """Mint an id for a CUDA module and record it. Returns the id or None.

        One body for both registration paths. _hooked_to/_hooked_cuda replace
        torch.nn.Module.to/.cuda GLOBALLY, so this runs on whatever thread a
        pack uses: the dedup check is re-tested under the lock, and the
        counter bump is a read-modify-write that must not interleave.
        """
        obj_id = id(module)
        if obj_id in _model_id_by_obj:
            return None
        try:
            first_param = next(module.parameters(), None)
            if first_param is None or first_param.device.type != "cuda":
                return None
        except Exception:
            return None
        with _registry_lock:
            if obj_id in _model_id_by_obj:
                return None
            _model_counter[0] += 1
            model_id = f"{module.__class__.__name__}_{_model_counter[0]}"
            size = _compute_model_size(module)
            _model_registry[model_id] = module
            _model_registry_meta[model_id] = {"size": size, "kind": "other"}
            _model_id_by_obj[obj_id] = model_id
            _new_models_this_call.append({"id": model_id, "size": size, "kind": "other"})
            _bump_seq(model_id)
        wlog(f"[worker] {label} '{model_id}': {size / 1e9:.2f} GB")
        return model_id

    def _auto_register_if_cuda(module):
        """Auto-register an nn.Module if it just landed on CUDA."""
        if _loading_via_shim[0]:
            return  # Parent already coordinates VRAM during shimmed loads
        _register_cuda_module(module, "Auto-registered")

    def _register_if_cuda(module):
        """Register an nn.Module with parent if it's on CUDA.

        Like _auto_register_if_cuda but bypasses the _loading_via_shim guard.
        Called after shimmed load_models_gpu to ensure the parent can evict
        models that were loaded inside the shim.
        """
        _register_cuda_module(module, "Post-shim registered")

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

    # Bidirectional RPC -- call parent methods during execution
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
                                "resident": _after,
                                "seq": _bump_seq(_mid)})
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
                # seq even on moved False: the parent still zeroes its
                # books, and the fresh seq lets that zero supersede any
                # census sampled before this command.
                transport.send({"status": "ok", "call_id": _req_call_id,
                                "device": _target, "moved": False,
                                "seq": _bump_seq(_mid)})
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
            transport.send({"status": "ok", "call_id": _req_call_id,
                            "device": _target, "moved": True,
                            "seq": _bump_seq(_mid)})
        except Exception as _e:
            wlog(f"[worker] model_to_device error: {_e}")
            transport.send({"status": "error", "call_id": _req_call_id, "error": str(_e)})

    class _InterruptedError(RuntimeError):
        """Raised when the user cancels the current run.

        Defined at main scope, not inside the progress-hook try: the error
        frame stamper does an isinstance check against it, and a name that
        only exists when comfy.utils imported would NameError the error path
        of every worker whose env lacks comfy."""
        pass

    def _oom_stats():
        """Three allocator-level integers for an OOM error frame.

        Defined as raw torch.cuda reads on purpose: mirrored flags (async
        offload, dtypes) change what sits INSIDE reserved, but never how
        these are measured, so the numbers stay comparable across mirror
        settings. largest_free_block is the biggest inactive contiguous
        block, the fragmentation signal the host cannot reconstruct."""
        stats = {"allocated": None, "reserved": None, "largest_free_block": None}
        try:
            import torch
            stats["allocated"] = int(torch.cuda.memory_allocated())
            stats["reserved"] = int(torch.cuda.memory_reserved())
            largest = 0
            for seg in torch.cuda.memory_snapshot():
                for blk in seg.get("blocks", []):
                    if blk.get("state") == "inactive":
                        largest = max(largest, int(blk.get("size", 0)))
            stats["largest_free_block"] = largest
        except Exception:
            pass
        return stats

    def _error_kind_fields(e):
        """Typed verdict for an error frame, from the LIVE exception object.

        The verdict is computed here, at the raise site, where the object
        still exists: the parent only ever sees strings. mm.is_oom is
        ComfyUI's own function (isinstance plus the AcceleratorError code 2
        case), so the worker and a non-isolated node agree by construction.
        Never derived from message text. Any failure omits the keys, which
        degrades to today's untyped WorkerError."""
        fields = {}
        try:
            if isinstance(e, _InterruptedError):
                fields["error_kind"] = "interrupt"
                return fields
            import comfy.model_management as _emm
            if _emm.is_oom(e):
                fields["error_kind"] = "oom"
                fields["oom_stats"] = _oom_stats()
        except Exception:
            return {}
        return fields

    def _call_parent(method, **params):
        """Call a method on the parent process and wait for result.

        Can only be called during method execution (while transport is active).
        The parent handles the callback and sends back a response.
        Handles interleaved management commands (model_to_device, ping, etc.)
        that may arrive while waiting for the callback_response.
        """
        transport.send({"type": "callback", "method": method, "call_id": _current_call_id, **params})
        while True:
            # recv() raises ConnectionError if the parent went away; None is
            # only ever a timeout, and this call passes none.
            response = transport.recv()
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
                    # Typed field first; a new parent stamps interrupts so the
                    # verdict never has to be recovered from message text.
                    if response.get("error_kind") == "interrupt":
                        raise _InterruptedError(
                            response.get("error", "Processing interrupted by user"))
                    raise RuntimeError(response.get("error", "Callback failed"))
                return response.get("result")
            # Unknown message — log and skip
            wlog(f"[worker] _call_parent: unexpected message type={response.get('type')}, keys={list(response.keys())}")

    # ---------------------------------------------------------------
    # Attention backend: the worker FOLLOWS the host (same rule as aimdo).
    # The mirrored key is the host's RESOLVED backend, not a store_true flag
    # (host-False is indistinguishable from host-default on the wire), and a
    # host that could import sage yet resolved pytorch attention made a
    # deliberate choice the worker must not upgrade past. The old auto-probe
    # guessed over that known answer -- an operator who removed sage because
    # it broke on their card got it back in every worker -- so it now runs
    # only under COMFY_ENV_WORKER_ATTENTION=auto (for pack envs richer than
    # the host). A mirrored backend this env cannot import is skipped and
    # reported, never silently substituted. Still before the first
    # comfy.model_management import below, so comfy.ldm.modules.attention
    # sees the flags when it is first imported.
    # ---------------------------------------------------------------
    try:
        _att_mode = os.environ.get("COMFY_ENV_WORKER_ATTENTION", "").strip().lower()
        if _att_mode == "auto":
            import torch as _torch_check
            if _torch_check.cuda.is_available() and _torch_check.cuda.get_device_capability()[0] >= 8:
                from comfy.cli_args import args as _cli_args
                try:
                    import sageattention  # noqa: F401
                    _cli_args.use_sage_attention = True
                    wlog("[worker] Auto-enabled sage attention (WORKER_ATTENTION=auto)")
                except ImportError:
                    pass
                try:
                    import flash_attn  # noqa: F401
                    _cli_args.use_flash_attention = True
                    wlog("[worker] Auto-enabled flash attention (WORKER_ATTENTION=auto)")
                except ImportError:
                    pass
        else:
            _want_att = (_mirror_payload or {}).get(
                _mirrored_args.ATTENTION_KEY) if _mirrored_args else None
            if _want_att in ("sage", "flash"):
                from comfy.cli_args import args as _cli_args
                try:
                    if _want_att == "sage":
                        import sageattention  # noqa: F401
                        _cli_args.use_sage_attention = True
                    else:
                        import flash_attn  # noqa: F401
                        _cli_args.use_flash_attention = True
                    _mirror_report["applied"].append("attention")
                    wlog(f"[worker] Mirrored host attention backend: {_want_att}")
                except ImportError:
                    _mirror_report["skipped"].append(
                        {"name": "attention",
                         "reason": "unimportable:" + _want_att})
                    wlog(f"[worker] host attention backend '{_want_att}' not "
                         f"importable in this env; using comfy's default")
    except Exception as _ae:
        wlog(f"[worker] attention setup failed: {_ae}")

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
                    # Largest weight(+bias) of any module: what ops.py sizes a
                    # cast buffer to. The parent's need formula books the
                    # buffers this load will allocate lazily at its first
                    # forward, AFTER admission -- bytes neither NVML nor a
                    # measured field can see yet. Best effort; absent means
                    # the parent falls back to min_inference alone.
                    _largest = 0
                    try:
                        for _mod in m.model.modules():
                            _w = getattr(_mod, "weight", None)
                            if _w is None or not hasattr(_w, "nbytes"):
                                continue
                            _n = int(_w.nbytes)
                            _b = getattr(_mod, "bias", None)
                            if _b is not None and hasattr(_b, "nbytes"):
                                _n += int(_b.nbytes)
                            _largest = max(_largest, _n)
                    except Exception:
                        _largest = 0
                    model_info.append({"size": size, "key": str(id(m)),
                                       "largest_tensor": _largest})

                total_size = sum(mi["size"] for mi in model_info)
                wlog(f"[worker] load_models_gpu shim: {len(models)} models, {total_size / 1e9:.2f} GB total")

                # Ask parent to evict its models and make room. The pin state
                # rides along: this is the low-frequency channel the parent's
                # allocator reads, and the reply is the ONLY grant channel
                # (censuses are worker-to-parent piggybacks; no parent push
                # exists at node boundaries).
                _ps = None
                try:
                    if _memmgr is not None:
                        _ps = _memmgr.pin_state()
                except Exception:
                    pass
                # LIVE resolved stream count (never cli_mirror: this is a
                # booking input and must be the value the cast path will
                # actually use, read after import froze it).
                _ns = None
                try:
                    _ns = int(getattr(_cmm, "NUM_STREAMS", 0))
                except Exception:
                    pass
                result = _call_parent("request_vram_budget",
                             model_info=model_info,
                             total_size=total_size,
                             pin_state=_ps,
                             num_streams=_ns)

                # Propagate parent's VRAM constraints to subprocess
                if result:
                    extra_reserved = result.get("extra_reserved_vram")
                    # Pin grant (present only under COMFY_ENV_PIN_SPLIT=auto).
                    # Clamp-only by contract; grow before load is the point of
                    # applying it here, before the real load_models_gpu runs.
                    if _memmgr is not None and (
                            result.get("pin_max") is not None
                            or result.get("pin_headroom") is not None):
                        try:
                            _memmgr.apply_pin_budget(
                                grant=result.get("pin_max"),
                                headroom=result.get("pin_headroom"),
                                log=wlog)
                        except Exception as _pe:
                            wlog(f"[worker] pin grant apply failed: {_pe}")

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
                            # SUM, not max. The blindness term (bytes siblings
                            # hold) and the host margin (extra_reserved_vram)
                            # serve different purposes: one makes our view
                            # honest, the other keeps a buffer ABOVE the honest
                            # view for cast transients and other apps. max()
                            # collapsed them: whenever siblings held more than
                            # the margin, the margin vanished. Note: the margin
                            # does NOT protect dynamic models, whose budget is 0
                            # and rewritten to 1e32 upstream (model_load), so an
                            # unpageable load path can still fill the card.
                            extra_reserved = int(extra_reserved or 0) + _others
                            wlog(f"[worker] blindness correction: my_free="
                                 f"{_my_blind / 1e9:.2f}GB device_free="
                                 f"{_dev_free / 1e9:.2f}GB -> reserve "
                                 f"{_others / 1e9:.2f}GB for others + "
                                 f"{int(result.get('extra_reserved_vram') or 0) / 1e9:.2f}GB margin")
                        except Exception as _be:
                            wlog(f"[worker] blindness correction skipped: {_be}")

                    if extra_reserved is not None:
                        _cmm.EXTRA_RESERVED_VRAM = extra_reserved
                        wlog(f"[worker] margin now {extra_reserved / 1e9:.2f}GB "
                             f"(host {int(result.get('extra_reserved_vram') or 0) / 1e9:.2f}GB "
                             f"+ others; settles any bootstrap advance)")

                    parent_vram_state = result.get("vram_state")
                    if parent_vram_state:
                        try:
                            _cmm.vram_state = _cmm.VRAMState[parent_vram_state]
                            wlog(f"[worker] Set vram_state = {parent_vram_state}")
                        except (KeyError, AttributeError):
                            pass

                # Mark the models this load is about to touch BEFORE the real
                # load runs: pin tier 1 ignores `active`, so pressure fired
                # inside the load can shred the loading model's own staging
                # pins mid-copy unless current_prompt already protects it.
                _prompt_marks_on_load(models)

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
        def _progress_hook(value, total, preview=None, node_id=None):
            try:
                _call_parent("report_progress", value=value, total=total)
            except _InterruptedError:
                raise
            except RuntimeError as e:
                # OLD-PARENT FALLBACK ONLY: parents predating the typed
                # error_kind field send a bare RuntimeError whose text is the
                # only signal. New code paths never match message text.
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

    # Resolve and report the memory manager. A worker never runs main.py, so
    # this is the ledger unless COMFY_ENV_WORKER_AIMDO opted in above.
    _mem_info = {}
    if _memmgr is not None:
        try:
            _memmgr.maybe_enable_aimdo(log=wlog)
            _mem_info = _memmgr.describe()
            wlog(_memmgr.summary_line("[worker] "))
        except Exception as _e:
            wlog(f"[worker] memory manager probe failed: {_e}")

    # Pin budget bootstrap. The counter installs unconditionally (telemetry);
    # the grant and headroom apply only when the parent exported them, which
    # it does only under COMFY_ENV_PIN_SPLIT=auto -- absent vars are a no-op,
    # keeping the off default byte-identical to today. apply_pin_budget is
    # clamp-only, so a mirrored --disable-pinned-memory stays disabled.
    if _memmgr is not None:
        try:
            _memmgr.install_pin_error_counter()
            # Eviction counters (observability, ungated): pin_errors is blind
            # to the churn loop, which unregisters and re-registers pins
            # successfully; these count what actually moved, and
            # pins_evicted_active_bytes is the prompt-mark fix's own
            # regression signal (must stay 0).
            _memmgr.install_pin_eviction_counters(log=wlog)
            _bs_grant = os.environ.get("COMFY_ENV_PIN_SHARE")
            _bs_headroom = os.environ.get("COMFY_ENV_PIN_HEADROOM")
            if _bs_grant is not None or _bs_headroom is not None:
                _memmgr.apply_pin_budget(
                    grant=int(_bs_grant) if _bs_grant is not None else None,
                    headroom=int(_bs_headroom) if _bs_headroom is not None else None,
                    log=wlog)
        except Exception as _e:
            wlog(f"[worker] pin budget bootstrap failed: {_e}")
        # Reserve bootstrap: the budget owner's advance payment. Parses
        # inside apply_reserve_bootstrap (a garbage value WARNs there instead
        # of killing this whole block); absent var is a no-op byte-identical
        # to today; the first budget reply's plain assignment supersedes it.
        try:
            _memmgr.apply_reserve_bootstrap(
                os.environ.get("COMFY_ENV_EXTRA_RESERVED_VRAM"), log=wlog)
        except Exception as _e:
            wlog(f"[worker] reserve bootstrap failed: {_e}")


    def _residency_census():
        """Worker-measured residency for every registered model. The same
        number _handle_model_partial trusts, sampled at the node boundary.
        Never raises: the census is advisory and a failed read keeps the
        parent's last receipt (missing means unknown, not zero)."""
        out = []
        for _mid, _model in list(_model_registry.items()):
            try:
                _lm = _find_loaded_model(_model)
                if _lm is not None:
                    _res = int(_lm.model.loaded_size())
                else:
                    _prm = next(_model.parameters(), None)
                    _on = _prm is not None and _prm.device.type == "cuda"
                    _res = int(_model_registry_meta.get(_mid, {}).get("size", 0)) if _on else 0
                _dev = "cpu"
                try:
                    _prm = next(_model.parameters(), None)
                    if _prm is not None:
                        _dev = str(_prm.device)
                except Exception:
                    pass
                out.append({"id": _mid, "seq": _residency_seq.get(_mid, 0),
                            "resident": _res, "device": _dev})
            except Exception:
                continue
        return out

    def _attach_new_models(resp):
        """Attach auto-detected models to ANY outgoing frame, ok or error.

        A node can move a 10GB model to CUDA and THEN raise (an OOM does exactly
        that). The model is resident either way, and _new_models_this_call is
        cleared on the next request, so a frame that omits it makes that VRAM
        invisible to the host for the life of the worker. The parent harvests
        _new_models on every response path before any status check, so
        attaching here is sufficient.
        """
        # Census FIRST, above the early return: with no new models this call
        # (the steady state) the early-out would suppress residency in exactly
        # the case where the parent's stamp is going stale.
        _census_list = None
        try:
            if _model_registry:
                _census_list = _residency_census()
                resp["_model_residency"] = _census_list
        except Exception:
            pass
        # Measured VRAM overhead: allocator bytes beyond registered residency
        # (cast buffers, allocator cache, slack). Computed from the census
        # list JUST BUILT, same instant, so reserved and model bytes cannot
        # disagree across frames (the mixed-frame double count). NOT gated on
        # _model_registry: an empty-registry worker holding leaked scratch
        # must still report. Absent on failure, never a fabricated zero.
        try:
            import torch as _ovt
            if _ovt.cuda.is_initialized():
                _resident_sum = sum(int(e.get("resident", 0))
                                    for e in (_census_list or []))
                resp["_vram_overhead"] = max(
                    0, int(_ovt.cuda.memory_reserved()) - _resident_sum)
        except Exception:
            pass
        # Pin census: one bare int (hot frames stay small; the five-field
        # _pin_state rides the low-frequency channels). None means comfy is
        # not imported here -- report nothing, never a fabricated zero.
        try:
            if _memmgr is not None:
                _tp = _memmgr.total_pinned()
                if _tp is not None:
                    resp["_pinned"] = _tp
        except Exception:
            pass
        if _pending_state_out[0] is not None:
            resp["_self_state_out"] = _pending_state_out[0]
            _pending_state_out[0] = None
        if not _new_models_this_call:
            return resp
        # Resolve actual device at response time.  Models are auto-detected
        # when they land on CUDA, but the subprocess may have moved them back
        # to CPU before the call finished (or the raise interrupted a move).
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
        resp["_new_models"] = list(_new_models_this_call)
        return resp

    # Signal ready. full_release advertises the deep-release handler: the
    # parent broadcasts /free only to advertisers (an unknown method gets no
    # reply and the sender would eat the 60 s recv timeout).
    _ready_frame = {"status": "ready", "memory_manager": _mem_info,
                    "full_release": True, "release_pins": True}
    try:
        if _memmgr is not None:
            _ps = _memmgr.pin_state()
            if _ps is not None:
                _ready_frame["_pin_state"] = _ps
    except Exception:
        pass
    # Mirror divergence report: the hash is READ BACK off this process's args
    # object after apply (never the received payload, which would compare a
    # copy to itself and detect nothing). num_streams is read back from the
    # frozen module constant, never recomputed -- it is Group A's budget
    # input for cast-buffer VRAM.
    try:
        if _mirrored_args is not None and _mirror_payload is not None:
            from comfy.cli_args import args as _ra_args
            _applied_names = [n for n in _mirror_report.get("applied", [])
                              if n != "attention"]
            _cm = {
                "hash": _mirrored_args.readback_hash(_ra_args, _applied_names),
                "applied": _mirror_report.get("applied", []),
                "skipped": _mirror_report.get("skipped", []),
            }
            _cmm_ready = sys.modules.get("comfy.model_management")
            if _cmm_ready is not None:
                _ns = getattr(_cmm_ready, "NUM_STREAMS", None)
                if _ns is not None:
                    _cm["num_streams"] = int(_ns)
            _ready_frame["cli_mirror"] = _cm
    except Exception as _cme:
        wlog(f"[worker] mirror report failed: {_cme}")
    transport.send(_ready_frame)
    wlog("[worker] Ready")

    # --- Pool IPC handshake: create shareable pool and send FD to parent ---
    global _pool_ipc_ok, _our_pool
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

        # Deep release for the host's /free. MAIN LOOP ONLY, never dispatched
        # from _call_parent's interleave: the worker is idle between requests
        # here, so gc and empty_cache cannot fire under an active forward.
        # Reply rides _attach_new_models so the census and _pinned converge
        # the parent's ledgers at release time (a released worker may go
        # quiet, so the parent must not wait for a next call).
        if request.get("method") == "full_release":
            _fr = {"status": "ok", "call_id": request.get("call_id")}
            try:
                if _memmgr is not None:
                    _fr["receipt"] = _memmgr.full_release(log=wlog)
                else:
                    _fr["receipt"] = {"steps": [], "errors": ["no memory_manager"]}
            except Exception as _fre:
                _fr["receipt"] = {"steps": [], "errors": [str(_fre)]}
            transport.send(_attach_new_models(_fr))
            continue

        # Host RAM-pressure pin reclaim: release N pinned bytes through the
        # worker's OWN free_pins ladder (tiers, hysteresis, prompt marks all
        # apply, same as a non-isolated node's pins). Main loop only, like
        # full_release.
        if request.get("method") == "release_pins":
            _pr = {"status": "ok", "call_id": request.get("call_id")}
            try:
                if _memmgr is not None:
                    _pr["receipt"] = _memmgr.release_pins(
                        int(request.get("size", 0)), log=wlog)
                else:
                    _pr["receipt"] = {"errors": ["no memory_manager"]}
            except Exception as _pre:
                _pr["receipt"] = {"errors": [str(_pre)]}
            transport.send(_attach_new_models(_pr))
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
        _pending_state_out[0] = None

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
                _frame = {"status": "error", "call_id": _current_call_id,
                          "error": str(e),
                          "traceback": traceback.format_exc()}
                _frame.update(_error_kind_fields(e))
                transport.send(_attach_new_models(_frame))
            continue

        if "module" not in request:
            wlog(f"[worker] Ignoring unknown request format: {list(request.keys())}")
            continue

        shm_registry = []
        try:
            request_type = request.get("type", "call_module")
            module_name = request["module"]
            wlog(f"[worker] Request: {request_type} {module_name} call_id={_current_call_id}")
            # Retire the previous prompt's pin marks before this call's
            # method (and therefore before any of its loads) runs.
            _prompt_marks_preamble(request)
            # New prompt starts with clean cast buffers (the non-aimdo
            # ratchet fix); same epoch is a no-op, so intra-prompt reuse
            # never reallocs.
            if _memmgr is not None:
                _memmgr.cast_epoch_boundary(request.get("prompt_gen"),
                                            log=wlog)

            # Load inputs from shared memory
            kwargs_meta = request.get("kwargs")
            if kwargs_meta:
                wlog(f"[worker] Reconstructing inputs from shm...")
                inputs = _from_shm(kwargs_meta)
                inputs = _deserialize_isolated_objects(inputs)
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
                # no_grad, NOT inference_mode. Only the node call is wrapped, so a
                # model that lazily creates an nn.Parameter on its first forward
                # creates it INSIDE this context -- and inference_mode stamps it
                # `is_inference`, permanently. Anything that later touches that
                # parameter from OUTSIDE the context then raises: an autograd-
                # tracked op with "Inference tensors cannot be saved for backward",
                # an in-place update with "Inplace update to inference tensor
                # outside InferenceMode is not allowed". For us that outside
                # toucher is routine -- SubprocessModelPatcher moving the model
                # between devices, a LoRA weight patch, load_state_dict.
                # Measured cost of the safer context: +0.4% on a 16-layer forward.
                # See tests/test_infer_mode.py; upstream pytorch#90882.
                _infer_mode = _torch_worker.no_grad
            except ImportError:
                import contextlib as _contextlib_worker
                _infer_mode = _contextlib_worker.nullcontext
            if request_type == "call_method":
                class_name = request["class_name"]
                method_name = request["method_name"]
                self_state = request.get("self_state")
                _state_sync_on = (
                    _state_sync is not None
                    and self_state is not None
                    and os.environ.get(_state_sync.STATE_ENV_VAR, "sync").lower()
                    not in ("off", "0", "false")
                )
                wlog(f"[worker] Getting class {class_name}...")

                cls = getattr(module, class_name)
                wlog(f"[worker] Creating instance...")
                if _state_sync_on and request.get("seed"):
                    # Run the REAL __init__ once per parent instance: upstream
                    # parity is class_def() (execution.py:499). Today no
                    # __init__ ever runs anywhere, so a node reading an
                    # __init__-set attribute raises. Falls back to the old
                    # path with a WARN rather than breaking the node.
                    try:
                        instance = cls()
                    except Exception as _se:
                        wlog(f"[worker] WARNING: {class_name}() raised during "
                             f"seeding ({_se}); falling back to object.__new__")
                        instance = object.__new__(cls)
                else:
                    instance = object.__new__(cls)
                if self_state:
                    self_state = _deserialize_isolated_objects(self_state)
                    if _state_sync_on:
                        # resolve overflow markers back into live values; a
                        # marker from a previous worker generation is state
                        # this process never held, and pretending otherwise is
                        # the silent-wrong-default this design refuses.
                        for _k in list(self_state.keys()):
                            _v = self_state[_k]
                            if _state_sync.is_overflow_marker(_v):
                                if _v.get("gen") != _STATE_GEN:
                                    raise RuntimeError(
                                        f"node attribute '{_k}' was held in a "
                                        f"worker that has restarted; its value "
                                        f"is gone and will be recomputed. "
                                        f"(gen {_v.get('gen')} != {_STATE_GEN})"
                                    )
                                _held = _overflow_store.get(int(_v.get("handle", -1)))
                                if _held is None:
                                    raise RuntimeError(
                                        f"node attribute '{_k}' overflow handle "
                                        f"{_v.get('handle')} is unknown (evicted?)"
                                    )
                                self_state[_k] = _held[1]
                    instance.__dict__.update(self_state)
                # The diff baseline is what the PARENT sent, not the post-seed
                # instance dict: on a seeding call, __init__-set values are new
                # to the parent and must ship, which they cannot if they count
                # as "already known".
                _pre_state = (dict(self_state) if self_state else {}) \
                    if _state_sync_on else None
                _state_owner = (request.get("state_id") or "?") if _state_sync_on else None
                wlog(f"[worker] Calling {method_name}...")
                method = getattr(instance, method_name)
                try:
                    with _infer_mode():
                        result = method(**inputs)
                finally:
                    # State diff FIRST, in the finally: a non-isolated node
                    # that mutates self and then raises keeps the mutation, so
                    # an isolated one must too. Never raises: losing the state
                    # return must not mask the node's own outcome.
                    if _state_sync_on and _pre_state is not None:
                        try:
                            _cap = int(os.environ.get(
                                _state_sync.STATE_MAX_BYTES_ENV_VAR,
                                _state_sync.STATE_MAX_BYTES_DEFAULT))
                            _inbound_handles = {
                                int(v.get("handle", -1))
                                for v in _pre_state.values()
                                if _state_sync.is_overflow_marker(v)
                            }
                            _minted = set()

                            def _store(h, v):
                                _minted.add(h)
                                _overflow_store[h] = (_state_owner, v)

                            _pending_state_out[0] = _state_sync.diff_state(
                                _pre_state, dict(instance.__dict__), _cap,
                                _STATE_GEN, _mint_handle, _store)
                            # owner-scoped reap: a handle of THIS instance that
                            # the parent no longer references (and that this
                            # call did not just mint) has been deleted or swept
                            # parent-side; holding it would leak until restart.
                            for _h in list(_overflow_store.keys()):
                                _own, _ = _overflow_store[_h]
                                if (_own == _state_owner
                                        and _h not in _inbound_handles
                                        and _h not in _minted):
                                    del _overflow_store[_h]
                        except Exception as _de:
                            wlog(f"[worker] state diff failed: {_de}")
                    # ComfyUI runs this in a finally around every node
                    # (execution.py:550). A worker never reaches that code, so
                    # without this an aimdo-enabled worker would allocate cast
                    # buffers and CUDA graph pools with nothing to free them.
                    # No-op unless aimdo is actually live in this process.
                    if _memmgr is not None:
                        _memmgr.release_node_boundary(log=wlog)
                    _prompt_marks_sweep()
                wlog(f"[worker] Method returned")
            else:
                func_name = request["func"]
                func = getattr(module, func_name)
                try:
                    with _infer_mode():
                        result = func(**inputs)
                finally:
                    if _memmgr is not None:
                        _memmgr.release_node_boundary(log=wlog)
                    _prompt_marks_sweep()

            # Serialize result to shared memory
            wlog(f"[worker] Serializing result to shm...")
            _serializing_call_id = _current_call_id
            result_meta = _to_shm(result, shm_registry)
            wlog(f"[worker] Created {len(shm_registry)} shm blocks for result")

            response = {"status": "ok", "call_id": _current_call_id, "result": result_meta}
            transport.send(_attach_new_models(response))
            # Kept until the parent's "consumed" ack (TTL is the fallback)
            _shm_keeper.keep(shm_registry, _current_call_id)

        except Exception as e:
            # Cleanup shm on error since host won't read it
            _cleanup_shm(shm_registry)
            _frame = {
                "status": "error",
                "call_id": _current_call_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            _frame.update(_error_kind_fields(e))
            transport.send(_attach_new_models(_frame))

    transport.close()

if __name__ == "__main__":
    main()
