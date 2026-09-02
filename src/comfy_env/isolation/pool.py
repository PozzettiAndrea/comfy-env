"""Persistent worker pool -- one worker per isolation env, reused across calls.

Extracted from wrap.py so `metadata.py` (proxy synthesis) can import the pool
DOWNWARD instead of reaching UP into the orchestrator -- the wrap<->metadata
cycle. Owns the pool state, worker lifecycle (create/restart/shutdown), the
VRAM-budget and progress callbacks, the stale-patcher invariant (ADR-0019),
and API route proxying. Depends downward on workers/subprocess and
model_patcher (both lazy); imports nothing from wrap or metadata.
"""

import atexit
import glob
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_HEALTH_CHECK_TIMEOUT
from .. import state_sync
from ..debug import WORKER as _DBG_WORKER, MODELS as _DBG_MODELS, log as _log


_CLEANUP_DONE = False

# ---------------------------------------------------------------------------
# Persistent worker pool -- one worker per isolation env, reused across calls.
# Workers auto-restart on crash (native segfault, etc.).
# ---------------------------------------------------------------------------
_WORKER_POOL: Dict[str, Any] = {}  # str(env_dir) -> (SubprocessWorker, generation)
_WORKER_PATCHERS: Dict[str, Dict[str, Any]] = {}  # str(env_dir) -> {model_id: SubprocessModelPatcher}
_STALE_PATCHERS: List[Any] = []  # Keeps stale patchers alive until free_memory finishes
_POOL_LOCK = threading.Lock()
_WORKER_GENERATION = 0  # Monotonically increasing; incremented on each new worker

# --- Pin budget ledger (COMFY_ENV_PIN_SPLIT) ------------------------------
# reports: "host" plus str(env_dir) -> {"pinned": bytes, "seq": n}. seq is a
# PARENT-side arrival stamp (one socket per worker makes arrival order causal
# order); the worker's own _pin_state.seq is observability, not the ledger.
_PIN_REPORTS: Dict[str, Dict[str, int]] = {}
_PIN_GRANTS: Dict[str, int] = {}  # last grant emitted per worker key (damping)
_PIN_INGEST_SEQ = 0
_PIN_STABLE = 0  # consecutive censuses with an unchanged consumer set
_PIN_LAST_CONSUMERS: frozenset = frozenset()
_PIN_ROLLUP_LAST = 0  # last logged pinned total (rollup fires on >1 GiB moves)


#: Measured per-worker allocator excess (reserved minus census residency),
#: keyed like _WORKER_PATCHERS; REPLACE on newer parent-side arrival seq.
_OVERHEAD_REPORTS: Dict[str, Dict[str, int]] = {}
_OVERHEAD_SEQ = 0

_DEVICE_TOTAL_CACHE: List[Optional[int]] = []  # [] = unprobed, [None] = unknowable


def _device_total_bytes() -> Optional[int]:
    """Total VRAM of the torch device, probed once (it is static). Used only
    to clamp absurd overhead reports; None (no NVML) means uncapped, and the
    ingest WARN covers the junk-report case there."""
    if not _DEVICE_TOTAL_CACHE:
        total = None
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
                idx = device.index if getattr(device, "index", None) is not None else 0
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                total = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            total = None
        _DEVICE_TOTAL_CACHE.append(total)
    return _DEVICE_TOTAL_CACHE[0]


def _pin_split_mode() -> str:
    """"off" (default, byte-identical to today) or "auto"."""
    return os.environ.get(state_sync.PIN_SPLIT_ENV_VAR, "off").strip().lower()


def _pin_ingest(key: str, pinned) -> None:
    """Stamp one pin report into the ledger and track consumer-set stability
    for the grow damping. Never raises."""
    global _PIN_INGEST_SEQ, _PIN_STABLE, _PIN_LAST_CONSUMERS, _PIN_ROLLUP_LAST
    try:
        _PIN_INGEST_SEQ += 1
        state_sync.update_pin_reports(_PIN_REPORTS, key, int(pinned),
                                      _PIN_INGEST_SEQ)
        consumers = frozenset(k for k, r in _PIN_REPORTS.items()
                              if r.get("pinned", 0) > 0)
        if consumers == _PIN_LAST_CONSUMERS:
            _PIN_STABLE += 1
        else:
            _PIN_STABLE = 0
            _PIN_LAST_CONSUMERS = consumers
        total = sum(r.get("pinned", 0) for r in _PIN_REPORTS.values())
        if abs(total - _PIN_ROLLUP_LAST) > 1024 ** 3:
            _PIN_ROLLUP_LAST = total
            _log("[comfy-env] pinned RAM rollup: "
                 + ", ".join(f"{k}={r.get('pinned', 0) / 1e9:.2f}GB"
                             for k, r in sorted(_PIN_REPORTS.items()))
                 + f" (sum {total / 1e9:.2f}GB)")
    except Exception:
        pass


def _cleanup_stale_workers():
    """Kill worker processes and remove socket/temp litter left by a DEAD
    ComfyUI, on startup.

    Everything here is guarded by "is the owning process still alive?", so a
    second ComfyUI running on the same machine is never touched. psutil is
    available unconditionally -- ComfyUI itself depends on it
    (requirements.txt, and comfy/model_management.py imports it).
    """
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True

    import psutil

    temp_dir = tempfile.gettempdir()

    # Sockets. A unix socket file is unlinked only on CLEAN shutdown
    # (workers/subprocess.py, _shutdown), so a live instance's socket sits on
    # disk for its entire session and is indistinguishable by name from one a
    # crashed instance abandoned. The owning pid is in the filename for
    # exactly this reason. Deleting a live sibling's socket does not break
    # connections already established, but a worker that has not dialed in yet
    # can no longer reach it.
    # Linux binds in the ABSTRACT namespace (no filesystem entry), so this
    # only ever finds anything on macOS/Windows.
    sock_owner = re.compile(r"^comfy_worker_(\d+)_[0-9a-f]+\.sock$")
    socket_patterns = [
        "/dev/shm/comfy_worker_*.sock",
        os.path.join(temp_dir, "comfy_worker_*.sock"),
    ]
    for pattern in socket_patterns:
        for sock_file in glob.glob(pattern):
            m = sock_owner.match(os.path.basename(sock_file))
            # No pid in the name: written by a comfy-env older than this one.
            # Pre-1.0 ships as a barrage, so a sibling that old is not a case
            # we carry -- treat it as litter, which is the old behavior.
            if m and psutil.pid_exists(int(m.group(1))):
                continue
            try:
                os.unlink(sock_file)
                print(f"[comfy-env] Removed stale socket: {sock_file}")
            except Exception:
                pass

    # Worker processes whose parent is gone.
    for proc in psutil.process_iter(['pid', 'ppid', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('persistent_worker.py' in arg for arg in cmdline):
                parent_pid = proc.info.get('ppid')
                if parent_pid and not psutil.pid_exists(parent_pid):
                    print(f"[comfy-env] Killing orphaned worker (parent {parent_pid} dead): {proc.pid}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Temp dirs no live process is sitting in.
    for stale_dir in glob.glob(os.path.join(temp_dir, "comfyui_pvenv_*")):
        try:
            dir_in_use = False
            for proc in psutil.process_iter(['pid', 'cmdline', 'cwd']):
                try:
                    cwd = proc.info.get('cwd') or ''
                    cmdline = ' '.join(proc.info.get('cmdline') or [])
                    if stale_dir in cwd or stale_dir in cmdline:
                        dir_in_use = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if not dir_in_use:
                shutil.rmtree(stale_dir)
                print(f"[comfy-env] Removed stale temp dir: {stale_dir}")
        except Exception:
            pass


def _create_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                   env_vars: Optional[dict] = None,
                   health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
    """Create a fresh subprocess worker."""
    python = env_dir / ("python.exe" if sys.platform == "win32" else "bin/python")
    from .workers.subprocess import SubprocessWorker
    if _DBG_WORKER:
        print(f"[comfy-env] SubprocessWorker: {python}")
        if env_vars:
            print(f"[comfy-env] env_vars: {env_vars}")
    return SubprocessWorker(
        python=str(python), working_dir=working_dir, sys_path=sys_path,
        name=working_dir.name, env=env_vars, health_check_timeout=health_check_timeout
    )


def _handle_progress(request: dict) -> dict:
    """Parent-side callback: forward subprocess progress to ComfyUI frontend.

    Raises on cancel so the worker learns about it. Two reasons this is not
    a `return {"status": "error"}`:

    * InterruptProcessingException derives from BaseException, so the old
      `except Exception` never caught it. It unwound out of _send_request's
      read loop MID-CONVERSATION while the worker sat blocked in
      _call_parent's recv() awaiting a callback_response that never came --
      and that loop then ate the next real request as an unexpected frame.
    * _handle_callback wraps any RETURN as {"status": "ok", "result": ...},
      and the worker only inspects the outer status, so an error dict was
      indistinguishable from success. Raising lets _handle_callback's
      `except Exception` produce a genuine error frame, which _call_parent
      turns into _InterruptedError inside the worker.

    The message must keep containing "interrupted": old workers' _progress_hook
    text-matches on it (new workers read the typed error_kind field instead).
    """
    from .workers.base import InterruptRequested
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None  # not running inside ComfyUI -- nothing to cancel
    if mm is not None:
        try:
            mm.throw_exception_if_processing_interrupted()
        except mm.InterruptProcessingException:
            raise InterruptRequested("Processing interrupted by user")
    try:
        import comfy.utils
        if comfy.utils.PROGRESS_BAR_HOOK:
            value = request.get("value", 0)
            total = request.get("total", 1)
            comfy.utils.PROGRESS_BAR_HOOK(value, total, None)
    except Exception:
        pass
    return {}


#: Fixed VRAM cost of an extra CUDA-using process that ComfyUI's ledger never
#: sees: CUDA context + cuBLAS/cuDNN handles, all OUTSIDE the caching
#: allocator (torch.cuda.memory_reserved structurally cannot see them, which
#: is why the measured _vram_overhead excess ADDS to this floor instead of
#: maxing with it). 250 MB was measured on a Windows RTX 4060 Ti; a Linux
#: RTX 3090 measured 276 to 300 MB (2026-09), so the floor rose to cover it.
#: Additive per live worker -- unlike the model-size headroom, which is
#: multiplicative.
_WORKER_FIXED_VRAM_COST = state_sync.WORKER_VRAM_FLOOR

#: Multiplicative slack on the requested model bytes (allocator rounding).
#: Small because cudaMallocAsync (the default backend) keeps slack near 1%;
#: the dominant hidden cost is the per-process constant above.
_REQUEST_SLACK = 1.02


def _true_device_free(device) -> "int | None":
    """Device-wide free VRAM, across processes. None if unobtainable.

    `torch.cuda.mem_get_info` -- which ComfyUI's get_free_memory relies on --
    reports the CALLING PROCESS's budget on Windows/WDDM, not the device total.
    Measured: a sibling process allocated 13.0 GiB; nvidia-smi free fell
    13,443 MB while the parent's mem_get_info fell 75 MB. Every admission
    decision made from that number is fiction on the majority platform.

    Ladder: pynvml -> nvidia-smi -> None (caller falls back to its own ledger).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            idx = device.index if getattr(device, "index", None) is not None else 0
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            return int(pynvml.nvmlDeviceGetMemoryInfo(h).free)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    except Exception:
        pass
    try:
        import subprocess as _sp
        idx = device.index if getattr(device, "index", None) is not None else 0
        out = _sp.run(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3)
        if out.returncode == 0 and out.stdout.strip():
            return int(float(out.stdout.strip().splitlines()[0])) * 1024 * 1024
    except Exception:
        pass
    return None


def _worker_held_bytes() -> int:
    """Bytes this process's workers hold on the GPU, from comfy-env's own books.

    Zero-dependency fallback for `_true_device_free`: comfy-env already knows
    every worker model's size and residency, so it can reconstruct what
    ComfyUI's view is missing without NVML. Undercounts allocations the
    Module.to()/.cuda() hooks never saw (ADR-0025 records that gap).
    """
    # Snapshot: _register_new_patchers and _cleanup_stale_patchers mutate
    # these from the aiohttp executor thread, and _cleanup_stale_patchers runs
    # outside _POOL_LOCK. Not taking the lock here on purpose -- it is a plain
    # Lock held across verify_transport(), so re-entering would deadlock.
    # The arithmetic itself is pure (state_sync.held_from_snapshot): per
    # worker, each patcher charges size while a call is IN FLIGHT (unsignaled
    # lazy re-faults can never exceed the supremum) and max(ledger, peak)
    # while idle (an idle worker cannot re-fault; its receipts are
    # authoritative), plus the context floor, plus the measured allocator
    # excess. Every live pool entry books the floor, patchers or not (the old
    # `if patchers:` skip booked a modelless worker's CUDA context at zero).
    snapshot: Dict[str, Dict[str, Any]] = {}
    keys = set(_WORKER_POOL) | {k for k, v in _WORKER_PATCHERS.items() if v}
    for key in keys:
        entry = _WORKER_POOL.get(key)
        worker = entry[0] if entry else None
        models = []
        for p in list(_WORKER_PATCHERS.get(key, {}).values()):
            try:
                models.append({
                    "ledger": int(getattr(p.model, "model_loaded_weight_memory", 0)),
                    "peak": int(getattr(p, "_residency_peak", 0)),
                    "size": int(getattr(p, "size", 0)),
                })
            except Exception:
                try:
                    models.append({"ledger": int(p.loaded_size()),
                                   "peak": 0, "size": 0})
                except Exception:
                    pass
        snapshot[key] = {
            "in_flight": getattr(worker, "_calls_in_flight", 0) > 0,
            "excess": _OVERHEAD_REPORTS.get(key, {}).get("excess"),
            "models": models,
        }
    return state_sync.held_from_snapshot(snapshot,
                                         floor=_WORKER_FIXED_VRAM_COST,
                                         cap=_device_total_bytes())


def _maybe_add_pin_grant(reply: dict, request: dict, worker_key) -> None:
    """Attach ``pin_max``/``pin_headroom`` to a budget reply.

    The reply is the ONLY grant channel (a debate verdict: censuses are
    worker-to-parent piggybacks; no parent push exists at node boundaries).
    Ingestion of the rider ``pin_state`` always happens (observability);
    the grant fields appear only under ``COMFY_ENV_PIN_SPLIT=auto``, so the
    shipped default is byte-identical to today. Never raises."""
    try:
        ps = request.get("pin_state")
        if isinstance(ps, dict) and worker_key:
            _pin_ingest(worker_key, ps.get("total_pinned", 0))
        if _pin_split_mode() != "auto" or not worker_key:
            return
        import comfy.model_management as mm
        host_max = int(getattr(mm, "MAX_PINNED_MEMORY", 0) or 0)
        _pin_ingest("host", int(getattr(mm, "TOTAL_PINNED_MEMORY", 0) or 0))
        if worker_key not in _PIN_REPORTS:
            _pin_ingest(worker_key, 0)  # first contact: request IS the report
        floor = int(os.environ.get(state_sync.PIN_FLOOR_ENV_VAR,
                                   state_sync.PIN_FLOOR_DEFAULT))
        reserve = float(os.environ.get(state_sync.PIN_RESERVE_ENV_VAR,
                                       state_sync.PIN_RESERVE_DEFAULT))
        grants = state_sync.allocate_pin_budgets(
            host_max, _PIN_REPORTS, floor_bytes=floor, reserve=reserve,
            requester=worker_key)
        raw = grants.get(worker_key)
        if raw is None:
            return
        damped = state_sync.damp_pin_grant(_PIN_GRANTS.get(worker_key),
                                           int(raw), _PIN_STABLE)
        _PIN_GRANTS[worker_key] = damped
        reply["pin_max"] = int(damped)
        try:
            import comfy.memory_management as cmm
            reply["pin_headroom"] = int(getattr(cmm, "RAM_CACHE_HEADROOM", 0))
        except Exception:
            pass
        if _DBG_MODELS:
            _log(f"[comfy-env] pin grant for {Path(worker_key).name}: "
                 f"{reply['pin_max'] / 1e9:.2f}GB "
                 f"(host_max {host_max / 1e9:.2f}GB, stable {_PIN_STABLE})")
    except Exception:
        pass


def _handle_vram_budget(request: dict, worker_key=None) -> dict:
    """Parent-side callback: free VRAM for subprocess model loading.

    Called when the worker's shimmed load_models_gpu() needs to load models.

    The parent cannot simply ask ComfyUI to free N bytes: ComfyUI decides how
    much to evict from `memory_required - get_free_memory(device)`, and on
    WDDM that free number cannot see worker memory at all -- it stays near
    full-card, the difference goes negative, and `free_memory` evicts NOTHING.
    So we PRE-COMPENSATE: add the parent's over-report to the target we pass.

    The compensation is exact rather than a fudge: the offset is worker-held
    memory, which is constant across the eviction loop, and every parent-side
    unload moves the blind number and the true number by the same amount. So
    ComfyUI's internal comparison evaluates as if it could see the whole
    device, and the loop still self-terminates at the minimum eviction.
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        return {"device": "cuda"}

    total_requested = request.get("total_size", 0)
    device = mm.get_torch_device()

    blind_free = mm.get_free_memory(device)
    true_free = _true_device_free(device)
    if true_free is None:
        # No NVML/nvidia-smi: reconstruct from comfy-env's own ledger.
        true_free = max(0, blind_free - _worker_held_bytes())
        offset_source = "ledger"
    else:
        offset_source = "nvml"
    offset = max(0, blind_free - true_free)

    # Headroom shaped to the real costs: multiplicative slack on the weights,
    # plus the per-process constant, plus the inference reserve ComfyUI would
    # have applied to an equivalent in-process load (mm.free_memory's callers
    # add it; our callback previously did not, so worker loads got ~1GB less
    # headroom than identical host loads).
    try:
        min_inference = mm.minimum_inference_memory()
    except Exception:
        min_inference = 0
    # The requester's booking: its context floor plus its measured allocator
    # excess (an old worker sends no report and books the floor, today's
    # behavior). The forward term books the CAST BUFFERS the incoming load
    # will allocate lazily at its first forward, AFTER this admission --
    # bytes neither NVML nor any measured field can see yet, computed from
    # the load's own largest tensor times the worker's live stream count.
    # max(), never sum: cast buffers are inference transients competing for
    # the same reserve min_inference already books.
    requester_key = str(worker_key) if worker_key else None
    requester_excess = 0
    if requester_key and requester_key in _OVERHEAD_REPORTS:
        requester_excess = int(_OVERHEAD_REPORTS[requester_key].get("excess", 0))
    largest = 0
    for _mi in request.get("model_info") or []:
        try:
            largest = max(largest, int(_mi.get("largest_tensor") or 0))
        except Exception:
            pass
    forward = state_sync.forward_cast_need(largest, request.get("num_streams"))
    need = (int(total_requested * _REQUEST_SLACK)
            + _WORKER_FIXED_VRAM_COST + requester_excess
            + max(min_inference, forward))

    if _DBG_MODELS:
        _inflight = sum(1 for _e in _WORKER_POOL.values()
                        if getattr(_e[0], "_calls_in_flight", 0) > 0)
        _log(f"[comfy-env] VRAM request: {total_requested / 1e9:.2f}GB | "
             f"free: blind={blind_free / 1e9:.2f}GB true={true_free / 1e9:.2f}GB "
             f"offset={offset / 1e9:.2f}GB ({offset_source}, "
             f"{_inflight} in-flight worker(s) charge size) | "
             f"need={need / 1e9:.2f}GB (forward={forward / 1e9:.2f}GB, "
             f"excess={requester_excess / 1e9:.2f}GB) -> asking free_memory for "
             f"{(need + offset) / 1e9:.2f}GB")

    # Offset-compensated target: makes ComfyUI's own arithmetic behave as if
    # get_free_memory were device-wide.
    mm.free_memory(need + offset, device)

    if _DBG_MODELS:
        _log(f"[comfy-env] VRAM after eviction: "
             f"blind={mm.get_free_memory(device) / 1e9:.2f}GB")

    # vram_state/extra_reserved pass through as ComfyUI computed them; the
    # negotiation below is the only mechanism that adjusts them.
    vram_state_name = mm.vram_state.name
    extra_reserved = mm.EXTRA_RESERVED_VRAM

    # Re-measure after eviction: the worker corrects its own blind view from
    # this (its get_free_memory - device_free = what everyone else holds).
    post_true_free = _true_device_free(device)
    if post_true_free is None:
        post_true_free = max(0, mm.get_free_memory(device) - _worker_held_bytes())

    reply = {
        "device": str(device),
        "extra_reserved_vram": extra_reserved,
        "vram_state": vram_state_name,
        "device_free_bytes": int(post_true_free),
    }
    _maybe_add_pin_grant(reply, request, worker_key)
    return reply


def _cleanup_stale_patchers(env_dir):
    """Mark stale SubprocessModelPatchers for cleanup.

    Called when a worker is replaced (crash/restart).  We clear the patcher
    registry so they won't be re-registered.  The patchers themselves stay in
    ComfyUI's current_loaded_models -- the safety net in _send_device_command
    handles "not registered" IPC errors gracefully, and free_memory will
    remove them during its normal unload loop.

    We must NOT modify current_loaded_models here because this callback can
    fire inside free_memory's iteration (via model_unload -> send_command ->
    _ensure_started -> _on_restart), which would invalidate captured indices.

    We also must keep the old patchers alive (in _STALE_PATCHERS) because
    LoadedModel._model is a weakref -- if the patcher is GC'd, the
    SubprocessModel finalizer fires cleanup_models() which pops items from
    current_loaded_models, corrupting free_memory's index-based iteration.
    The stale references are cleared on the next _register_new_patchers call.
    """
    key = str(env_dir)
    _OVERHEAD_REPORTS.pop(key, None)  # the replaced process's scratch is gone
    old_patchers = _WORKER_PATCHERS.pop(key, None)
    if not old_patchers:
        return
    # Keep strong references to prevent GC during free_memory iteration
    _STALE_PATCHERS.extend(old_patchers.values())
    _log(f"[comfy-env] Invalidated {len(old_patchers)} stale model patchers "
         f"(will be cleaned up during next unload)")


def _register_proxy_routes(routes, env_dir, package_root, sys_path, env_vars,
                           health_check_timeout):
    """Register aiohttp routes in the main process that forward to the isolation worker.

    Nodes in isolation environments can declare API routes via a module-level
    ``ROUTES`` list.  Since the isolation subprocess has no access to the ComfyUI
    HTTP server, this function registers proxy handlers in the main process that
    forward JSON requests to the worker via IPC (``call_module``).

    ROUTES convention::

        ROUTES = [
            {"method": "POST", "path": "/my/endpoint", "handler": "my_handler_func"},
        ]

        def my_handler_func(body: dict) -> dict:
            # Runs in the isolation subprocess.
            # Return {"_status": 400, "error": "..."} for non-200 responses.
            return {"result": "ok"}
    """
    try:
        import server
        from aiohttp import web
    except Exception:
        return  # No server available (e.g. CLI mode, testing)

    if not hasattr(server, 'PromptServer') or not hasattr(server.PromptServer, 'instance'):
        return
    if server.PromptServer.instance is None:
        return

    _proxy_call_counts = {}  # path -> call count (for first-call debug)

    for route in routes:
        method = route.get("method", "POST").upper()
        path = route.get("path")
        handler_func = route.get("handler")
        module_name = route.get("module")
        if not path or not handler_func or not module_name:
            continue

        # Each closure must capture its own copy of the loop variables
        async def _make_proxy(request, _env_dir=env_dir, _pkg_root=package_root,
                              _sys_path=sys_path, _env_vars=env_vars,
                              _module=module_name, _func=handler_func,
                              _hc_timeout=health_check_timeout, _path=path,
                              _counts=_proxy_call_counts):
            _counts[_path] = _counts.get(_path, 0) + 1
            _first = _counts[_path] == 1

            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

            if _first:
                _log(f"[comfy-env] Route {_path}: first call, body keys={list(body.keys())}")

            worker, _ = _get_or_create_worker(
                _env_dir, _pkg_root, _sys_path, _env_vars, _hc_timeout,
            )
            if _first:
                _log(f"[comfy-env] Route {_path}: worker={worker.name}, calling {_module}.{_func}")

            import asyncio
            loop = asyncio.get_event_loop()

            def _routed_call():
                # In-flight bracket for admission (models charge full size
                # while the worker computes). call_module has no node
                # boundary, so on exit run a PEAK-ONLY raise pass over the
                # harvested census: peak writes are legal any time (upstream
                # never reads them), but the ledger and seq stay boundary
                # only, so the census remains in place for the full apply at
                # the env's next call_method.
                worker._calls_in_flight = getattr(worker, "_calls_in_flight", 0) + 1
                try:
                    return worker.call_module(_module, _func, 120.0, body=body)
                finally:
                    try:
                        _census = getattr(worker, "_last_residency", None)
                        _live = _WORKER_PATCHERS.get(str(_env_dir), {})
                        for _entry in _census or []:
                            _p = _live.get(_entry.get("id"))
                            if _p is not None:
                                _p._residency_peak = max(
                                    int(getattr(_p, "_residency_peak", 0)),
                                    int(_entry.get("resident", 0)))
                    except Exception:
                        pass
                    worker._calls_in_flight = max(
                        0, getattr(worker, "_calls_in_flight", 1) - 1)

            try:
                result = await loop.run_in_executor(None, _routed_call)
            except Exception as exc:
                _log(f"[comfy-env] Route {_path} error: {exc}")
                return web.json_response({"error": str(exc)}, status=500)

            if _first:
                _log(f"[comfy-env] Route {_path}: result keys={list(result.keys()) if isinstance(result, dict) else type(result)}")

            status = 200
            if isinstance(result, dict) and "_status" in result:
                status = result.pop("_status")
            return web.json_response(result, status=status)

        route_method = getattr(server.PromptServer.instance.routes, method.lower(), None)
        if route_method is None:
            _log(f"[comfy-env] Unknown HTTP method {method} for route {path}, skipping")
            continue
        route_method(path)(_make_proxy)
        _log(f"[comfy-env] Registered proxy route: {method} {path} -> {module_name}.{handler_func}")


# --- /free broadcast: the host's free button crosses the process boundary ---

#: Kill switch for comfy-env's first host-side function wrap. Off restores
#: byte-identical behavior; one release with a revert path that needs no
#: package rollback.
FREE_BROADCAST_ENV_VAR = "COMFY_ENV_FREE_BROADCAST"

_FREE_WRAP_INSTALLED = False
_LAST_RELEASE_BROADCAST = [0.0]


def _install_free_broadcast() -> None:
    """Wrap comfy.model_management.unload_all_models, once.

    Its exactly three upstream callers (main.py's /free flag path and
    execution.py's OOM and DISABLE_SMART_MEMORY fallbacks) all mean "release
    everything", which is the only honest trigger comfy-env can observe: the
    proxy-detach hook is blind to call_module packs and to a second /free
    (upstream pops unloaded ledger entries), and comfy-env's own admission
    eviction calls free_memory, never this, so no self-eviction guard is
    needed. Wrap calls the original FIRST (the sweep detaches worker models
    and drops their pin registrations through the real unpatch path), then
    broadcasts. Install failure logs once and degrades, never raises."""
    global _FREE_WRAP_INSTALLED
    if _FREE_WRAP_INSTALLED:
        return
    _FREE_WRAP_INSTALLED = True
    if os.environ.get(FREE_BROADCAST_ENV_VAR, "1").strip().lower() in (
            "0", "false", "off"):
        return
    try:
        import comfy.model_management as mm
        _original = mm.unload_all_models

        def _wrapped_unload_all_models(*args, **kwargs):
            result = _original(*args, **kwargs)
            try:
                broadcast_release()
            except Exception as exc:
                _log(f"[comfy-env] release broadcast failed: {exc}")
            return result

        mm.unload_all_models = _wrapped_unload_all_models
    except Exception as exc:
        _log(f"[comfy-env] free-broadcast wrap not installed: {exc}")


# --- Prompt epoch source: the host's one observer of prompt boundaries ----

_PROMPT_EPOCH_INSTALLED = False


def _install_prompt_epoch() -> None:
    """Class-patch comfy.model_patcher.PromptModelTracker.start, once.

    start() runs exactly once per prompt on the executor; bumping a monotonic
    counter there is the only prompt-boundary signal comfy-env can observe
    (workers run no executor). The counter rides every worker request as
    prompt_gen so workers can retire the PREVIOUS prompt's pin marks; the
    counter's 0 start is translated to None by senders, so a failed or
    switched-off patch degrades to the workers' sticky-with-decay fallback.
    Behind the same switch that gates the worker mark writes
    (COMFY_ENV_PIN_MARKS); install failure logs once and degrades, never
    raises. Order-independent with the /free wrap (disjoint targets)."""
    global _PROMPT_EPOCH_INSTALLED
    if _PROMPT_EPOCH_INSTALLED:
        return
    _PROMPT_EPOCH_INSTALLED = True
    if os.environ.get(state_sync.PIN_MARKS_ENV_VAR, "1").strip().lower() in (
            "0", "false", "off"):
        return
    try:
        import comfy.model_patcher as _cmp
        _orig_start = _cmp.PromptModelTracker.start

        def _epoch_start(self, *args, **kwargs):
            state_sync.PROMPT_GEN[0] += 1
            return _orig_start(self, *args, **kwargs)

        _cmp.PromptModelTracker.start = _epoch_start
    except Exception as exc:
        _log(f"[comfy-env] prompt-epoch patch not installed: {exc} "
             f"(workers fall back to sticky marks with decay)")


def broadcast_release() -> None:
    """Send full_release to every idle advertising worker, in parallel.

    Busy workers get a parent-owned deferral flag drained at their next node
    boundary (a mid-compute worker is not reading its socket, and its memory
    is in use anyway). Dead workers are skipped, never respawned. Every send
    binds its reply: the receipt's measured numbers are logged per worker,
    and the reply's piggybacked census and pin scalar are INGESTED here (a
    released worker may go quiet; waiting for a next call that never comes
    would advertise stale pins forever)."""
    now = time.monotonic()
    with _POOL_LOCK:
        entries = dict(_WORKER_POOL)
    plan = state_sync.plan_release_broadcast(
        {key: {"alive": worker.is_alive(),
               "advertises": getattr(worker, "supports_full_release", False)}
         for key, (worker, _g) in entries.items()},
        now, _LAST_RELEASE_BROADCAST[0])
    if plan["send"]:
        _LAST_RELEASE_BROADCAST[0] = now
    for key in plan["skip_dead"]:
        _log(f"[comfy-env] /free: worker {Path(key).name} is dead, skipped")

    def _release_one(key):
        worker, gen = entries[key]
        try:
            r = worker.send_command_no_spawn("full_release", lock_timeout=2.0)
            if r == "busy":
                worker._release_deferred = True
                _log(f"[comfy-env] /free: worker {Path(key).name} busy, "
                     f"release deferred to its next node boundary")
                return
            if r == "dead":
                return
            receipt = (r or {}).get("receipt") or {}
            _ingest_worker_frames(key, worker, gen)
            _log(f"[comfy-env] /free worker {Path(key).name}: reserved "
                 f"{receipt.get('reserved_before', 0) / 1e9:.2f}GB -> "
                 f"{receipt.get('reserved_after', 0) / 1e9:.2f}GB, pinned "
                 f"{receipt.get('pinned_before', 0) / 1e9:.2f}GB -> "
                 f"{receipt.get('pinned_after', 0) / 1e9:.2f}GB"
                 + (f", errors={receipt.get('errors')}"
                    if receipt.get("errors") else ""))
        except Exception as exc:
            _log(f"[comfy-env] /free: release of {Path(key).name} failed: {exc}")

    threads = [threading.Thread(target=_release_one, args=(k,), daemon=True)
               for k in plan["send"]]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90.0)


def _get_or_create_worker(env_dir: Path, working_dir: Path, sys_path: list[str],
                          env_vars: Optional[dict] = None,
                          health_check_timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT):
    """Get existing worker for this env, or create a new one.

    Returns (worker, generation) tuple.  The generation is a monotonically
    increasing integer used to detect stale ModelPatchers after worker restart.
    """
    global _WORKER_GENERATION
    key = str(env_dir)
    with _POOL_LOCK:
        entry = _WORKER_POOL.get(key)
        if entry is not None:
            worker, gen = entry
            if worker.is_alive():
                return worker, gen
            # Dead -- clean up stale patchers before replacing worker
            _cleanup_stale_patchers(env_dir)
            try:
                worker.shutdown()
            except Exception:
                pass
        _WORKER_GENERATION += 1
        gen = _WORKER_GENERATION
        # Pin budget bootstrap: a worker that has not seen a budget reply yet
        # starts on an equal split rather than believing it owns the whole
        # host allowance. Guarded not-in so pack [env_vars] wins; exported
        # only under PIN_SPLIT=auto so the off default stays byte-identical.
        if _pin_split_mode() == "auto":
            env_vars = dict(env_vars or {})
            try:
                import comfy.model_management as _mm
                _hm = int(getattr(_mm, "MAX_PINNED_MEMORY", 0) or 0)
                if _hm > 0 and state_sync.PIN_SHARE_ENV_VAR not in env_vars:
                    env_vars[state_sync.PIN_SHARE_ENV_VAR] = str(
                        _hm // (len(_WORKER_POOL) + 2))
                import comfy.memory_management as _cmm
                if state_sync.PIN_HEADROOM_ENV_VAR not in env_vars:
                    env_vars[state_sync.PIN_HEADROOM_ENV_VAR] = str(
                        int(getattr(_cmm, "RAM_CACHE_HEADROOM", 0)))
            except Exception:
                pass
        # Reserve bootstrap: the budget owner's advance payment, deliberately
        # OUTSIDE the pin split gate (it is not experimental). Injected only
        # when the host explicitly set --reserve-vram, read from the SAME
        # attribute the budget reply forwards (never recomputed from the GB
        # float flag: one computation, one owner, and the unit trap of
        # exporting "8" where bytes are owed dies structurally). Guarded
        # not-in so pack [env_vars] wins.
        try:
            import comfy.model_management as _rmm
            from comfy.cli_args import args as _rargs
            if getattr(_rargs, "reserve_vram", None) is not None \
                    and state_sync.RESERVE_ENV_VAR not in (env_vars or {}):
                env_vars = dict(env_vars or {})
                env_vars[state_sync.RESERVE_ENV_VAR] = str(
                    int(_rmm.EXTRA_RESERVED_VRAM))
        except Exception:
            pass
        worker = _create_worker(env_dir, working_dir, sys_path, env_vars, health_check_timeout)
        # Register bidirectional RPC callbacks. The budget callback carries
        # this worker's key so the pin allocator knows who is asking.
        worker.register_callback(
            "request_vram_budget",
            lambda req, _wk=key: _handle_vram_budget(req, worker_key=_wk))
        worker.register_callback("report_progress", _handle_progress)
        # Clean up stale patchers if worker restarts transparently via _ensure_started()
        worker._on_restart = lambda: _cleanup_stale_patchers(env_dir)
        # Canary handshake: verify each transport tier through the production
        # serialization path; demotes GPU zero-copy for this worker if its
        # round-trip fails. A CPU-tier failure raises (broken IPC).
        # Unconditional -- a correctness check with an off switch is a
        # doctrine with an asterisk (the old COMFY_ENV_TRANSPORT_PROBE=0
        # opt-out meant "assume every tier works, unverified").
        worker.verify_transport()
        _WORKER_POOL[key] = (worker, gen)
    # Deliberately outside _POOL_LOCK: this imports comfy modules and formats
    # strings, and it is idempotent per env, so it must not be held across
    # worker creation.
    _report_memory_manager(worker, env_dir)
    # Host-side patches install at first worker creation (idempotent, each
    # behind its own kill switch): without workers there is nothing for the
    # /free broadcast to reach and nobody to consume the prompt epoch.
    _install_free_broadcast()
    _install_prompt_epoch()
    return worker, gen


#: Env dirs already reported, so the routine line fires once per env rather
#: than once per node execution.
_MEMORY_MANAGER_REPORTED: set = set()




def _report_memory_manager(worker, env_dir) -> None:
    """Log which memory manager this worker resolved to, and warn on a mismatch.

    A worker never runs ``main.py``, so it resolves to the legacy ledger while
    the host is normally on aimdo. That is invisible today, and it is not even
    stable across installs: whether a pack is isolated at all is a per-pack
    decision, so two packs in one ComfyUI run can resolve differently with
    nothing announcing it. See :mod:`comfy_env.memory_manager`.
    """
    key = str(env_dir)
    if key in _MEMORY_MANAGER_REPORTED:
        return
    _MEMORY_MANAGER_REPORTED.add(key)
    try:
        from ..memory_manager import describe

        worker_info = getattr(worker, "memory_manager", None) or {}
        host_info = describe()
        worker_mgr = worker_info.get("manager", "unknown")
        host_mgr = host_info.get("manager", "unknown")
        name = getattr(worker, "name", key)
        if _DBG_WORKER:
            _log(
                f"[comfy-env] {name}: memory manager={worker_mgr} "
                f"(aimdo {worker_info.get('aimdo_version') or 'absent'}); "
                f"host={host_mgr} (aimdo {host_info.get('aimdo_version') or 'absent'})"
            )
        # Under follow-the-host a mismatch means THIS worker fell back
        # (failed init, CPU, skew), which is noteworthy per env, so the line is
        # ungated and carries the worker's own reason. "unknown" means the
        # report itself failed, which is equally worth a line.
        if worker_mgr != host_mgr or worker_mgr == "unknown":
            reason = worker_info.get("enable_error") or worker_info.get("reason", "unknown")
            _log(
                f"[comfy-env] WARNING {name}: memory manager={worker_mgr}, "
                f"host={host_mgr}; worker fell back ({reason}). "
                f"COMFY_ENV_WORKER_AIMDO=0 to silence."
            )
        # Version skew is reportable even when both sides resolve to the same
        # manager, and it happens: an unpinned `comfy-aimdo = "*"` in a pack's
        # comfy-env.toml resolves at solve time and drifts off the host's pin.
        worker_ver = worker_info.get("aimdo_version")
        host_ver = host_info.get("aimdo_version")
        if worker_ver and host_ver and worker_ver != host_ver:
            _log(
                f"[comfy-env] NOTE: {name} has comfy-aimdo {worker_ver}, host has "
                f"{host_ver}. Pin it in the pack's comfy-env.toml or let comfy-env "
                f"replicate the host's pin."
            )
    except Exception as exc:  # never let reporting break a worker start
        _log(f"[comfy-env] memory manager report failed: {exc}")


def _remove_worker(env_dir):
    """Remove a dead worker from the pool (called after crash)."""
    key = str(env_dir)
    # A replacement worker may resolve to a different manager (for example a
    # failed aimdo init this time), so let it be reported afresh.
    _MEMORY_MANAGER_REPORTED.discard(key)
    # Dead worker's pin report and grant leave the ledger: its key must be
    # ABSENT from the allocator's input (not retained at 0), so its share
    # redistributes on the next budget RPC.
    _PIN_REPORTS.pop(key, None)
    _PIN_GRANTS.pop(key, None)
    # Dead worker's overhead died with it; a retained entry would book ~1 GB
    # of phantom scratch per crash in a restart loop.
    _OVERHEAD_REPORTS.pop(key, None)
    with _POOL_LOCK:
        entry = _WORKER_POOL.pop(key, None)
        _WORKER_PATCHERS.pop(key, None)
        if entry is not None:
            worker, _ = entry
            try:
                worker.shutdown()
            except Exception:
                pass


def _shutdown_all_workers():
    """Shut down all persistent workers. Called via atexit."""
    with _POOL_LOCK:
        for key, (worker, _gen) in list(_WORKER_POOL.items()):
            try:
                worker.shutdown()
            except Exception:
                pass
        _WORKER_POOL.clear()
        _WORKER_PATCHERS.clear()
        _STALE_PATCHERS.clear()


atexit.register(_shutdown_all_workers)


def _insert_loaded_model(p, currently_used):
    """Insert one proxy into ComfyUI's ledger as a LoadedModel.

    Shared by first registration and by the post-eviction repair, so the two
    cannot drift. Inserting directly rather than via load_models_gpu is
    deliberate: that would try to load every model at once and OOM.
    """
    import weakref

    import comfy.model_management

    lm = comfy.model_management.LoadedModel(p)
    lm.currently_used = currently_used
    # Set real_model and model_finalizer (needed by model_unload)
    lm.real_model = weakref.ref(p.model)
    lm.model_finalizer = weakref.finalize(
        p.model, comfy.model_management.cleanup_models)
    lm.model_finalizer.atexit = False
    comfy.model_management.current_loaded_models.insert(0, lm)


def _ingest_worker_frames(env_dir, worker, generation):
    """Drain and apply everything a worker's frames piggybacked: the residency
    census, the pin scalar, and the measured VRAM overhead.

    One helper, two callers: the node boundary (_register_new_patchers) and
    the full_release broadcast's reply path -- harvest happens in
    _send_request for both, but harvested is not applied, and a released
    worker may go quiet, so the broadcast must apply its receipt itself
    rather than waiting for a next call that may never come.
    """
    global _OVERHEAD_SEQ
    _mode = os.environ.get(state_sync.RESIDENCY_ENV_VAR, "boundary").lower()
    if _mode not in ("off", "command", "0", "false"):
        census = getattr(worker, "_last_residency", None)
        if census:
            worker._last_residency = None
            live = {
                mid: p
                for mid, p in _WORKER_PATCHERS.get(str(env_dir), {}).items()
                if getattr(p, "_worker_generation", generation) == generation
            }
            state_sync.apply_residency(live, census, log=_log)

    # Pin census (observability, ungated: the rollup ships live while the
    # clamp stays dark behind COMFY_ENV_PIN_SPLIT).
    _pin = getattr(worker, "_last_pinned", None)
    if _pin is not None:
        worker._last_pinned = None
        _pin_ingest(str(env_dir), _pin)

    # Measured VRAM overhead: allocator bytes beyond registered residency
    # (cast buffers, cache). REPLACE on arrival order; self-measured
    # in-frame, so no peak is needed and stale-HIGH while idle over-books,
    # the safe direction.
    _ov = getattr(worker, "_last_vram_overhead", None)
    if _ov is not None:
        worker._last_vram_overhead = None
        _OVERHEAD_SEQ += 1
        state_sync.update_overhead_reports(_OVERHEAD_REPORTS, str(env_dir),
                                           _ov, _OVERHEAD_SEQ, log=_log)


def _register_new_patchers(env_dir, worker, generation):
    """Create SubprocessModelPatchers for any models auto-detected during the last call.

    Called after each call_method.  The worker's Module.to()/cuda() hooks
    auto-register nn.Modules that land on CUDA; the worker returns their
    metadata in response['_new_models'].  We create patchers here and register
    them with ComfyUI's memory manager so they participate in VRAM eviction.
    """
    # Release stale patchers from previous worker restarts.  Safe to do here
    # because we're outside free_memory's iteration loop.
    _STALE_PATCHERS.clear()

    # Apply the residency census FIRST, before the eviction repair and long
    # before the `if model_id in patchers: continue` skip below: that skip is
    # exactly what made already-known models unreachable, freezing their
    # registration-time stamp while aimdo paged residency out from under it.
    _ingest_worker_frames(env_dir, worker, generation)

    # Drain a /free release the broadcast deferred because this worker was
    # mid-call: the call just ended, the worker is idle between requests, and
    # this thread can win the lock immediately.
    if getattr(worker, "_release_deferred", False):
        worker._release_deferred = False
        try:
            r = worker.send_command_no_spawn("full_release", lock_timeout=2.0)
            if r == "busy":
                worker._release_deferred = True  # try again next boundary
            elif isinstance(r, dict):
                _ingest_worker_frames(env_dir, worker, generation)
                _log(f"[comfy-env] /free: deferred release of "
                     f"{Path(str(env_dir)).name} completed")
        except Exception as _fre:
            _log(f"[comfy-env] /free: deferred release failed: {_fre}")

    # Repair entries free_memory removed on a FAILED eviction. Upstream's
    # model_unload returns True even when detach() could not reach the worker
    # (model_management.py:811-815), so the ledger loses a model that is still
    # resident, and the skip-if-known check below would never re-add it. Safe
    # here: outside free_memory's iteration, same guarantee as the clear above.
    import comfy.model_management

    for p in list(_WORKER_PATCHERS.get(str(env_dir), {}).values()):
        if not getattr(p, "eviction_deferred", False):
            continue
        p.eviction_deferred = False
        if p.model.model_loaded_weight_memory <= 0:
            continue  # it drained on its own; nothing to repair
        if any(lm.model is p
               for lm in comfy.model_management.current_loaded_models):
            continue  # still listed; nothing was lost
        # Not currently_used: free_memory cleared that before popping, and
        # recomputing it from the device would resurrect eviction priority.
        _insert_loaded_model(p, currently_used=False)
        _log(f"[comfy-env] restored ledger entry for '{p._model_id}': "
             f"eviction could not reach a busy worker and upstream dropped it")

    # Drain: _send_request ACCUMULATES registrations (so no path drops them and
    # no interleaved command wipes them); this is the single consumer.
    new_models = list(getattr(worker, '_last_new_models', []))
    try:
        worker._last_new_models = []
    except Exception:
        pass
    if not new_models:
        return

    from .model_patcher import SubprocessModelPatcher

    try:
        import comfy.model_management
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
    except Exception:
        return

    key = str(env_dir)
    patchers = _WORKER_PATCHERS.setdefault(key, {})

    created = []
    for ref in new_models:
        model_id = ref["id"]
        if model_id in patchers:
            continue  # Already tracked
        patcher = SubprocessModelPatcher(
            worker=worker,
            worker_generation=generation,
            model_id=model_id,
            model_size=ref["size"],
            load_device=load_device,
            offload_device=offload_device,
            kind=ref.get("kind", "other"),
        )
        # Set device based on where the model actually is in the subprocess.
        # Models are auto-detected when they land on CUDA, but may have been
        # offloaded back to CPU by the time the call finishes.
        reported_device = ref.get("device", "cpu")
        if reported_device.startswith("cuda"):
            patcher.model.device = load_device
            patcher.model.model_loaded_weight_memory = ref["size"]
        else:
            patcher.model.device = offload_device
            patcher.model.model_loaded_weight_memory = 0
        patchers[model_id] = patcher
        created.append(model_id)

    if created:
        if _DBG_MODELS:
            _log(f"[comfy-env] Created {len(created)} model patchers: {created}")
        # Register with ComfyUI memory manager.  We insert LoadedModel
        # wrappers directly instead of calling load_models_gpu (which
        # would try to load all models simultaneously and OOM).
        for model_id in created:
            p = patchers[model_id]
            _insert_loaded_model(p, currently_used=(p.model.device == load_device))
