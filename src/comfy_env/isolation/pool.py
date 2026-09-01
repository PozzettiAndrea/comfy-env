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
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DEFAULT_HEALTH_CHECK_TIMEOUT
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

    The message must keep containing "interrupted": _progress_hook matches on
    it (_persistent_worker.py).
    """
    try:
        import comfy.model_management as mm
    except ImportError:
        mm = None  # not running inside ComfyUI -- nothing to cancel
    if mm is not None:
        try:
            mm.throw_exception_if_processing_interrupted()
        except mm.InterruptProcessingException:
            raise RuntimeError("Processing interrupted by user")
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
#: sees: CUDA context (~160 MB) + cuBLAS/cuDNN handles (~55 MB). Measured on
#: RTX 4060 Ti / driver 581.57 / torch 2.10+cu128. Additive per live worker --
#: unlike the model-size headroom, which is multiplicative.
_WORKER_FIXED_VRAM_COST = 250 * 1024 * 1024

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
    held = 0
    n_workers = 0
    # Snapshot: _register_new_patchers and _cleanup_stale_patchers mutate
    # these from the aiohttp executor thread, and _cleanup_stale_patchers runs
    # outside _POOL_LOCK. Not taking the lock here on purpose -- it is a plain
    # Lock held across verify_transport(), so re-entering would deadlock.
    for _key, patchers in list(_WORKER_PATCHERS.items()):
        if patchers:
            n_workers += 1
        for p in list(patchers.values()):
            try:
                held += int(p.loaded_size())
            except Exception:
                pass
    return held + n_workers * _WORKER_FIXED_VRAM_COST


def _handle_vram_budget(request: dict) -> dict:
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
    need = int(total_requested * _REQUEST_SLACK) + _WORKER_FIXED_VRAM_COST + min_inference

    if _DBG_MODELS:
        _log(f"[comfy-env] VRAM request: {total_requested / 1e9:.2f}GB | "
             f"free: blind={blind_free / 1e9:.2f}GB true={true_free / 1e9:.2f}GB "
             f"offset={offset / 1e9:.2f}GB ({offset_source}) | "
             f"need={need / 1e9:.2f}GB -> asking free_memory for "
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

    return {
        "device": str(device),
        "extra_reserved_vram": extra_reserved,
        "vram_state": vram_state_name,
        "device_free_bytes": int(post_true_free),
    }


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
            try:
                result = await loop.run_in_executor(
                    None, lambda: worker.call_module(_module, _func, 120.0, body=body),
                )
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
        worker = _create_worker(env_dir, working_dir, sys_path, env_vars, health_check_timeout)
        # Register bidirectional RPC callbacks
        worker.register_callback("request_vram_budget", _handle_vram_budget)
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
