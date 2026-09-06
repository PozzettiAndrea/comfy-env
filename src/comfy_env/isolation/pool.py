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
from .. import reserve, state_sync
from . import observer
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

# --- Pin census (observability and the reclaim path) ----------------------
# reports: "host" plus str(env_dir) -> {"pinned": bytes, "seq": n}. seq is a
# PARENT-side arrival stamp (one socket per worker makes arrival order causal
# order); the worker's own _pin_state.seq is observability, not the ledger.
_PIN_REPORTS: Dict[str, Dict[str, int]] = {}
_PIN_INGEST_SEQ = 0
_PIN_ROLLUP_LAST = 0  # last logged pinned total (rollup fires on >1 GiB moves)
_PIN_REGRESSION_SEEN: Dict[str, int] = {}  # active-eviction bytes last logged per worker


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


def _pin_ingest(key: str, pinned) -> None:
    """Stamp one pin report into the ledger. Never raises.

    Consumer-set stability used to be tracked here to damp the grant's
    growth. With the allocation half deleted there is no grant to damp.
    """
    global _PIN_INGEST_SEQ, _PIN_ROLLUP_LAST
    try:
        _PIN_INGEST_SEQ += 1
        state_sync.update_pin_reports(_PIN_REPORTS, key, int(pinned),
                                      _PIN_INGEST_SEQ)
        reports = list(_PIN_REPORTS.items())  # snapshot: pops race this
        total = sum(r.get("pinned", 0) for _k, r in reports)
        if abs(total - _PIN_ROLLUP_LAST) > 1024 ** 3:
            _PIN_ROLLUP_LAST = total
            _log("[comfy-env] pinned RAM rollup: "
                 + ", ".join(f"{k}={r.get('pinned', 0) / 1e9:.2f}GB"
                             for k, r in sorted(reports))
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



def _blind_free_is_process_local() -> bool:
    """Platform seam for the ledger fallback; the verdict itself is pure."""
    return state_sync.blind_free_is_process_local(sys.platform)


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


#: The reserve comfy-env has published into ComfyUI's own EXTRA_RESERVED_VRAM,
#: and the base it found there first. The base is the operator's own
#: --reserve-vram instruction: comfy-env ADDS to it and must never lose it.
_RESERVE_BASE = None
_RESERVE_PUBLISHED = 0
#: Per-worker high-water residency. High-water rather than current, because
#: the reserve exists to stop the host taking space a worker is about to need
#: again; see reserve.entitlement.
_RESERVE_HIGHWATER: Dict[str, int] = {}


#: When each worker last finished a call, for the idle release policy.
_LAST_ACTIVITY: Dict[str, float] = {}
#: Which prompt each worker last served, so a worker whose prompt is over is
#: released at once instead of after the idle timer.
_LAST_PROMPT: Dict[str, Any] = {}

#: The host pager's own headroom seed (--reserve-vram, else aimdo's default),
#: read once: the reserve is forwarded into the pager as seed plus what
#: comfy-env added, and the seed must not drift with our own writes.
_AIMDO_SEED = None
_AIMDO_SETTER_MISSING_LOGGED = False

#: The idle sweep used to run only at worker call boundaries, so a prompt
#: made of host nodes after a worker prompt never released anything: the
#: worker sat on its VRAM until the next worker call. A daemon timer runs
#: the same sweep on a clock instead.
IDLE_SWEEP_INTERVAL_SECONDS = 10.0
_IDLE_SWEEP_STARTED = False


def _current_prompt():
    """The running prompt's id, read from ComfyUI's progress registry.

    Same source as workers/subprocess._current_prompt_gen, kept separate so
    pool never imports the worker module. None outside a prompt or on trees
    that predate the registry.
    """
    try:
        from comfy_execution.progress import get_progress_state
        return getattr(get_progress_state(), "prompt_id", None) or None
    except Exception:
        return None


def _note_activity(env_dir) -> None:
    """Mark a worker as active now. Cheap enough for every node boundary."""
    _LAST_ACTIVITY[str(env_dir)] = time.monotonic()
    prompt = _current_prompt()
    if prompt is not None:
        _LAST_PROMPT[str(env_dir)] = prompt


def _idle_sweep_loop() -> None:
    while True:
        time.sleep(IDLE_SWEEP_INTERVAL_SECONDS)
        _release_idle_workers()


def _start_idle_sweep() -> None:
    """Start the clock driven idle sweep, once per process. Daemon: it must
    never keep ComfyUI alive at shutdown."""
    global _IDLE_SWEEP_STARTED
    if _IDLE_SWEEP_STARTED:
        return
    _IDLE_SWEEP_STARTED = True
    threading.Thread(target=_idle_sweep_loop, name="comfy-env-idle-sweep",
                     daemon=True).start()


def _card_is_tight() -> bool:
    """Whether the card is short enough that a warm worker should let go.

    The prompt boundary is the cheapest moment to take memory back, not a
    reason to take it back: cooling a worker between two queued prompts
    reloads its model on the next one, which is what a model cache exists to
    avoid. So the early release needs a reason, and this is it. Compared
    against what the reserve already says the workers will want, so the test
    is "the card cannot cover what is booked", not an arbitrary fraction.
    """
    try:
        import comfy.model_management as mm
        device = mm.get_torch_device()
        free = _true_device_free(device)
        if free is None:
            free = mm.get_free_memory(device)
        return int(free) < int(_RESERVE_PUBLISHED)
    except Exception:
        return False


def _release_idle_workers() -> None:
    """Ask workers that have been idle a while to give their VRAM back.

    This is the whole replacement for host-driven reclaim. comfy-env no
    longer registers a proxy that ComfyUI can evict, so nothing can take a
    worker's memory from outside; instead the worker lets go on its own and
    the reserve follows it down.

    A successful release is the measured receipt that permits the reserve to
    SHRINK: the worker has told us what it freed, so the space is provably
    back and its high-water forecast starts again from nothing.
    """
    try:
        with _POOL_LOCK:
            entries = dict(_WORKER_POOL)
        states = {}
        for key, (worker, _gen) in entries.items():
            states[key] = {
                "alive": worker.is_alive(),
                "advertises": getattr(worker, "supports_full_release", False),
                "in_flight": getattr(worker, "_calls_in_flight", 0) > 0,
                "idle_since": _LAST_ACTIVITY.get(key),
                "holding": _RESERVE_HIGHWATER.get(key, 0) > 0,
                "last_prompt": _LAST_PROMPT.get(key),
            }
        due = state_sync.plan_idle_release(
            states, time.monotonic(), current_prompt=_current_prompt(),
            under_pressure=_card_is_tight())
        for key in due:
            worker, _gen = entries[key]
            try:
                reply = worker.send_command_no_spawn("full_release",
                                                     lock_timeout=2.0)
            except Exception as exc:
                _log(f"[comfy-env] idle release of {Path(key).name} failed: {exc}")
                continue
            if reply == "busy":
                continue
            receipt = (reply or {}).get("receipt") if isinstance(reply, dict) else None
            _RESERVE_HIGHWATER[key] = 0
            _LAST_ACTIVITY[key] = time.monotonic()
            _LAST_PROMPT.pop(key, None)
            _log(f"[comfy-env] idle release: {Path(key).name} gave back "
                 f"{(receipt or {}).get('freed_bytes', 0) / 1e9:.2f}GB")
        if due:
            # The receipts are the evidence a shrink needs.
            _publish_reserve(shrink_allowed=True)
    except Exception as exc:
        _log(f"[comfy-env] idle release sweep failed: {exc}")


def _worker_charges() -> Dict[str, int]:
    """Each live worker's charge against the reserve, from what it MEASURED.

    Not from our proxies: ComfyUI's ledger reads zero for a paged model and
    torch cannot see aimdo at all, so a ledger sum is wrong in exactly the
    configuration the reserve exists for. The high-water is updated here as
    a side effect, because this is the one place residency is read.
    """
    process_local = _blind_free_is_process_local()
    charges: Dict[str, int] = {}
    for key, entry in list(_WORKER_POOL.items()):
        worker = entry[0] if entry else None
        if worker is not None and not worker.is_alive():
            continue
        residency = 0
        report = getattr(worker, "_last_vram_report", None) or {}
        measured = report.get("held")
        if measured is not None:
            residency = max(0, int(measured))
        else:
            for patcher in list(_WORKER_PATCHERS.get(key, {}).values()):
                try:
                    residency += int(getattr(
                        patcher.model, "model_loaded_weight_memory", 0))
                except Exception:
                    pass
        high = max(_RESERVE_HIGHWATER.get(key, 0), residency)
        _RESERVE_HIGHWATER[key] = high
        charges[key] = reserve.charge(
            reserve.entitlement(high, floor=_WORKER_FIXED_VRAM_COST),
            residency, process_local)
    return charges


def _aimdo_headroom_seed() -> int:
    """The pager's startup headroom, read once: --reserve-vram in bytes the
    way main.py seeds it, else aimdo's compile time default."""
    global _AIMDO_SEED
    if _AIMDO_SEED is None:
        seed = None
        try:
            from comfy.cli_args import args as _args
            rv = getattr(_args, "reserve_vram", None)
            if rv is not None:
                seed = int(float(rv) * 1024 ** 3)
        except Exception:
            seed = None
        _AIMDO_SEED = reserve.AIMDO_DEFAULT_HEADROOM if seed is None else seed
    return _AIMDO_SEED


def _forward_reserve_to_aimdo(published: int) -> bool:
    """Mirror the published reserve into the host pager's headroom.

    ComfyUI's EXTRA_RESERVED_VRAM never reaches comfy-aimdo: the pager is
    seeded once at startup (main.py) and decides residency at fault time
    from its own headroom. Measured on the paged path (research/memory-floor
    P2 and its 2026-09-05 replication): the published reserve is inert, the
    pager's own setter is live at the next fault. So the second of the two
    writes comfy-env makes is this one, a value into the knob the library
    exports for it, the same knob ComfyUI seeds from --reserve-vram.

    Bound lazily: the Python wrapper is comfy-aimdo #107; every wheel since
    0.4.10 carries the raw export. Gated on the host actually running the
    pager; on the legacy path the published reserve already does the job.
    """
    try:
        import comfy.memory_management as _cmm
        if not getattr(_cmm, "aimdo_enabled", False):
            return False
        import comfy_aimdo.control as _control
    except Exception:
        return False
    setter = getattr(_control, "set_simple_vram_headroom", None)
    if setter is None:
        lib = getattr(_control, "lib", None)
        setter = getattr(lib, "set_simple_vram_headroom", None) if lib is not None else None
    if setter is None:
        # Silent here would be wrong: the reserve keeps being published and
        # keeps being inert on this path, which looks like the floor working.
        global _AIMDO_SETTER_MISSING_LOGGED
        if not _AIMDO_SETTER_MISSING_LOGGED:
            _AIMDO_SETTER_MISSING_LOGGED = True
            _log("[comfy-env] NOTE: this comfy-aimdo exports no "
                 "set_simple_vram_headroom, so the published reserve cannot "
                 "reach the pager; it stays preventive on the legacy path "
                 "and reactive on the paged one.")
        return False
    headroom = reserve.aimdo_headroom(_aimdo_headroom_seed(), published, _RESERVE_BASE)
    try:
        setter(int(headroom))
    except Exception as exc:
        _log(f"[comfy-env] pager headroom forward failed: {exc}")
        return False
    if _DBG_MODELS:
        _log(f"[comfy-env] pager headroom {headroom / 1e9:.2f}GB "
             f"(seed {_aimdo_headroom_seed() / 1e9:.2f}GB + reserve added)")
    return True


#: The optional listener in ComfyUI's loaded-model list, if an operator
#: turned it on. One per process, planted at first worker creation.
_OBSERVER = None


def _install_observer() -> bool:
    """Plant the read-only listener, once, if COMFY_ENV_MEMORY_OBSERVER is on.

    comfy-env patches nothing, so two signals it would like are simply not
    delivered: the Free-memory button and the host's OOM handler both reach
    ComfyUI's own eviction loop and never leave the process. An entry in the
    list is asked, and that is the only way to hear them without replacing a
    function.

    Off by default because it is still a coupling: an object of ours inside
    their bookkeeping, which is the surface both of comfy-env's loud breaks
    came through. This one reports holding nothing, so no upstream decision
    depends on its answers; see isolation/observer.py for why that is the
    difference that matters.

    Both callbacks POST and return. They run on ComfyUI's thread inside
    free_memory inside a node, so waiting on a worker here would stall every
    host load.
    """
    global _OBSERVER
    if _OBSERVER is not None:
        return True
    if not observer.enabled(os.environ):
        return False
    try:
        import comfy.model_management as mm
    except ImportError:
        return False
    try:
        def _on_free_all():
            threading.Thread(target=broadcast_release,
                             name="comfy-env-free-all", daemon=True).start()

        def _on_pressure(nbytes):
            threading.Thread(target=_ask_idle_workers, args=(int(nbytes),),
                             name="comfy-env-pressure", daemon=True).start()

        obs = observer.MemoryObserver(device=mm.get_torch_device(),
                                      on_free_all=_on_free_all,
                                      on_pressure=_on_pressure)
        mm.current_loaded_models.append(obs)
        _OBSERVER = obs
        _log("[comfy-env] memory observer on: the Free button and OOM now "
             "reach workers (COMFY_ENV_MEMORY_OBSERVER)")
        return True
    except Exception as exc:
        _log(f"[comfy-env] memory observer not installed: {exc}")
        return False


def _ask_idle_workers(shortfall: int, requester_key=None) -> int:
    """Ask idle siblings to give back ``shortfall`` bytes, now.

    The host has just evicted everything it owns and is still short. Nothing
    outside a process can free that process's VRAM, so the parent asks the
    processes it owns: idle workers shrink through their own manager and
    keep their models loaded, refaulting from pinned RAM on their next call.
    Measured (research/memory-floor/p6_partial_release): 0.05 to 0.20 s to
    free 2 to 6 GiB, visible to the driver at once, 0.09 to 0.30 s to
    refault.

    This is the admission time half of what replaces host driven reclaim;
    the idle timer and the prompt boundary are the unhurried halves. Returns
    bytes actually given back, measured from the receipts.
    """
    if shortfall <= 0:
        return 0
    with _POOL_LOCK:
        entries = dict(_WORKER_POOL)
    states = {}
    for key, (worker, _gen) in entries.items():
        states[key] = {
            "alive": worker.is_alive(),
            "advertises": getattr(worker, "supports_partial_release", False),
            "in_flight": getattr(worker, "_calls_in_flight", 0) > 0,
            "held": _RESERVE_HIGHWATER.get(key, 0),
        }
    plan = state_sync.plan_pressure_release(states, shortfall,
                                            requester=str(requester_key)
                                            if requester_key else None)
    freed_total = 0
    for key, ask in plan:
        worker, _gen = entries[key]
        try:
            reply = worker.send_command_no_spawn("partial_release", size=int(ask),
                                                 lock_timeout=2.0)
        except Exception as exc:
            _log(f"[comfy-env] admission ask to {Path(key).name} failed: {exc}")
            continue
        if reply == "busy":
            continue
        receipt = (reply or {}).get("receipt") if isinstance(reply, dict) else None
        freed = int((receipt or {}).get("freed_bytes", 0))
        freed_total += freed
        # The high-water is now a forecast the worker has disproved: it let
        # this much go on request, so the reserve may follow it down.
        _RESERVE_HIGHWATER[key] = max(0, _RESERVE_HIGHWATER.get(key, 0) - freed)
        _log(f"[comfy-env] admission ask: {Path(key).name} gave back "
             f"{freed / 1e9:.2f}GB of {ask / 1e9:.2f}GB asked")
    if freed_total:
        # Receipts in hand, so the reserve may shrink to match.
        _publish_reserve(shrink_allowed=True)
    return freed_total


def _publish_reserve(shrink_allowed: bool = False) -> int:
    """Tell ComfyUI the card is smaller by what workers will take.

    Writes ComfyUI's own EXTRA_RESERVED_VRAM, the knob --reserve-vram sets,
    which its admission reads live on every load, and mirrors the same
    reserve into the host pager's headroom (see _forward_reserve_to_aimdo).
    Preventive on the legacy path, where the partial load budget shrinks
    with it; on the paged path only the pager forward moves residency.

    The base is read once, before comfy-env has ever written it, or every
    republish would add the previous reserve back. Grow at once; shrink only
    with ``shrink_allowed``, which callers pass with a measured release
    receipt in hand (reserve.next_reserve).
    """
    global _RESERVE_BASE, _RESERVE_PUBLISHED
    try:
        import comfy.model_management as mm
    except ImportError:
        return 0
    try:
        if _RESERVE_BASE is None:
            # Read once, before comfy-env has ever written it, or the base
            # compounds: every republish would add the previous reserve back.
            _RESERVE_BASE = int(mm.EXTRA_RESERVED_VRAM)

        charges = _worker_charges()
        proposed = reserve.total_reserve(
            _RESERVE_BASE, list(charges.values()), device_total=_device_total_bytes())
        value = reserve.next_reserve(
            _RESERVE_PUBLISHED, proposed, shrink_allowed)
        if value != _RESERVE_PUBLISHED:
            mm.EXTRA_RESERVED_VRAM = value
            _RESERVE_PUBLISHED = value
            _forward_reserve_to_aimdo(value)
            if _DBG_MODELS:
                _log(f"[comfy-env] reserve published {value / 1e9:.2f}GB "
                     f"(base {_RESERVE_BASE / 1e9:.2f}GB, "
                     f"{len(charges)} worker(s))")
        return value
    except Exception as exc:
        _log(f"[comfy-env] reserve publish failed: {exc}")
        return _RESERVE_PUBLISHED


def _forget_reserve(env_dir) -> None:
    """Drop a dead worker's high-water so its space returns to the host.

    Called on real worker removal only. A restart keeps nothing: the new
    worker starts at the context floor and earns its high-water again.
    """
    _RESERVE_HIGHWATER.pop(str(env_dir), None)


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
    # list() first: a single C-level call, GIL-atomic, so a concurrent
    # setdefault from the aiohttp thread cannot raise "dictionary changed
    # size during iteration" mid-comprehension (which was swallowed into
    # a failed budget reply: no eviction, then OOM). A one-tick-stale
    # snapshot is benign here.
    keys = set(_WORKER_POOL) | {k for k, v in list(_WORKER_PATCHERS.items()) if v}
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


def _ingest_pin_state(request: dict, worker_key) -> None:
    """Take the pin census riding a budget request. Never raises.

    This used to also hand back a per-worker pin CEILING split from the
    host's own. That half is gone. It never shipped (dark behind
    COMFY_ENV_PIN_SPLIT) and three separate findings said it should not:
    comfy's own ensure_pin_budget already stops pinning from the global
    available-RAM figure, so the ceiling was never the binding guard; the
    same ceiling sizes each model's host buffer through pinned_hostbuf_size,
    so a small grant would have silently capped a large model's buffer; and
    it re-derived an upstream number, which is the coupling this redesign
    exists to remove.

    The census stays, because the reclaim path and the regression line read
    it and neither re-derives anything.
    """
    try:
        ps = request.get("pin_state")
        if not isinstance(ps, dict) or not worker_key:
            return
        _pin_ingest(worker_key, ps.get("total_pinned", 0))
        # Always on: an active-pin eviction is the regression the prompt
        # marks exist to prevent, so it must be visible without a debug flag.
        line = state_sync.pin_regression_line(
            Path(worker_key).name, ps, _PIN_REGRESSION_SEEN)
        if line:
            _log(line)
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
        if _blind_free_is_process_local():
            # WDDM, no NVML/nvidia-smi: the blind number excludes every
            # other process, so reconstruct from comfy-env's own ledger.
            true_free = max(0, blind_free - _worker_held_bytes())
            offset_source = "ledger"
        else:
            # Device-wide mem_get_info (Linux, macOS): the blind number
            # already counts the workers; subtracting the ledger again
            # double-books them (state_sync.blind_free_is_process_local).
            true_free = blind_free
            offset_source = "blind"
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
    # Upstream's own expression, not a comfy-env invention. ComfyUI frees
    # `weights * 1.1 + max(inference_memory, memory_required + reserved)` for
    # its own loads (model_management.py load_models_gpu), and every place
    # comfy-env re-derived that shape it drifted: the shipped 1.02 slack
    # under-freed by 680 MiB on a 12 GiB model. `forward` plays the part of
    # upstream's `memory_required`, being what the incoming load will want
    # for cast buffers at its first forward.
    try:
        _host_reserved = int(mm.extra_reserved_memory())
    except Exception:
        _host_reserved = 0
    # The published reserve already holds the requester's own growth; this
    # load IS that growth. Asking the host to keep it free on top of the
    # weights evicts host models for the same bytes twice.
    try:
        _own_charge = _worker_charges().get(requester_key, 0) if requester_key else 0
    except Exception:
        _own_charge = 0
    _host_reserved = reserve.reserve_for_requester(_host_reserved, _own_charge)
    need = reserve.ask_target(
        weights=total_requested,
        slack=state_sync.WEIGHT_SLACK,
        min_inference=min_inference,
        extra_reserved=_host_reserved,
        want_inference=forward,
    )
    # Two terms upstream has no analogue for, because an in-process load has
    # neither: the worker's own CUDA context, and the allocator overhead it
    # measured and reported. Additive, never folded into the multiplier.
    need += _WORKER_FIXED_VRAM_COST + requester_excess

    _inflight = sum(1 for _e in list(_WORKER_POOL.values())
                    if getattr(_e[0], "_calls_in_flight", 0) > 0)
    if need > true_free:
        # Always on: this is the moment the host is about to evict on a
        # worker's behalf (or fail to). Silent, it is indistinguishable from
        # a plain host OOM in a user's log.
        _log(f"[comfy-env] admission tight env={Path(str(worker_key)).name} "
             f"need={need / 1e9:.2f}GB true_free={true_free / 1e9:.2f}GB "
             f"in_flight={_inflight} forward={forward / 1e9:.2f}GB "
             f"excess={requester_excess / 1e9:.2f}GB offset={offset / 1e9:.2f}GB "
             f"({offset_source})")
    if _DBG_MODELS:
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

    # The host has now given up everything it owns. If the card is still
    # short, the rest is held by processes ComfyUI cannot reach and the
    # parent can: ask idle siblings before the requester loads into a card
    # that has no room, which is the OOM this whole floor exists to avoid.
    post_free = _true_device_free(device)
    if post_free is None:
        _pb = mm.get_free_memory(device)
        post_free = (max(0, _pb - _worker_held_bytes())
                     if _blind_free_is_process_local() else _pb)
    if need > post_free:
        _ask_idle_workers(need - int(post_free), requester_key)

    # vram_state/extra_reserved pass through as ComfyUI computed them; the
    # negotiation below is the only mechanism that adjusts them.
    vram_state_name = mm.vram_state.name
    # Discounted the same way the ask above is. The published reserve holds
    # every worker's growth including this one's, and the worker assigns this
    # number to its OWN EXTRA_RESERVED_VRAM, where it shrinks the weight
    # budget of the very load it is about to do. Handing over the raw total
    # made both ends under load: the host freed for a reserve that already
    # counted the requester, and the requester then reserved against itself.
    extra_reserved = reserve.reserve_for_requester(
        int(mm.EXTRA_RESERVED_VRAM), _own_charge)

    # Re-measure after eviction: the worker corrects its own blind view from
    # this (its get_free_memory - device_free = what everyone else holds).
    post_true_free = _true_device_free(device)
    if post_true_free is None:
        post_blind = mm.get_free_memory(device)
        # Same platform verdict as above: the ledger corrects a process-local
        # number only (a device-wide one already counts the workers).
        post_true_free = (max(0, post_blind - _worker_held_bytes())
                          if _blind_free_is_process_local() else post_blind)

    reply = {
        "device": str(device),
        "extra_reserved_vram": extra_reserved,
        "vram_state": vram_state_name,
        "device_free_bytes": int(post_true_free),
    }
    _ingest_pin_state(request, worker_key)
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
    _PIN_REPORTS.pop(key, None)       # and its pins; parity with _remove_worker
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
                # harvested census (state_sync.apply_peak_raise, under the
                # worker's leaf mutex like every residency writer): peak
                # writes are legal any time (upstream never reads them), but
                # the ledger and seq stay boundary only, so the census
                # remains in place for the full apply at the env's next
                # call_method.
                worker.begin_call()
                try:
                    return worker.call_module(_module, _func, 120.0, body=body)
                finally:
                    try:
                        _census = (getattr(worker, "_last_vram_report", None)
                                   or {}).get("residency")
                        _live = dict(_WORKER_PATCHERS.get(str(_env_dir), {}))
                        with worker._mem_lock:
                            state_sync.apply_peak_raise(_live, _census)
                    except Exception:
                        pass
                    worker.end_call()

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


# --- Release levers: full_release and pin release, sent to workers -----
# Nothing in the host calls these on its own: comfy-env does not patch the
# host, so the Free button and the host's RAM-pressure sweep do not reach
# workers unless something inside comfy-env (idle release, the optional
# observer) decides to call them.


_LAST_RELEASE_BROADCAST = [0.0]

#: Serializes the once-per-process host contract check. Nothing else ever
#: takes this lock.
_INSTALL_LOCK = threading.Lock()




_LAST_PIN_PRESSURE_SWEEP = [0.0]



def broadcast_pin_release(target_bytes: int) -> None:
    """Ask workers to release pinned RAM, proportional to their census.

    NO thread join, unlike broadcast_release: this fires from the execution
    loop between nodes and must never block it. Daemon threads send, replies
    converge the ledgers, busy workers get a parent-owned deferral drained at
    their next node boundary, dead workers are skipped."""
    now = time.monotonic()
    asks = state_sync.plan_pin_pressure(dict(_PIN_REPORTS), int(target_bytes),
                                        now, _LAST_PIN_PRESSURE_SWEEP[0])
    if not asks:
        return
    _LAST_PIN_PRESSURE_SWEEP[0] = now
    with _POOL_LOCK:
        entries = dict(_WORKER_POOL)

    def _ask_one(key, size):
        entry = entries.get(key)
        if entry is None:
            return
        worker, gen = entry
        if not worker.is_alive() or                 not getattr(worker, "supports_release_pins", False):
            return
        try:
            r = worker.send_command_no_spawn("release_pins", size=size,
                                             lock_timeout=2.0)
            if r == "busy":
                worker._pin_release_deferred = size
                return
            if isinstance(r, dict):
                _ingest_worker_frames(key, worker, gen)
                receipt = r.get("receipt") or {}
                _log(f"[comfy-env] RAM pressure: worker {Path(key).name} "
                     f"released pins "
                     f"{receipt.get('pinned_before', 0) / 1e9:.2f}GB -> "
                     f"{receipt.get('pinned_after', 0) / 1e9:.2f}GB"
                     + (f", errors={receipt.get('errors')}"
                        if receipt.get("errors") else ""))
        except Exception as exc:
            _log(f"[comfy-env] pin release of {Path(key).name} failed: {exc}")

    for key, size in asks.items():
        threading.Thread(target=_ask_one, args=(key, size), daemon=True).start()


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


_CONTRACT_CHECKED = False


_MEMORY_LEVEL = None


def _memory_level_facts() -> dict:
    """What this host can actually support, probed by name. Never raises."""
    facts = {"aimdo_available": False, "marks_available": False,
             "pressure_available": False}
    try:
        import comfy.model_management as mm
        facts["pressure_available"] = hasattr(
            mm, "should_free_pins_for_ram_pressure")
        facts["marks_available"] = hasattr(mm, "free_model_pins")
    except Exception:
        pass
    try:
        import comfy.model_patcher as cmp
        facts["marks_available"] = facts["marks_available"] and hasattr(
            cmp, "PromptModelTracker")
    except Exception:
        facts["marks_available"] = False
    try:
        from ..memory_manager import aimdo_version
        facts["aimdo_available"] = bool(aimdo_version())
    except Exception:
        pass
    return facts


def _resolve_memory_level() -> str:
    """Settle COMFY_ENV_MEMORY_MANAGEMENT once, and say so if it dropped.

    An unrequested demotion is loud and a requested one is silent. That
    polarity is the point: four worker environments on the development
    machine ran a different memory manager than their host and nothing
    reported it, while a deliberate choice of a lower level is a decision
    and warning about it would train the operator to ignore the channel.
    """
    global _MEMORY_LEVEL
    if _MEMORY_LEVEL is not None:
        return _MEMORY_LEVEL
    from .. import memlevel
    requested = os.environ.get(memlevel.ENV_VAR)
    level, note = memlevel.resolve(requested, _memory_level_facts())
    _MEMORY_LEVEL = level
    if note:
        _log(f"[comfy-env] memory management: {note}")
    elif _DBG_MODELS:
        _log(f"[comfy-env] memory management: {level}")
    return level


def _check_host_contract() -> None:
    """Verify the host satisfies what comfy-env requires of it, once.

    Runs before the first worker exists, so a host that cannot support the
    floor says so at that moment rather than as a wrong number later. FATAL
    gaps raise, because they produce wrong VRAM arithmetic; everything else
    is one named line carrying the ComfyUI version that introduced it.

    Never patches, wraps or otherwise touches ComfyUI: this only reads.
    """
    global _CONTRACT_CHECKED
    with _INSTALL_LOCK:
        if _CONTRACT_CHECKED:
            return
        _CONTRACT_CHECKED = True
    try:
        from .. import contract
        ok, failures, notes = contract.check(
            side=contract.HOST, tiers=(contract.FLOOR,))
    except Exception as exc:
        _log(f"[comfy-env] contract check skipped: {exc}")
        return
    for note in notes:
        _log(f"[comfy-env] host contract: {note}")
    if not ok:
        raise RuntimeError(
            "comfy-env cannot manage memory against this ComfyUI: "
            + "; ".join(failures)
        )



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
        # Reserve bootstrap: the budget owner's advance payment. Injected
        # only when the host explicitly set --reserve-vram, read from the
        # SAME attribute the budget reply forwards (never recomputed from
        # the GB float flag: one computation, one owner, and the unit trap
        # of exporting "8" where bytes are owed dies structurally). Guarded
        # not-in so pack [env_vars] wins.
        #
        # This block was previously adjacent to the pin-split bootstrap and
        # was deleted along with it on the first attempt; the seam tests
        # caught it. It is unrelated to the pin split and has no gate.
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
    _check_host_contract()
    _start_idle_sweep()
    _install_observer()
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
        # Two majors of one CUDA library in a worker: nobody chose it, it
        # costs real memory, and it is invisible without this line. Measured
        # about 92 MB of private RAM for a second cuBLASLt. Reported per
        # worker because it depends on that env's own wheels.
        # What the worker's own contract check found. Its entries are all
        # WORKER side and PAGED tier, so the host's FLOOR check cannot see
        # them; the worker evaluates and the parent reports.
        _wc = worker_info.get("contract") or {}
        for _line in list(_wc.get("failures") or []) + list(_wc.get("notes") or []):
            _log(f"[comfy-env] {name} contract: {_line}")
        for lib, paths in (worker_info.get("duplicate_cuda_majors") or {}).items():
            _log(
                f"[comfy-env] NOTE: {name} maps two majors of lib{lib}: "
                f"{', '.join(paths)}. One is a system CUDA found through the "
                f"loader cache; it costs private RAM per worker and runs two "
                f"majors against one torch build."
            )
        # comfy-kitchen gets a version line and nothing else. It fails loudly
        # by itself: ComfyUI calls int8_attention_is_available() at module
        # scope, so drift is an AttributeError at worker start with a file, a
        # line and a name. What a reader needs is which two versions were in
        # play when that happened.
        w_kitchen = worker_info.get("kitchen_version")
        h_kitchen = host_info.get("kitchen_version")
        if w_kitchen and h_kitchen and w_kitchen != h_kitchen:
            _log(
                f"[comfy-env] NOTE: {name} has comfy-kitchen {w_kitchen}, host "
                f"has {h_kitchen}. They share one ComfyUI tree, so a symbol the "
                f"host's version expects may be absent in the worker's."
            )
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
    # Dead worker's overhead died with it; a retained entry would book ~1 GB
    # of phantom scratch per crash in a restart loop.
    _OVERHEAD_REPORTS.pop(key, None)
    # Its reserve high-water goes too, and this is the one place a SHRINK is
    # allowed: the process is gone, so the memory is provably back.
    _forget_reserve(key)
    _LAST_ACTIVITY.pop(key, None)
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
    report = getattr(worker, "_last_vram_report", None)
    if not report:
        return
    worker._last_vram_report = None

    _mode = os.environ.get(state_sync.RESIDENCY_ENV_VAR, "boundary").lower()
    if _mode not in ("off", "command", "0", "false"):
        census = report.get("residency")
        if census:
            live = {
                mid: p
                for mid, p in list(_WORKER_PATCHERS.get(str(env_dir), {}).items())
                if getattr(p, "_worker_generation", generation) == generation
            }
            # Whole census under one _mem_lock hold: it is dict math over a
            # handful of entries, microseconds, zero IPC. Per-entry holds
            # would let a concurrent echo interleave mid-census and produce
            # mixed-epoch state. The seq guard inside apply_residency IS the
            # in-critical-section recheck once every competing writer
            # serializes on this lock.
            with worker._mem_lock:
                state_sync.apply_residency(live, census, log=_log)

    # Pin census: what the reclaim path and the regression line read.
    _pin = report.get("pinned")
    if _pin is not None:
        _pin_ingest(str(env_dir), _pin)

    # Measured VRAM overhead: allocator bytes beyond registered residency
    # (cast buffers, cache). REPLACE on arrival order; self-measured
    # in-frame, so no peak is needed and stale-HIGH while idle over-books,
    # the safe direction.
    _ov = report.get("overhead")
    if _ov is not None:
        _OVERHEAD_SEQ += 1
        state_sync.update_overhead_reports(
            _OVERHEAD_REPORTS, str(env_dir), _ov, _OVERHEAD_SEQ, log=_log,
            warn_bytes=state_sync.overhead_warn_threshold(_device_total_bytes()))


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
    _deferred_pins = getattr(worker, "_pin_release_deferred", None)
    if _deferred_pins:
        worker._pin_release_deferred = None
        try:
            r = worker.send_command_no_spawn("release_pins",
                                             size=int(_deferred_pins),
                                             lock_timeout=2.0)
            if r == "busy":
                worker._pin_release_deferred = _deferred_pins
            elif isinstance(r, dict):
                _ingest_worker_frames(env_dir, worker, generation)
        except Exception as _ppe:
            _log(f"[comfy-env] deferred pin release failed: {_ppe}")

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

    _note_activity(env_dir)
    # Workers that have been quiet a while give their VRAM back here. This
    # is what replaces the host reaching into a worker to take it.
    _release_idle_workers()

    # Republish the reserve now that this boundary's residency has landed.
    # Grow only: a worker whose ledger dropped may not have released yet, and
    # a reserve that falls before the memory is back is space the host loads
    # straight into. Shrinking waits for _forget_reserve on a real removal.
    _publish_reserve(shrink_allowed=False)

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
