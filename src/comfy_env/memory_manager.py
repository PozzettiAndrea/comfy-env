"""Which ComfyUI memory manager this process resolved to, and why.

ComfyUI has had two managers for model weights since ``f8acd9c4`` (2026-01-31):

* **aimdo** (``comfy-aimdo``), the default, which pages weights in per layer.
* **the ledger** (``ModelPatcher`` + ``current_loaded_models``), which evicts
  whole models.

In ComfyUI the selection lives entirely in ``main.py``:
``comfy.memory_management.aimdo_enabled`` defaults to ``False``
(``memory_management.py:173``) and ComfyUI sets it ``True`` in exactly one place,
``main.py:300``. The only other writer anywhere is ``maybe_enable_aimdo`` in this
module, which runs at worker start and can be disabled with
``COMFY_ENV_WORKER_AIMDO=0``.
An isolation worker never runs ``main.py`` and parses ComfyUI's args from an
empty argv, so left alone it would resolve to the ledger whatever the host is
doing. ``maybe_enable_aimdo`` closes that gap at worker start, and
``_report_memory_manager`` in the pool reports whichever way it resolved.

Two consequences worth knowing before changing anything here:

* Whether a node gets aimdo is decided by whether its pack was *isolated*, and
  that is a per-pack decision (``wrap.py`` falls back to in-process import in
  five separate cases). Two packs in one ComfyUI run can therefore resolve
  differently, with nothing announcing it.
* A **CPU** load device resolves to the ledger correctly and permanently.
  ``ModelPatcherDynamic._vbar_get`` returns ``None`` for ``torch.device("cpu")``
  and ``partially_unload`` asserts a non-CPU device, so aimdo has no CPU path at
  all. That is the one difference that is a fact rather than an accident.

This module is a leaf: it imports nothing from ``comfy_env`` so that both the
parent and the worker can use it without touching the isolation layering.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

#: Whether to initialise aimdo inside a worker. **The worker follows the
#: host**: the parent exports "1" when its own ``aimdo_enabled`` is True and
#: "0" otherwise, so a worker runs the same manager as the process that spawned
#: it. A host on the ledger is there deliberately (flags, unsupported GPU, old
#: torch), and a worker second-guessing that would create the opposite mismatch
#: to the one this closes. Operators can force either way by setting the
#: variable themselves; unset (no parent signal) means the ledger, which is the
#: behaviour before this existed.
#:
#: Every failure path falls through to the ledger. The node boundary release
#: (``release_node_boundary``) ships with this: without it an aimdo worker
#: would allocate cast buffers with nothing to free them.
ENABLE_ENV_VAR = "COMFY_ENV_WORKER_AIMDO"

#: Values that mean OFF. Anything else that is non-empty means ON.
DISABLE_VALUES = ("0", "false", "no", "off")

#: The aimdo version the parent resolved, exported to the worker so it never
#: has to guess. A worker that disagrees refuses to initialise.
VERSION_ENV_VAR = "COMFY_ENV_AIMDO_VERSION"

#: Per device VRAM headroom in bytes, as the parent resolved it from
#: ``--vram-headroom``. A worker paging with no headroom against a host that
#: reserves some is the admission problem this seam exists to prevent, so the
#: worker mirrors the parent rather than defaulting to zero.
HEADROOM_ENV_VAR = "COMFY_ENV_AIMDO_HEADROOM"

#: The parent's ``simple_vram_headroom`` (derived from ``--reserve-vram`` at
#: ``main.py:71``) and its NVML pressure choice (``--disable-nvml-pressure``).
#: Mirrored so an enabled worker pages under the same policy as its host.
SIMPLE_HEADROOM_ENV_VAR = "COMFY_ENV_AIMDO_SIMPLE_HEADROOM"
NVML_ENV_VAR = "COMFY_ENV_AIMDO_NVML"

LEDGER = "ledger"
AIMDO = "aimdo"

#: Why the last maybe_enable_aimdo() attempt did not enable. None means it
#: enabled, or was never attempted. describe() surfaces this so the parent can
#: log the true reason instead of deriving a guess from state.
_ENABLE_ERROR: Optional[str] = None


_AIMDO_VERSION: Optional[str] = None
_AIMDO_VERSION_READ = False


def aimdo_version() -> Optional[str]:
    """Installed ``comfy-aimdo`` version, or None when it is not present.

    Importable is not the same as usable: the wheel being present says nothing
    about whether ``control.init`` ever ran. In practice every worker env that
    can import ``comfy.model_management`` has the wheel, because that module
    imports ``comfy_aimdo.host_buffer`` unconditionally.
    """
    global _AIMDO_VERSION, _AIMDO_VERSION_READ
    if _AIMDO_VERSION_READ:
        return _AIMDO_VERSION
    # Cached: importlib.metadata.version walks sys.path, and describe() is on a
    # path that can be called per request.
    _AIMDO_VERSION_READ = True
    try:
        from importlib.metadata import version

        _AIMDO_VERSION = version("comfy-aimdo")
    except Exception:
        _AIMDO_VERSION = None
    return _AIMDO_VERSION


def describe() -> Dict[str, Any]:
    """Report the manager this process resolved to.

    Safe to call from anywhere, including before ComfyUI is imported. Never
    raises: an unknown answer is reported as such rather than thrown, because
    the whole point is to be readable when something is already wrong.
    """
    info: Dict[str, Any] = {
        "manager": LEDGER,
        "aimdo_version": aimdo_version(),
        "aimdo_importable": False,
        "reason": "comfy.memory_management not imported",
    }

    try:
        import comfy.memory_management as mm
    except Exception as exc:
        info["reason"] = f"comfy.memory_management unavailable: {type(exc).__name__}"
        return info

    enabled = bool(getattr(mm, "aimdo_enabled", False))
    info["manager"] = AIMDO if enabled else LEDGER

    try:
        import comfy_aimdo.control as control  # noqa: F401

        info["aimdo_importable"] = True
        info["aimdo_initialised"] = getattr(control, "lib", None) is not None
    except Exception:
        info["aimdo_importable"] = False
        info["aimdo_initialised"] = False

    if enabled:
        info["reason"] = "aimdo_enabled is True"
    elif _ENABLE_ERROR is not None:
        info["reason"] = _ENABLE_ERROR
    elif not info["aimdo_importable"]:
        info["reason"] = "comfy-aimdo is not installed in this environment"
    else:
        info["reason"] = "installed but never initialised (main.py was not run)"
    if _ENABLE_ERROR is not None:
        info["enable_error"] = _ENABLE_ERROR
    return info


def summary_line(prefix: str = "") -> str:
    """One line an operator can grep for, describing this process."""
    info = describe()
    version = info.get("aimdo_version") or "none"
    if info["manager"] == AIMDO:
        return f"{prefix}memory manager: aimdo {version}"
    return f"{prefix}memory manager: legacy ledger ({info['reason']})"


def _cuda_devices() -> list:
    """Device indices to hand aimdo, empty when there are none.

    Asks ComfyUI rather than torch directly, so ``--cuda-device`` and
    ``--default-device`` selection is honoured exactly as ``main.py:277`` does.
    Falls back to torch only when ComfyUI is not importable.

    Gated on real devices rather than on ``COMFY_CPU``: that variable is set
    only when the *parent* ran ``--cpu`` and says nothing about this process.
    """
    # Only consult ComfyUI if it is ALREADY imported: importing
    # comfy.model_management here would pull in comfy_aimdo.host_buffer, whose
    # module level `lib = control.lib` binds at import time. Doing that before
    # control.init() freezes lib=None in every shim, permanently: the flag then
    # says aimdo while the first real model load crashes with
    # "'NoneType' object has no attribute 'hostbuf_allocate'". Observed live.
    mm = sys.modules.get("comfy.model_management")
    if mm is not None:
        try:
            return [d.index for d in mm.get_all_torch_devices() if d.type == "cuda"]
        except Exception:
            pass
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def maybe_enable_aimdo(log=None) -> bool:
    """Initialise aimdo in this process. On by default, opt out with the env var.

    Returns True only when aimdo is fully live afterwards. Every failure path
    calls ``control.deinit()`` and leaves ``aimdo_enabled`` False, because a
    partially initialised aimdo is not the same state as a never initialised
    one: ``init_devices`` returns False without calling ``plat_cleanup`` when
    ``plat_init`` fails, and the library stays dlopened ``RTLD_GLOBAL`` for the
    life of the process.

    Known gaps, stated rather than hidden. Read all of them before enabling:

    * ``main.py`` calls ``control.init`` before torch is imported, and a worker
      cannot reproduce that ordering because torch is already loaded here.
      ``control.py`` dlopens ``RTLD_GLOBAL``.
    * The aimdo log level (``main.py:283-297``) is not replicated, so an
      enabled worker pages silently.

    The first has now been observed working (one full eviction round trip on a
    live worker); the others remain open. ``COMFY_ENV_WORKER_AIMDO=0`` turns the
    whole thing off without a code change.
    """

    global _ENABLE_ERROR

    def _fail(reason: str) -> bool:
        global _ENABLE_ERROR
        _ENABLE_ERROR = reason
        if log is not None:
            log(f"[worker] aimdo not enabled: {reason}")
        return False

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)

    flag = os.environ.get(ENABLE_ENV_VAR, "").strip().lower()
    if not flag or flag in DISABLE_VALUES:
        return _fail(
            "no parent signal" if not flag else "disabled by COMFY_ENV_WORKER_AIMDO"
        )

    devices = _cuda_devices()
    if not devices:
        return _fail("no CUDA device in this process")

    control = None
    try:
        import comfy_aimdo.control as control

        installed = aimdo_version()
        wanted = os.environ.get(VERSION_ENV_VAR, "")
        if wanted and installed != wanted:
            raise RuntimeError(
                f"aimdo version skew: worker has {installed}, parent has {wanted}"
            )

        try:
            simple = os.environ.get(SIMPLE_HEADROOM_ENV_VAR)
            simple_headroom = int(simple) if simple else None
        except ValueError:
            simple_headroom = None
        nvml = os.environ.get(NVML_ENV_VAR, "1") not in ("0", "false")
        try:
            control.init(
                simple_vram_headroom=simple_headroom, nvml_pressure=nvml
            )
        except TypeError:
            try:
                control.init(simple_vram_headroom=simple_headroom)
            except TypeError:
                control.init()

        # Mirror the parent's per device headroom. Zero here would let a worker
        # page right up against a card the host believes it has reserved room on.
        try:
            headroom = int(os.environ.get(HEADROOM_ENV_VAR, "0"))
        except ValueError:
            headroom = 0
        if not control.init_devices((index, headroom) for index in devices):
            raise RuntimeError("comfy_aimdo.control.init_devices returned False")
        _log(f"[worker] aimdo devices={devices} headroom={headroom} bytes")

        # Every comfy_aimdo shim does `lib = control.lib` at module import AND
        # configures ctypes argtypes/restypes under `if lib is not None:`. A
        # shim imported before control.init() therefore holds lib=None with NO
        # signatures. Setting `.lib` by hand is NOT enough: the functions then
        # default to int returns, 64-bit pointers truncate, and the first
        # hostbuf_free segfaults (observed live, HostBuffer.__del__). Reload
        # instead: it re-executes the module top with control.lib now set, into
        # the SAME module object, so existing references stay valid and the
        # signatures are configured exactly as an on-time import would have.
        import ctypes
        import importlib

        for _name in ("host_buffer", "model_mmap", "model_vbar",
                      "vram_buffer", "torch"):
            _mod = sys.modules.get(f"comfy_aimdo.{_name}")
            if _mod is not None and getattr(_mod, "lib", None) is None:
                importlib.reload(_mod)
        import comfy_aimdo.host_buffer as _hb

        # Verify the WIRING, not just the handle: an unconfigured restype is
        # precisely the state that segfaults later.
        if _hb.lib is None or _hb.lib.hostbuf_allocate.restype is not ctypes.c_void_p:
            raise RuntimeError(
                "comfy_aimdo shims are half wired after control.init "
                "(lib missing or ctypes signatures unconfigured); refusing, "
                "because this state segfaults on the first model load"
            )

        import comfy.memory_management as mm
        import comfy.model_patcher as model_patcher

        model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
        mm.aimdo_enabled = True  # last: everything else must already be true
        _ENABLE_ERROR = None
        _log(f"[worker] aimdo enabled ({installed})")
        return True
    except Exception as exc:
        _ENABLE_ERROR = str(exc)
        _log(f"[worker] aimdo not enabled: {exc}")
        if control is not None:
            try:
                control.deinit()
            except Exception:
                pass
        try:
            import comfy.memory_management as mm

            mm.aimdo_enabled = False
        except Exception:
            pass
        return False


#: Cast-buffer reset epoch for non-aimdo workers. The initial sentinel never
#: equals a prompt gen, so the first call always resets.
_CAST_EPOCH = [object()]


def cast_epoch_boundary(prompt_gen, log=None) -> None:
    """Reset STREAM_CAST_BUFFERS when the PROMPT epoch changes. Runs at the
    START of a worker request, so the new prompt begins clean instead of
    inheriting the previous prompt's buffers for its first node.

    Why this exists: both upstream's reset and release_node_boundary were
    aimdo gated, so a non-aimdo worker on the lowvram cast path ratcheted
    STREAM_CAST_BUFFERS to NUM_STREAMS x its largest-ever casted layer for
    the life of the process (measured: 2 x 512 MiB held through unloads and
    four small-model nodes needing 256 MiB). Epoch scoping bounds the
    ratchet to one prompt with zero intra-prompt realloc churn; a missing
    token (host patch off or pre-first-prompt) degrades to a reset per call,
    the safe default. Safe at call start: the worker is idle between
    requests, no forward is in flight. ``reset_cast_buffers`` is safe
    non-aimdo (its aimdo-only branch is gated on ``is_dynamic()``, False for
    base ModelPatcher) and idempotent against the aimdo per-node reset in
    release_node_boundary. Never raises."""
    epoch_changed = prompt_gen is None or prompt_gen != _CAST_EPOCH[0]
    _CAST_EPOCH[0] = prompt_gen
    if not epoch_changed:
        return
    try:
        import comfy.model_management as model_management

        model_management.reset_cast_buffers()
    except Exception as exc:
        if log is not None:
            log(f"[worker] cast epoch reset failed: {exc}")


def release_node_boundary(log=None) -> None:
    """The worker's equivalent of ComfyUI's per node release.

    ComfyUI runs this in a ``finally`` around every node (``execution.py:550``),
    gated on ``aimdo_enabled``. A worker never reaches that code, so without
    this hook an aimdo enabled worker would allocate cast buffers, CUDA graph
    pools and cross step tensors with nothing to ever free them. The
    non-aimdo cast-buffer ratchet is handled separately, per prompt, by
    ``cast_epoch_boundary`` at request start. Never raises: a release that
    fails must not fail the node that just succeeded.
    """
    # Hot path: this runs in a finally around every worker node call, so the
    # module is resolved from sys.modules rather than re-imported. When aimdo
    # is off this costs one dict lookup and a getattr.
    mm = sys.modules.get("comfy.memory_management")
    if mm is None or not getattr(mm, "aimdo_enabled", False):
        return

    try:
        import comfy.model_management as model_management

        model_management.reset_cast_buffers()
    except Exception as exc:  # pragma: no cover - depends on the ComfyUI tree
        if log is not None:
            log(f"[worker] reset_cast_buffers failed: {exc}")

    for module_name, attr in (
        ("comfy.model_prefetch", "cleanup_prefetch_queues"),
        ("comfy_aimdo.model_vbar", "vbars_reset_watermark_limits"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            getattr(module, attr)()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Pinned RAM budget (worker side)
# ---------------------------------------------------------------------------

class _PinErrorCounter(logging.Filter):
    """Counts upstream's "Pin error." warnings (model_management.py:1640).

    Upstream keeps no counter, and a clamped worker fails by warning and
    continuing, so this filter is the only machine-readable signal that a
    grant went too low. Telemetry only: nothing anywhere derives a VERDICT
    from this text. Always returns True (never suppresses the record)."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def filter(self, record):
        try:
            if record.getMessage().startswith("Pin error"):
                self.count += 1
        except Exception:
            pass
        return True


_PIN_ERROR_COUNTER: Optional[_PinErrorCounter] = None
_PIN_STATE_SEQ = 0


def install_pin_error_counter() -> None:
    """Idempotent. Attached to the ROOT logger because upstream calls
    ``logging.warning`` directly (root-level records are the only ones a
    root filter sees)."""
    global _PIN_ERROR_COUNTER
    if _PIN_ERROR_COUNTER is None:
        _PIN_ERROR_COUNTER = _PinErrorCounter()
        logging.getLogger().addFilter(_PIN_ERROR_COUNTER)


def pin_state() -> Optional[Dict[str, Any]]:
    """The worker's five-field pin scalar for ready frames and RPC requests.

    Hot frames carry only the bare ``_pinned`` int; this richer shape rides
    the low-frequency channels. None when comfy is not imported (report
    nothing rather than a fabricated zero)."""
    global _PIN_STATE_SEQ
    mm = sys.modules.get("comfy.model_management")
    if mm is None:
        return None
    _PIN_STATE_SEQ += 1
    return {
        "pid": os.getpid(),
        "total_pinned": int(getattr(mm, "TOTAL_PINNED_MEMORY", 0)),
        "max_pinned": int(getattr(mm, "MAX_PINNED_MEMORY", 0)),
        "pin_errors": _PIN_ERROR_COUNTER.count if _PIN_ERROR_COUNTER else 0,
        "pins_evicted_bytes": _PIN_EVICTED["bytes"],
        "pins_evicted_active_bytes": _PIN_EVICTED["active_bytes"],
        "pin_churn": _PIN_EVICTED["churn"],
        "seq": _PIN_STATE_SEQ,
    }


def total_pinned() -> Optional[int]:
    """The bare census scalar for hot frames. None when comfy is absent."""
    mm = sys.modules.get("comfy.model_management")
    if mm is None:
        return None
    try:
        return int(getattr(mm, "TOTAL_PINNED_MEMORY", 0))
    except Exception:
        return None


def apply_pin_budget(grant=None, headroom=None, log=None) -> bool:
    """Apply a parent pin grant and headroom mirror. CLAMP ONLY, by contract:

    * No-op when the local ``MAX_PINNED_MEMORY`` is already <= 0. A mirrored
      ``--disable-pinned-memory`` (or a platform that never enabled pinning)
      is terminal: host intent outranks a budget, and re-enabling pinning
      here would also strand registrations, because ``unpin_memory``
      early-returns on ``MAX <= 0``.
    * ``MAX = max(min(local, grant), TOTAL_PINNED)``: the grant can only
      LOWER the ceiling, and never below what this process already holds
      (a ceiling below current TOTAL makes the registration check at
      model_management.py:739 a permanent shortfall, evicting forever).
    * A ``-1`` (or any <= 0) grant is the disabled sentinel: no-op.

    ``headroom`` mirrors the host's RAM_CACHE_HEADROOM by direct assignment
    on ``comfy.memory_management`` (:176). Deliberately NOT via
    ``set_ram_cache_release_state``, which would also stamp a None callback;
    the only upstream setter (execution.py:748) never runs in a worker.

    What the mirrored headroom actually buys here, stated honestly: the
    "RAM cache" behind the release callback is the PromptExecutor's OUTPUT
    cache, an object a worker does not have, and ``extra_ram_release`` is a
    verified no-op when the callback is None -- so on those two call sites
    the mirrored value is inert in workers, BY DESIGN (a worker callback
    would have nothing safe to release that full_release and the pin ladder
    do not already cover, and firing gc on every pressured pin would thrash
    the allocator pinning depends on). The mirror's REAL consumer is the
    pin floor: ``ensure_pin_budget`` (model_management.py:720) reads
    ``RAM_CACHE_HEADROOM / 2`` on every hostbuf pin, so the mirror stops a
    worker from pinning into RAM the host reserved for its cache.
    Returns True if anything changed. Never raises.
    """
    _log = log or (lambda *_: None)
    changed = False
    try:
        mm = sys.modules.get("comfy.model_management")
        if mm is not None and grant is not None:
            grant = int(grant)
            local = int(getattr(mm, "MAX_PINNED_MEMORY", 0))
            if grant > 0 and local > 0:
                held = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0))
                new = max(min(local, grant), held)
                if new != local:
                    mm.MAX_PINNED_MEMORY = new
                    changed = True
                    _log(f"[worker] pin budget: MAX_PINNED_MEMORY "
                         f"{local / 1e9:.2f}GB -> {new / 1e9:.2f}GB "
                         f"(grant {grant / 1e9:.2f}GB, held {held / 1e9:.2f}GB)")
    except Exception as exc:
        _log(f"[worker] pin budget apply failed: {exc}")
    try:
        cm = sys.modules.get("comfy.memory_management")
        if cm is not None and headroom is not None \
                and hasattr(cm, "RAM_CACHE_HEADROOM"):
            headroom = max(0, int(headroom))
            if int(getattr(cm, "RAM_CACHE_HEADROOM", 0)) != headroom:
                cm.RAM_CACHE_HEADROOM = headroom
                changed = True
                _log(f"[worker] pin budget: RAM_CACHE_HEADROOM mirrored to "
                     f"{headroom / 1e9:.2f}GB")
    except Exception as exc:
        _log(f"[worker] pin headroom mirror failed: {exc}")
    return changed


# ---------------------------------------------------------------------------
# /free deep release (worker side)
# ---------------------------------------------------------------------------

def full_release(log=None, _modules=None) -> Dict[str, Any]:
    """Deep release for the host's /free button. Never raises.

    Runs AFTER the host's unload_all_models sweep already detached this
    worker's registered models (which also dropped their pin registrations
    through the real unpatch path), so the ladder here touches only
    rebuildable cache and garbage, never state:

    1. Node-boundary transients (cast buffers, prefetch queues, vbar
       watermarks) -- unconditional, unlike release_node_boundary's aimdo
       gate, because non-aimdo workers keep cast buffers forever (upstream's
       only reset call site is aimdo-gated).
    2. gc.collect: cycle-held tensors return their blocks to the device
       allocator and their pinned buffers to torch's host cache; both later
       rungs depend on running after it.
    3. synchronize + empty_cache + ipc_collect: returns the gc-fed device
       blocks; synchronize also guarantees no in-flight DMA still reads
       pinned buffers before rung 4 sweeps them.
    4. torch._C._host_emptyCache (hasattr-guarded private API): last, so it
       sweeps the host-cache buffers rungs 2 and 3 just returned. This is
       the ~2 GB of retained pinned RSS nothing else gives back.

    Deliberately untouched: the state-sync overflow store (node STATE; loss
    converts the next call into a pointed error), the shm/tensor keepers
    (lifetime belongs to the consumed-ack protocol), and ComfyUI's pin
    registration ledger (free_registrations here would desynchronize
    TOTAL_PINNED accounting; the unload sweep already dropped what should
    drop).

    ``_modules`` defaults to sys.modules; tests inject fakes so bare CI
    drives the whole ladder. Returns a receipt with per-rung outcomes and
    measured before/after numbers, never assumed ones.
    """
    _log = log or (lambda *_: None)
    modules = sys.modules if _modules is None else _modules
    receipt: Dict[str, Any] = {"steps": [], "errors": []}
    torch = modules.get("torch")
    mm = modules.get("comfy.model_management")

    def _measure(suffix: str) -> None:
        try:
            if torch is not None and torch.cuda.is_initialized():
                receipt["reserved_" + suffix] = int(torch.cuda.memory_reserved())
        except Exception:
            pass
        try:
            if mm is not None:
                receipt["pinned_" + suffix] = int(
                    getattr(mm, "TOTAL_PINNED_MEMORY", 0))
        except Exception:
            pass

    def _step(name: str, fn) -> None:
        try:
            fn()
            receipt["steps"].append({"name": name, "ok": True})
        except Exception as exc:
            receipt["steps"].append({"name": name, "ok": False,
                                     "error": str(exc)})
            receipt["errors"].append(f"{name}: {exc}")

    _measure("before")
    if mm is not None and hasattr(mm, "reset_cast_buffers"):
        _step("reset_cast_buffers", mm.reset_cast_buffers)
    _prefetch = modules.get("comfy.model_prefetch")
    if _prefetch is not None and hasattr(_prefetch, "cleanup_prefetch_queues"):
        _step("cleanup_prefetch_queues", _prefetch.cleanup_prefetch_queues)
    _vbar = modules.get("comfy_aimdo.model_vbar")
    if _vbar is not None and hasattr(_vbar, "vbars_reset_watermark_limits"):
        _step("vbars_reset_watermark_limits", _vbar.vbars_reset_watermark_limits)

    import gc
    _step("gc_collect", gc.collect)

    if torch is not None:
        try:
            cuda_up = torch.cuda.is_initialized()
        except Exception:
            cuda_up = False
        if cuda_up:
            _step("synchronize", torch.cuda.synchronize)
            _step("empty_cache", torch.cuda.empty_cache)
            _step("ipc_collect", torch.cuda.ipc_collect)
        _host_empty = getattr(getattr(torch, "_C", None),
                              "_host_emptyCache", None)
        if _host_empty is not None:
            _step("host_empty_cache", _host_empty)

    _measure("after")
    try:
        _log(f"[worker] full release: reserved "
             f"{receipt.get('reserved_before', 0) / 1e9:.2f}GB -> "
             f"{receipt.get('reserved_after', 0) / 1e9:.2f}GB, pinned "
             f"{receipt.get('pinned_before', 0) / 1e9:.2f}GB -> "
             f"{receipt.get('pinned_after', 0) / 1e9:.2f}GB, "
             f"errors={len(receipt['errors'])}")
    except Exception:
        pass
    return receipt


def apply_reserve_bootstrap(value, log=None) -> bool:
    """Mirror the host's resolved EXTRA_RESERVED_VRAM before the first budget
    reply. The reply channel remains sole owner: its plain assignment
    supersedes this on the first shimmed load, so steady state is unchanged;
    what this fixes is the pre-reply window, whose only real consumer is the
    dtype heuristics at model CREATION (maximum_vram_for_weights feeding
    unet_dtype / should_use_fp16 / should_use_bf16) -- permanent choices,
    shifted by the full host reserve (an 8 GB reserve host moves the worker's
    threshold 7.6 GB, flipping fp8 to fp16 in that band).

    Parses inside (a garbage env value must WARN here, not kill the caller's
    whole bootstrap block). ZERO IS A VALUE: upstream honors --reserve-vram 0
    (the guard there is `is not None`), and the reply assigns 0 verbatim, so
    the advance must too -- deliberately DIFFERENT from apply_pin_budget,
    where <= 0 is a disabled sentinel; a "harmonizing" refactor of the two is
    the named enemy. Negatives also cross verbatim (reachable from the CLI:
    the flag is an unbounded float) with one WARN, because clamping only the
    advance would make it disagree with its own settlement. Never raises.
    Returns True if the module value changed.
    """
    _log = log or (lambda *_: None)
    if value is None:
        return False
    try:
        bytes_value = int(str(value).strip())
    except Exception:
        _log(f"[worker] reserve bootstrap: unparseable value {value!r}, ignored")
        return False
    if bytes_value < 0:
        _log(f"[worker] reserve bootstrap: negative margin {bytes_value} "
             f"mirrored verbatim (matches what the budget reply would assign)")
    try:
        mm = sys.modules.get("comfy.model_management")
        if mm is None or not hasattr(mm, "EXTRA_RESERVED_VRAM"):
            return False
        old = mm.EXTRA_RESERVED_VRAM
        if old == bytes_value:
            return False
        mm.EXTRA_RESERVED_VRAM = bytes_value
        _log(f"[worker] comfy margin {bytes_value / 1e9:.2f}GB (host advance, "
             f"was {old / 1e9:.2f}GB upstream default; first budget reply "
             f"supersedes)")
        return True
    except Exception as exc:
        _log(f"[worker] reserve bootstrap failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Prompt-epoch pin marks (worker side)
# ---------------------------------------------------------------------------

def apply_prompt_marks(registry, to_set, to_clear, log=None) -> int:
    """Flip ``current_prompt`` on the worker's dynamic patchers.

    ``registry`` maps model key to patcher. Triple-gated per patcher, so
    ledger-mode workers (base ModelPatcher, no dynamic_pins) and older
    ComfyUIs (no setter) are structurally inert: ``is_dynamic()`` must be
    true AND ``set_in_use_by_current_prompt`` must exist. Never raises;
    returns flips applied. The marks themselves live only in worker memory,
    so a restart clears them for free."""
    _log = log or (lambda *_: None)
    flips = 0
    for keys, value in ((to_set, True), (to_clear, False)):
        for key in keys:
            try:
                p = registry.get(key)
                if p is None or not p.is_dynamic():
                    continue
                setter = getattr(p, "set_in_use_by_current_prompt", None)
                if setter is None:
                    continue
                setter(value)
                flips += 1
            except Exception as exc:
                _log(f"[worker] prompt mark {key}={value} failed: {exc}")
    return flips


#: Eviction counters. pin_errors (the register-failure warning counter) is
#: provably BLIND to churn: the ping-pong loop unregisters and re-registers
#: successfully, logging nothing. These count what actually moved.
_PIN_EVICTED = {"bytes": 0, "active_bytes": 0, "churn": 0}
_EVICTED_EPOCH_KEYS: set = set()
_EVICTION_COUNTERS_INSTALLED = False


def reset_pin_churn_epoch() -> None:
    """Called at a prompt-epoch change: churn counts re-evictions WITHIN one
    prompt (the ping-pong signature), not across prompts."""
    _EVICTED_EPOCH_KEYS.clear()


def install_pin_eviction_counters(log=None) -> None:
    """Wrap comfy.model_management.free_model_pins, the single choke point of
    both pin-eviction paths, with per-victim attribution via
    pinned_memory_size deltas. Idempotent; observability only (the wrapper
    calls the original unconditionally and returns its result verbatim).

    ``active_bytes`` is the fix's own regression signal: after prompt marks
    ship, bytes evicted from a victim whose ``active`` flag was set must be
    ZERO (tier 1 ignoring `active` is the corruption window)."""
    global _EVICTION_COUNTERS_INSTALLED
    if _EVICTION_COUNTERS_INSTALLED:
        return
    mm = sys.modules.get("comfy.model_management")
    if mm is None or not hasattr(mm, "free_model_pins"):
        return
    _EVICTION_COUNTERS_INSTALLED = True
    _log = log or (lambda *_: None)
    _orig = mm.free_model_pins

    def _counted_free_model_pins(size, subsets, current_prompt, active,
                                 registrations=False):
        candidates = []
        before = {}
        try:
            candidates = list(mm.models_for_pin_eviction(
                active, current_prompt=current_prompt))
            for m in candidates:
                try:
                    before[id(m)] = int(m.pinned_memory_size())
                except Exception:
                    pass
        except Exception:
            candidates = []
        freed = _orig(size, subsets, current_prompt, active,
                      registrations=registrations)
        try:
            _PIN_EVICTED["bytes"] += max(0, int(freed))
            for m in candidates:
                b = before.get(id(m))
                if b is None:
                    continue
                try:
                    delta = b - int(m.pinned_memory_size())
                except Exception:
                    continue
                if delta <= 0:
                    continue
                try:
                    st = m.model.dynamic_pins.get(m.load_device) or {}
                    if st.get("active"):
                        _PIN_EVICTED["active_bytes"] += delta
                        _log(f"[worker] pin evict hit ACTIVE model "
                             f"{type(m.model).__name__}: {delta / 1e6:.0f}MB "
                             f"tier=({subsets},{current_prompt},{active})")
                except Exception:
                    pass
                k = id(m.model)
                if k in _EVICTED_EPOCH_KEYS:
                    _PIN_EVICTED["churn"] += 1
                _EVICTED_EPOCH_KEYS.add(k)
        except Exception:
            pass
        return freed

    mm.free_model_pins = _counted_free_model_pins


def release_pins(size, log=None, _modules=None) -> Dict[str, Any]:
    """Release ``size`` bytes of this worker's pinned host RAM, for the
    host's RAM-pressure sweep. Never raises.

    The worker owns its own current_loaded_models and pin ledger, so
    ``mm.free_pins`` runs the SAME eviction ladder a non-isolated node's
    pins face under host pressure (tiers, hysteresis, current_prompt marks
    included). ``torch._C._host_emptyCache`` afterwards returns the unpinned
    buffers from torch's host caching allocator as actual RSS, which is the
    whole point under RAM pressure. Returns a measured receipt.
    """
    _log = log or (lambda *_: None)
    modules = sys.modules if _modules is None else _modules
    receipt: Dict[str, Any] = {"errors": []}
    mm = sys.modules.get("comfy.model_management") if _modules is None \
        else modules.get("comfy.model_management")
    torch = modules.get("torch")
    try:
        if mm is not None:
            receipt["pinned_before"] = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0))
            freed = mm.free_pins(int(size))
            receipt["freed"] = int(freed)
            receipt["pinned_after"] = int(getattr(mm, "TOTAL_PINNED_MEMORY", 0))
        else:
            receipt["errors"].append("no comfy.model_management")
    except Exception as exc:
        receipt["errors"].append(f"free_pins: {exc}")
    try:
        host_empty = getattr(getattr(torch, "_C", None), "_host_emptyCache", None)
        if host_empty is not None:
            host_empty()
    except Exception as exc:
        receipt["errors"].append(f"host_empty_cache: {exc}")
    try:
        _log(f"[worker] pressure pin release: asked {int(size) / 1e9:.2f}GB, "
             f"pinned {receipt.get('pinned_before', 0) / 1e9:.2f}GB -> "
             f"{receipt.get('pinned_after', 0) / 1e9:.2f}GB")
    except Exception:
        pass
    return receipt
