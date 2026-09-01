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
    try:
        import comfy.model_management as mm

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


def release_node_boundary(log=None) -> None:
    """The worker's equivalent of ComfyUI's per node release.

    ComfyUI runs this in a ``finally`` around every node (``execution.py:550``),
    gated on ``aimdo_enabled``. A worker never reaches that code, so without
    this hook an aimdo enabled worker would allocate cast buffers, CUDA graph
    pools and cross step tensors with nothing to ever free them. Never raises:
    a release that fails must not fail the node that just succeeded.
    """
    # Hot path: this runs in a finally around every worker node call, so the
    # module is resolved from sys.modules rather than re-imported. When aimdo is
    # off (CPU workers, failed init, or COMFY_ENV_WORKER_AIMDO=0), this costs
    # one dict lookup and a getattr.
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
