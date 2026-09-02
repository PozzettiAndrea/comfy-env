"""Typed error translation at the worker IPC frontier.

A worker's OOM crosses the socket as a generic WorkerError, so ComfyUI's
recovery arms (execution.py:641 for OOM, :619 for interrupt) never fire for
isolated nodes. The worker stamps ``error_kind`` on its error frames using its
OWN live exception object (``mm.is_oom(e)`` / an isinstance check); this module
restores the host-side type from that verdict, and only from that verdict.

The translation site matters as much as the type: ``torch.OutOfMemoryError``
IS a RuntimeError, and ``_call_in_worker`` tears the worker down on
``except (RuntimeError, ConnectionError)``. Translation therefore happens in
an ``except WorkerError`` handler AFTER that clause has already declined the
exception, never inside the try body.

Rules, each a debate verdict from the design panel:

* Raise ComfyUI's REAL exception classes, never a comfy-env subclass:
  ``full_type_name(typ)`` (execution.py:630) ships the class name to the
  frontend, and a ``comfy_env.*`` name there breaks node invisibility.
* Import-or-passthrough: if torch (or comfy) is absent on the host, return
  the WorkerError unchanged. Never synthesize a stand-in class; comfy's
  ``OOM_EXCEPTION = Exception`` fallback makes ``is_oom`` true for EVERY
  exception on a torch-less host, so leaning on it would misfire.
* The verdict comes only from ``error_kind``. No verdict in this module may
  ever be derived from message text (a filename can contain "out of memory";
  a real OOM can be localized).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional


def _resolve_oom() -> Optional[type]:
    """The host's real CUDA OOM class, or None to pass through."""
    try:
        import torch
        return torch.cuda.OutOfMemoryError
    except Exception:
        return None


def _resolve_interrupt() -> Optional[type]:
    """ComfyUI's own interrupt exception, or None to pass through."""
    try:
        from comfy.model_management import InterruptProcessingException
        return InterruptProcessingException
    except Exception:
        return None


#: Closed vocabulary. Keys are stable wire tokens, never Python class paths:
#: the host must not import a class by a name that arrived over the socket.
DEFAULT_REGISTRY: Dict[str, Callable[[], Optional[type]]] = {
    "oom": _resolve_oom,
    "interrupt": _resolve_interrupt,
}


def translate_error(exc, *, registry=None):
    """Return the exception the host should raise for a worker error.

    Returns, never raises. Given a WorkerError whose ``error_kind`` names a
    registry entry that resolves, returns the resolved class built from
    ``str(exc)`` (which embeds the worker traceback) with ``__cause__``
    chained to the original and ``comfy_env_worker`` carrying the worker's
    allocator numbers. In every other case (no kind, unknown kind, resolver
    unavailable, construction failure) returns ``exc`` unchanged, which is
    exactly today's behavior.
    """
    if registry is None:
        registry = DEFAULT_REGISTRY
    kind = getattr(exc, "error_kind", None)
    resolver = registry.get(kind) if kind is not None else None
    if resolver is None:
        return exc
    try:
        cls = resolver()
        if cls is None:
            return exc
        translated = cls(str(exc))
    except Exception:
        return exc
    translated.__cause__ = exc
    stats = getattr(exc, "oom_stats", None)
    if stats is not None:
        translated.comfy_env_worker = stats
    return translated
