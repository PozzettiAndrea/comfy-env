"""A stand-in for ComfyUI's ``server`` module, for processes that have none.

Staged beside the worker program and beside the metadata scan, and installed
into ``sys.modules["server"]`` before anything a pack could import. Two
reasons it is unconditional rather than a fallback:

* The real ``server.py`` imports aiohttp at line 32. A lean pack env does not
  have it, so ``from server import PromptServer`` at the top of a pack's
  module (154 of 493 surveyed packs) fails before any node is defined, and
  the error reads like a broken ComfyUI install.
* Even where aiohttp happens to be present, nobody constructs a
  ``PromptServer`` in a worker, so ``PromptServer.instance`` is ``None`` and
  every call on it raises anyway.

The surface is what the route-free importers in that survey touch, and no
more: ``send_sync`` (26 packs), ``client_id`` (7), ``send_progress_text``
(6) on the instance, and ``BinaryEventTypes`` at module level so the
``from server import BinaryEventTypes`` line works. Anything else raises an
AttributeError that says what is and is not forwarded, because the honest
answer for HTTP routes (123 of the 154 importers), the prompt queue and
prompt handlers is that they run in the host and cannot be forwarded to a
process the host talks to over a socket.

``send_sync`` is a fire-and-forget note handed to whatever the process
plugged in as the sender: the worker forwards it to the host on the same
channel as progress, the metadata scan has no sender and drops it. The
value of ``client_id`` is the host's own, shipped per call. ``sid`` keeps
upstream's meaning: ``None`` is a broadcast (229 of 240 surveyed call sites
pass none), anything else means "the client that queued this", which is
the host's ``client_id`` whatever string the pack put there.

Stdlib only; parses on Python 3.10; imports nothing from comfy_env.
"""

import sys
import types

#: Names a pack may use on ``PromptServer.instance`` in an isolated process.
SURFACE = ("send_sync", "send_progress_text", "client_id")

_DOC = "https://docs.comfy-forge.org/comfy-env/deliberately-unsupported/"


class _FallbackBinaryEventTypes:
    """Mirror of ComfyUI's protocol.BinaryEventTypes, used only when the
    real one is not importable (it lives at the ComfyUI root, which is on
    the path in a worker but need not be everywhere the stub is)."""
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2
    TEXT = 3
    PREVIEW_IMAGE_WITH_METADATA = 4


def _binary_event_types():
    try:
        from protocol import BinaryEventTypes as _real  # ComfyUI's, stdlib-only
        if getattr(_real, "UNENCODED_PREVIEW_IMAGE", None) == 2:
            return _real
    except Exception:
        pass
    return _FallbackBinaryEventTypes


class _Instance:
    #: The host's client id for the call in flight; None outside one.
    client_id = None
    #: callable(method, **params) plugged in by the process, or None.
    _sender = None

    def send_sync(self, event, data, sid=None):
        sender = self._sender
        if sender is None:
            return
        sender("send_sync", event=event, data=data, sid=sid)

    def send_progress_text(self, text, node_id, sid=None):
        sender = self._sender
        if sender is None:
            return
        if isinstance(text, (bytes, bytearray)):
            text = bytes(text).decode("utf-8", errors="replace")
        sender("send_progress_text", text=text, node_id=str(node_id), sid=sid)

    def __getattr__(self, name):
        # Only reached for names not defined above.
        raise AttributeError(
            f"PromptServer.instance.{name} is not available in an isolated "
            f"node process. comfy-env forwards {', '.join(SURFACE)}; HTTP "
            f"routes, the prompt queue and prompt handlers run in the host "
            f"and are not forwarded. See {_DOC}")


class PromptServer:
    """The class a pack imports. Never constructed in an isolated process;
    ``instance`` exists from import time, as it does in ComfyUI (the server
    is built before custom nodes load)."""
    instance = _Instance()


def install(modules=None):
    """Put the stand-in under ``sys.modules["server"]`` and return it.
    Idempotent; always replaces whatever is there."""
    modules = sys.modules if modules is None else modules
    mod = types.ModuleType("server")
    mod.__doc__ = "comfy-env stand-in for ComfyUI's server module (see server_stub.py)"
    mod.PromptServer = PromptServer
    mod.BinaryEventTypes = _binary_event_types()
    mod.__comfy_env_stub__ = True
    modules["server"] = mod
    return mod
