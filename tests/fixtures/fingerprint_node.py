"""Node classes for the fingerprint-forwarding contract test.

Deliberately plain: no comfy imports, no torch. Each class is one answer
shape the worker's _handle_fingerprint has to map onto the reply contract
(a primitive, a non-primitive, NaN, a raise, a coroutine, a subset
signature, hidden inputs under V1 and V3 spellings). The V3 shape is
duck-typed on `PREPARE_CLASS_CLONE`, as tests/fixtures/hidden_node.py does
for execute.
"""

import time


def ping():
    """Module-level start call: `send_command_no_spawn` answers "dead" for a
    never-started worker, so the tests start it through call_module."""
    return "pong"


def real_class_hidden():
    """The real V3 class's `hidden` after a fingerprint, to prove the clone
    was per-call and nothing was written onto the shared class."""
    return V3Shaped.hidden


class Mtime:
    @classmethod
    def IS_CHANGED(cls, path):
        return path


class Everything:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return ",".join(sorted(kwargs))


class Picky:
    """Names one input and takes no catch-all: an extra kwarg is a TypeError
    in the worker, which is exactly what native ComfyUI raises too."""

    @classmethod
    def IS_CHANGED(cls, path):
        return path


class _Opaque:
    pass


class Opaque:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return _Opaque()


class Nan:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class Boom:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        raise ValueError("fingerprint exploded")


class Slow:
    """The real call sleeps so a fingerprint asked meanwhile meets a held
    worker lock and gets "busy" instead of an answer."""

    def run(self, seconds=1.0):
        time.sleep(seconds)
        return ("done",)


class Async:
    @classmethod
    async def IS_CHANGED(cls, **kwargs):
        return "never"


class V1Hidden:
    """Declares UNIQUE_ID under the author's own spelling `node_id`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING",)},
                "hidden": {"node_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, path=None, node_id=None):
        return node_id


class _HiddenHolder:
    def __init__(self, d):
        self.prompt = d.get("PROMPT")
        self.unique_id = d.get("UNIQUE_ID")

    def __getattr__(self, _k):
        return None


class V3Shaped:
    """A V3 node: hidden values arrive on a per-call class clone, and the
    fingerprint reads them off `cls.hidden`."""

    hidden = None

    @classmethod
    def PREPARE_CLASS_CLONE(cls, v3_data):
        clone = type(f"{cls.__name__}Clone", (cls,), {})
        clone.hidden = _HiddenHolder((v3_data or {}).get("hidden_inputs") or {})
        return clone

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return cls.hidden.unique_id
