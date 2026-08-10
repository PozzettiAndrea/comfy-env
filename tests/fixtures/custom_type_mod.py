"""Fixture serializer module -- importable on BOTH sides of the boundary."""

try:  # parent process
    from comfy_env.isolation.workers import _ipc_shared as ipc
except ImportError:  # worker process (module copied next to the worker)
    import _ipc_shared as ipc


class ColoredPoint:
    """A custom node-pack data type with its own serialization rule."""

    def __init__(self, xy, color):
        self.xy = xy  # numpy array
        self.color = color


def _ser(obj, recurse):
    return {"xy": recurse(obj.xy), "color": obj.color}


def _deser(payload, recurse):
    return ColoredPoint(recurse(payload["xy"]), payload["color"])


ipc.register_serializer("ColoredPoint", _ser, _deser)
