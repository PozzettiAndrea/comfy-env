"""Fixture serializer module the PARENT deliberately never imports."""

try:
    from comfy_env.isolation.workers import _ipc_shared as ipc
except ImportError:
    import _ipc_shared as ipc


class WorkerOnly:
    def __init__(self, secret):
        self.secret = secret


ipc.register_serializer(
    "WorkerOnly",
    lambda obj, recurse: {"secret": obj.secret},
    lambda payload, recurse: WorkerOnly(payload["secret"]),
)
