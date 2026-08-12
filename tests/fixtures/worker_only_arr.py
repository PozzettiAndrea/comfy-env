"""Fixture serializer module the PARENT deliberately never imports.

Unlike worker_only_type (primitive payload), this type carries a numpy
array through recurse -- so the parent's held OpaquePayload contains a
real shared-memory frame that must be MATERIALIZED (copied/owned) to
survive the worker restarting or freeing its blocks.
"""

try:
    from comfy_env.isolation.workers import _ipc_shared as ipc
except ImportError:
    import _ipc_shared as ipc


class WorkerOnlyArr:
    def __init__(self, data):
        self.data = data  # numpy array


ipc.register_serializer(
    "WorkerOnlyArr",
    lambda obj, recurse: {"data": recurse(obj.data)},
    lambda payload, recurse: WorkerOnlyArr(recurse(payload["data"])),
)
