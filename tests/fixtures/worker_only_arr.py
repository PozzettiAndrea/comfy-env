"""Fixture serializer file for the array-bearing worker-only type.

Unlike WorkerOnly (primitive payload), WorkerOnlyArr carries a numpy
array through recurse -- so a parent without the class holds an
OpaquePayload containing a real shared-memory frame that must be
MATERIALIZED (copied/owned) to survive the worker restarting.
"""

try:
    from comfy_env.isolation.workers import _ipc_shared as ipc
except ImportError:
    import _ipc_shared as ipc


def _serialize(obj, recurse):
    return {"data": recurse(obj.data)}


def _deserialize(payload, recurse):
    from worker_only_class import WorkerOnlyArr
    return WorkerOnlyArr(recurse(payload["data"]))


try:  # register deserialize only where the class resolves
    from worker_only_class import WorkerOnlyArr  # noqa: F401
    _DESER = _deserialize
except ImportError:
    _DESER = None

ipc.register_serializer("WorkerOnlyArr", _serialize, _DESER)
