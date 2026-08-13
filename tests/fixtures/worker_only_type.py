"""Fixture serializer file the PARENT can read but not fully use.

Follows the ADR-0015 shape: self-contained top level, lazy class import
inside deserialize, deserialize registered only where the class module
imports (the parent has no fixtures dir on sys.path, so it registers
deserialize=None and holds materialized OpaquePayload receipts).
"""

try:
    from comfy_env.isolation.workers import _ipc_shared as ipc
except ImportError:
    import _ipc_shared as ipc


def _serialize(obj, recurse):
    return {"secret": obj.secret}


def _deserialize(payload, recurse):
    from worker_only_class import WorkerOnly
    return WorkerOnly(payload["secret"])


try:  # register deserialize only where the class resolves
    from worker_only_class import WorkerOnly  # noqa: F401
    _DESER = _deserialize
except ImportError:
    _DESER = None

ipc.register_serializer("WorkerOnly", _serialize, _DESER)
