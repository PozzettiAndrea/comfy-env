"""Contract: the exporter must retain a tensor it exported an IPC handle for.

CUDA IPC inverts the POSIX rule. There is no cross-process refcount: the
EXPORTER must keep the allocation alive until the importer maps it. But
`_serialize_cuda_ipc` returns base64 handles and no tensor reference, so on the
`"received from another process"` branch -- where it must clone before it can
re-export -- the clone had ZERO references the moment the function returned.
The worker then imported a handle onto freed device memory.

`tensor_utils`' 30s keeper masked this, but only for callers that came through
`prepare_for_ipc_recursive` (the two proxy bodies in metadata.py). Anything
else reaching the serializer -- call_module from the pool/route path, a tensor
nested inside a pack object -- had no keeper at all.

Runs on CPU with a stubbed `reduce_tensor`: `_serialize_cuda_ipc` never checks
`is_cuda`, so the branch is reachable without a GPU. That matters, because the
GPU lane that would otherwise cover this does not exist -- `pytest.mark.gpu` is
declared in pyproject.toml and applied to zero tests.
"""

import pytest

torch = pytest.importorskip("torch")

from comfy_env.isolation.workers import _ipc_parent  # noqa: E402


def _fake_reduce_args():
    """The 15-tuple shape `_serialize_cuda_ipc` unpacks from reduce_tensor."""
    return (
        None,             # 0  cls
        [2, 2],           # 1  tensor_size
        [2, 1],           # 2  tensor_stride
        0,                # 3  tensor_offset
        None,             # 4
        torch.float32,    # 5  dtype
        0,                # 6  device_idx
        b"handle-bytes",  # 7  handle
        16,               # 8  storage_size
        0,                # 9  storage_offset
        False,            # 10 requires_grad
        b"refcount",      # 11 ref_counter_handle
        0,                # 12 ref_counter_offset
        None,             # 13 event_handle
        False,            # 14 event_sync_required
    )


def test_exported_clone_is_retained(monkeypatch):
    """The regression: the clone must outlive the call that exported it."""
    import torch.multiprocessing.reductions as reductions

    original = torch.ones(2, 2)
    calls = {"n": 0}
    exported = []

    def fake_reduce(t):
        calls["n"] += 1
        if calls["n"] == 1:
            # Force the branch that must clone before it can re-export.
            raise RuntimeError("received from another process")
        exported.append(t)
        return (None, _fake_reduce_args())

    monkeypatch.setattr(reductions, "reduce_tensor", fake_reduce)

    keeper = _ipc_parent._parent_tensor_keeper
    before = len(keeper._keeper)

    meta = _ipc_parent._serialize_cuda_ipc(original)

    assert meta["__type__"] == "CudaIPC"
    assert len(exported) == 1, "expected exactly one successful export"
    clone = exported[0]
    assert clone is not original, "the branch under test must clone"

    kept = [obj for _ts, obj in list(keeper._keeper)[before:]]
    assert any(obj is clone for obj in kept), (
        "the exported clone is not retained -- the importer would map freed memory"
    )


def test_retention_survives_dropping_every_local_reference(monkeypatch):
    """What the bug actually looked like: nothing else holds the clone."""
    import torch.multiprocessing.reductions as reductions

    calls = {"n": 0}
    ref = {}

    def fake_reduce(t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("received from another process")
        # Weakly witness the clone so we can tell if it was collected.
        ref["w"] = t
        return (None, _fake_reduce_args())

    monkeypatch.setattr(reductions, "reduce_tensor", fake_reduce)

    _ipc_parent._serialize_cuda_ipc(torch.ones(2, 2))

    # The caller's tensor is gone; only the keeper can be holding the clone.
    import gc

    gc.collect()
    kept = [obj for _ts, obj in list(_ipc_parent._parent_tensor_keeper._keeper)]
    assert any(obj is ref["w"] for obj in kept), (
        "clone was not retained by the transport after the call returned"
    )


def test_non_cloning_path_is_unaffected(monkeypatch):
    """A tensor that reduces cleanly must not be cloned, and needs no rescue."""
    import torch.multiprocessing.reductions as reductions

    original = torch.ones(2, 2)
    seen = []

    def fake_reduce(t):
        seen.append(t)
        return (None, _fake_reduce_args())

    monkeypatch.setattr(reductions, "reduce_tensor", fake_reduce)
    _ipc_parent._serialize_cuda_ipc(original)

    assert seen == [original], "clean path must export the original, not a clone"
