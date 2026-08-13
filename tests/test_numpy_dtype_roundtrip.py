"""Regression: numpy STRUCTURED dtypes survive the shared-memory boundary.

The shm frame stored the dtype as ``str(arr.dtype)`` and rebuilt it with
``np.dtype(str)``. That is fine for simple dtypes but ``np.dtype()`` cannot
parse the ``str()`` of a structured/record dtype -- e.g. a PLY point cloud
with per-vertex fields ``[('x','<f4'), ... ('pressure_cp','<f4')]`` -- so
deserialization crashed with "data type ... not understood".

A first fix using numpy's descr form was still fragile: JSON collapses the
tuple-vs-list distinction the descr format relies on, and ``descr_to_dtype``
then misreads it (crashing on ``'f0'`` on some numpy versions). The robust
fix encodes structured/subarray dtypes as a pickled blob and keeps simple
dtypes as their plain type string.
"""

import json

import numpy as np
import pytest

from comfy_env.isolation.workers._ipc_shared import (
    _encode_np_dtype,
    _decode_np_dtype,
    _to_shm_generic,
    _cleanup_shm,
)

SIMPLE = [np.dtype(s) for s in ("float32", "int64", "uint8", "bool", "float64",
                                "int16", "datetime64[ns]")]
STRUCTURED = [
    # the exact crashing case: a PLY vertex record with normals + a scalar field
    np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("pressure_cp", "<f4")]),
    np.dtype([("pos", "<f4", (3,)), ("col", "u1", (4,))]),   # subarray fields
    np.dtype([("a", [("b", "<f4"), ("c", "<i4")]), ("d", "<f8")]),  # nested struct
    np.dtype([("", "<f4"), ("", "<i4")]),                    # auto-named -> f0,f1
]
ALL = SIMPLE + STRUCTURED


@pytest.mark.parametrize("dt", ALL, ids=lambda d: (d.name if d.names else str(d))[:24])
def test_dtype_encoding_roundtrips_through_json(dt):
    wire = json.loads(json.dumps(_encode_np_dtype(dt)))  # simulate the JSON hop
    assert _decode_np_dtype(wire) == dt


def test_simple_stays_string_structured_becomes_pickle_blob():
    assert _encode_np_dtype(np.dtype("float32")) == "<f4"          # readable, small
    enc = _encode_np_dtype(STRUCTURED[0])
    assert isinstance(enc, dict) and "__pickle_dtype__" in enc      # structured -> blob


def test_legacy_simple_dtype_string_still_decodes():
    # frames that carried str(dtype) for a simple dtype must still load
    assert _decode_np_dtype("float32") == np.dtype("float32")


def _reconstruct(res):
    """Mirror _from_shm's __shm_np__ branch (the fixed decode line)."""
    from multiprocessing import shared_memory as shm
    shape = tuple(res["shape"])
    dtype = _decode_np_dtype(res["dtype"])
    block = shm.SharedMemory(name=res["__shm_np__"])
    try:
        return np.ndarray(shape, dtype=dtype, buffer=block.buf).copy()
    finally:
        block.close()
        block.unlink()


@pytest.mark.parametrize("dt", STRUCTURED, ids=lambda d: (d.name or "struct")[:24])
def test_structured_array_survives_to_shm_and_back(dt):
    a = np.zeros(37, dtype=dt)
    for name in (dt.names or ())[:2]:          # non-trivial bytes so zeros can't fake it
        a[name].flat[:] = np.arange(a[name].size) % 251

    registry = []
    # structured arrays fail torch.from_numpy, so this exercises the shm-copy
    # branch on Windows too (_USE_MEMFD is False there -> named SharedMemory).
    res = _to_shm_generic(a, registry, {}, tensor_serializer=lambda *x, **k: {})
    try:
        if "__shm_np__" not in res or res.get("__shm_np__") is True:
            pytest.skip("memfd frame (Linux) -- covered by the encode/decode test")
        b = _reconstruct(res)
        assert b.dtype == a.dtype
        assert b.shape == a.shape
        assert a.tobytes() == b.tobytes()
    finally:
        _cleanup_shm(registry)
