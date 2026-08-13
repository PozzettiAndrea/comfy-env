"""Regression: numpy STRUCTURED dtypes survive the shared-memory boundary.

The shm frame stored the dtype as ``str(arr.dtype)`` and rebuilt it with
``np.dtype(str)``. That is fine for simple dtypes but ``np.dtype()`` cannot
parse the ``str()`` of a structured/record dtype -- e.g. a PLY point cloud
with per-vertex fields ``[('x','<f4'), ... ('pressure_cp','<f4')]`` -- so
deserialization crashed with "data type ... not understood". Fixed by
encoding the dtype via numpy's own ``dtype_to_descr`` / ``descr_to_dtype``
(``_encode_np_dtype`` / ``_decode_np_dtype``), which round-trips both kinds
through JSON.
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

DTYPES = [
    np.dtype("float32"),
    np.dtype("int64"),
    np.dtype("uint8"),
    np.dtype("bool"),
    np.dtype("float64"),
    # the exact crashing case: a PLY vertex record with normals + a scalar field
    np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
              ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"), ("pressure_cp", "<f4")]),
    # structured with subarray fields (shapes must survive the JSON tuple->list trip)
    np.dtype([("pos", "<f4", (3,)), ("col", "u1", (4,))]),
]


@pytest.mark.parametrize("dt", DTYPES, ids=lambda d: d.name if d.names else str(d))
def test_dtype_encoding_roundtrips_through_json(dt):
    wire = json.loads(json.dumps(_encode_np_dtype(dt)))  # simulate the JSON hop
    assert _decode_np_dtype(wire) == dt


def test_simple_dtype_stays_a_plain_string_on_the_wire():
    # keeps frames small/readable and matches legacy simple-dtype frames
    assert _encode_np_dtype(np.dtype("float32")) == "<f4"


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


@pytest.mark.parametrize("dt", DTYPES, ids=lambda d: d.name if d.names else str(d))
def test_structured_array_survives_to_shm_and_back(dt):
    a = np.zeros(37, dtype=dt) if dt.names else np.arange(37 * 3).reshape(37, 3).astype(dt)
    if dt.names:  # put non-trivial bytes in so an all-zero pass can't fake success
        for name in dt.names[:2]:
            a[name].flat[:] = np.arange(a[name].size)

    registry = []
    # structured arrays fail torch.from_numpy, so this exercises the shm-copy
    # branch on Windows too (_USE_MEMFD is False there -> named SharedMemory).
    res = _to_shm_generic(a, registry, {}, tensor_serializer=lambda *x, **k: {})
    try:
        # a memfd frame (Linux) can't be reopened by name; skip that leg there
        if "__shm_np__" not in res or res.get("__shm_np__") is True:
            pytest.skip("memfd frame (Linux) -- covered by the encode/decode test")
        b = _reconstruct(res)
        assert b.dtype == a.dtype
        assert b.shape == a.shape
        assert a.tobytes() == b.tobytes()
    finally:
        _cleanup_shm(registry)
