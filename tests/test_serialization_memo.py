"""Contract: the walk's memo is keyed by id(obj), so it must hold obj alive.

A value produced DURING the walk -- a computed property, a `.double()`, a
serializer temporary -- is freed the moment `_to_shm_generic` returns. CPython
hands the identical address to the next field's temporary, which the memo then
reports as already-serialized and answers with the previous field's frame.

Wrong data, right dtype, right shape, no exception. The built-in ComfyUI
geometry codec is written in exactly the vulnerable shape:

    dict((f, recurse(getattr(obj, f, None))) for f in fields)
"""

import pytest

torch = pytest.importorskip("torch")

from comfy_env.isolation.workers import _ipc_shared as S  # noqa: E402


class _Geom:
    """Fields are computed, so each access yields a fresh short-lived tensor."""

    def __init__(self, a, b):
        self._a, self._b = a, b

    @property
    def vertices(self):
        return self._a.double()

    @property
    def vertex_colors(self):
        return self._b.double()


def _summing_serializer():
    calls = {"n": 0}

    def ser(t, registry, visited):
        calls["n"] += 1
        return {"frame": calls["n"], "sum": float(t.sum())}

    return ser


def test_computed_fields_do_not_collide_on_a_recycled_address():
    """The regression: vertex_colors shipped vertices' data, 200/200 trials."""
    ser = _summing_serializer()

    for _ in range(200):
        g = _Geom(torch.full((8,), 1.0), torch.full((8,), 9.0))
        truth = {f: float(getattr(g, f).sum()) for f in ("vertices", "vertex_colors")}

        visited = {}
        wire = {
            f: S._to_shm_generic(getattr(g, f), [], visited, tensor_serializer=ser)
            for f in ("vertices", "vertex_colors")
        }

        for f in truth:
            assert wire[f]["sum"] == truth[f], (
                f"{f} shipped {wire[f]['sum']}, expected {truth[f]} -- "
                "the memo answered with another field's frame"
            )


def test_memo_still_deduplicates_a_genuinely_shared_object():
    """The memo must keep doing its actual job: one frame per shared object."""
    ser = _summing_serializer()
    shared = torch.ones(4)
    visited = {}

    first = S._to_shm_generic(shared, [], visited, tensor_serializer=ser)
    second = S._to_shm_generic(shared, [], visited, tensor_serializer=ser)

    assert first is second, "a repeated reference must reuse the first frame"


def test_memo_entry_holds_a_reference_to_its_key():
    """Why the value is a pair: the key is an address, not an identity."""
    ser = _summing_serializer()
    visited = {}

    S._to_shm_generic(torch.ones(4).double(), [], visited, tensor_serializer=ser)

    assert visited, "nothing memoized"
    for key, entry in visited.items():
        assert isinstance(entry, tuple) and len(entry) == 2, (
            "memo value must be (obj, result) -- storing the result alone lets "
            "the key's address be recycled by a later object"
        )
        assert id(entry[0]) == key, "memo does not hold the object its key names"
