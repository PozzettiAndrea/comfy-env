"""Contract: ComfyUI's core geometry types round-trip through the registry.

MESH / VOXEL / SPLAT (comfy_api/latest/_util/geometry_types.py) are plain
classes whose fields are all tensors. Before they were registered they fell to
the pickle rung, which pickles those tensors instead of letting them take the
shared-memory tensor path.

These tests drive the codecs directly with an identity `recurse`, so they run
without torch and without a ComfyUI checkout. What they pin down is the part
that was easy to get wrong: field decomposition, reconstruction into the REAL
class, tolerance of absent optional fields, and the degradation path on a side
that cannot import comfy_api -- the failure mode that made __shm_sparse_tensor__
a one-way street.
"""

import sys
import types

from comfy_env.isolation.workers._ipc_shared import (
    OpaquePayload,
    _COMFY_GEOM_FIELDS,
    _COMFY_GEOM_CLASSES,
    _make_comfy_geom_codec,
)


class _FakeMESH:
    def __init__(self, vertices, faces, uvs=None, vertex_colors=None,
                 texture=None, vertex_counts=None, face_counts=None,
                 unlit=False):
        self.vertices, self.faces, self.uvs = vertices, faces, uvs
        self.vertex_colors, self.texture = vertex_colors, texture
        self.vertex_counts, self.face_counts = vertex_counts, face_counts
        self.unlit = unlit


def _install_fake_comfy_api(monkeypatch, **classes):
    latest = types.ModuleType("comfy_api.latest")
    for name, cls in classes.items():
        setattr(latest, name, cls)
    pkg = types.ModuleType("comfy_api")
    pkg.latest = latest
    monkeypatch.setitem(sys.modules, "comfy_api", pkg)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)


def _identity(v):
    return v


def test_every_field_goes_through_recurse():
    """The point of registering: each field is walked, so tensors inside take
    the tensor path instead of being pickled with the object."""
    ser, _ = _make_comfy_geom_codec("MESH")
    seen = []

    def recurse(v):
        seen.append(v)
        return v

    mesh = _FakeMESH(vertices="V", faces="F", uvs="U", unlit=True)
    payload = ser(mesh, recurse)

    assert set(payload) == set(_COMFY_GEOM_FIELDS["MESH"])
    assert payload["vertices"] == "V" and payload["faces"] == "F"
    assert payload["unlit"] is True
    assert payload["texture"] is None          # absent optional, not an error
    assert len(seen) == len(_COMFY_GEOM_FIELDS["MESH"])


def test_round_trip_rebuilds_the_real_class(monkeypatch):
    _COMFY_GEOM_CLASSES.pop("MESH", None)
    _install_fake_comfy_api(monkeypatch, MESH=_FakeMESH)
    ser, deser = _make_comfy_geom_codec("MESH")

    out = deser(ser(_FakeMESH("V", "F", uvs="U"), _identity), _identity)

    assert isinstance(out, _FakeMESH)          # NOT a marker dict
    assert (out.vertices, out.faces, out.uvs) == ("V", "F", "U")
    assert out.texture is None and out.unlit is False
    _COMFY_GEOM_CLASSES.pop("MESH", None)


def test_side_without_comfy_api_degrades_to_opaque(monkeypatch):
    """A bare host forwards the value instead of dying on it."""
    _COMFY_GEOM_CLASSES.pop("VOXEL", None)
    monkeypatch.setitem(sys.modules, "comfy_api", None)  # import raises
    ser, deser = _make_comfy_geom_codec("VOXEL")

    class _V:
        data = "D"

    out = deser(ser(_V(), _identity), _identity)

    assert isinstance(out, OpaquePayload)
    assert out.tag == "comfy_api.VOXEL"
    assert out.payload["data"] == "D"          # still forwardable


def test_a_failed_lookup_is_not_cached(monkeypatch):
    """The worker imports _ipc_shared before ComfyUI reaches sys.path, so an
    early miss must not poison every later call."""
    _COMFY_GEOM_CLASSES.pop("VOXEL", None)
    monkeypatch.setitem(sys.modules, "comfy_api", None)
    ser, deser = _make_comfy_geom_codec("VOXEL")

    class _V:
        data = "D"

    assert isinstance(deser(ser(_V(), _identity), _identity), OpaquePayload)

    class _RealV:
        def __init__(self, data):
            self.data = data

    _install_fake_comfy_api(monkeypatch, VOXEL=_RealV)
    out = deser(ser(_V(), _identity), _identity)
    assert isinstance(out, _RealV) and out.data == "D"
    _COMFY_GEOM_CLASSES.pop("VOXEL", None)


def test_all_three_types_are_registered():
    from comfy_env.isolation.workers._ipc_shared import REGISTRY
    for name in ("MESH", "VOXEL", "SPLAT"):
        assert REGISTRY.lookup_deserializer("comfy_api." + name) is not None
