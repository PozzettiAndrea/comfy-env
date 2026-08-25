"""Contract: what a proxy does around worker.call_method.

The V1 and V3 proxies each wrap the same body -- acquire worker, prepare
kwargs for IPC, call, register patchers, prepare result, drop the worker on a
transport error. Two rules in that body are load-bearing and neither had any
test:

  * `_register_new_patchers` runs in a `finally`, so models auto-detected
    during a call that RAISED still reach the parent's ledger. Weights are on
    the GPU either way, and a model with no ledger entry can never be evicted.
  * A RuntimeError/ConnectionError removes the worker from the pool before
    propagating, so the next call gets a fresh process.

They also differ in exactly one place: V1 sends the instance __dict__ as
self_state, V3 sends None. V3's `bound` is the CLASS -- cls.__dict__ is truthy
and full of classmethod objects, which are not JSON-serializable, so a shared
body that reads __dict__ off `bound` kills every V3 node call.
"""

import sys
import types

import pytest


@pytest.fixture()
def md(monkeypatch):
    """Import metadata with comfy.model_management stubbed."""
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cpu")
    mm.get_total_memory = lambda dev: 0
    mm.get_free_memory = lambda dev: 0
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    # Without comfy_api the V3 builder silently falls back to a V1 proxy, so
    # the v3 parametrization would quietly test V1 twice. Only ComfyNode is
    # touched (used as the base class).
    class _ComfyNode:
        pass

    io_mod = types.ModuleType("comfy_api.latest.io")
    io_mod.ComfyNode = _ComfyNode
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_mod
    api = types.ModuleType("comfy_api")
    api.latest = latest
    for name, mod in (("comfy_api", api), ("comfy_api.latest", latest),
                      ("comfy_api.latest.io", io_mod)):
        monkeypatch.setitem(sys.modules, name, mod)

    from comfy_env.isolation import metadata
    return metadata


class _RecordingWorker:
    def __init__(self, boom=False):
        self.boom = boom
        self.seen = []

    def call_method(self, **kw):
        self.seen.append(kw)
        if self.boom:
            raise RuntimeError("transport exploded")
        return ("ok",)


@pytest.fixture()
def pool_stub(monkeypatch, md):
    """Intercept the pool so no subprocess is ever spawned."""
    from comfy_env.isolation import pool

    state = {"worker": _RecordingWorker(), "registered": [], "removed": []}
    monkeypatch.setattr(pool, "_get_or_create_worker",
                        lambda *a, **k: (state["worker"], 42))
    monkeypatch.setattr(pool, "_register_new_patchers",
                        lambda ed, w, g: state["registered"].append(g))
    monkeypatch.setattr(pool, "_remove_worker",
                        lambda ed: state["removed"].append(ed))
    return state


def _meta(is_v3):
    m = {
        "module_name": "mypack.nodes", "class_name": "MyNode", "function": "run",
        "input_types": {"required": {"x": ("INT", {})}},
        "return_types": ["IMAGE"], "return_names": ["image"],
        "category": "test", "output_node": False,
    }
    if is_v3:
        m["is_v3"] = True
        m["node_info_v1"] = {
            "input": {"required": {"x": ("INT", {})}},
            "output": ["IMAGE"], "output_name": ["image"],
            "output_is_list": [False], "category": "test",
            "name": "MyNode", "display_name": "My Node",
            "description": "", "output_node": False,
        }
    return m


def _build(md, is_v3, tmp_path):
    return md.build_proxy_class(
        node_name="MyNode", meta=_meta(is_v3), env_dir=tmp_path,
        package_root=tmp_path, sys_path=[], env_vars={},
    )


def _invoke(cls):
    """Call the proxy the way ComfyUI would, V1 or V3."""
    if hasattr(cls, "execute"):
        return cls.execute(x=1)
    return cls().run(x=1)


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_self_state_is_json_serializable(md, pool_stub, tmp_path, is_v3):
    """V3 must send self_state=None, never the class __dict__.

    cls.__dict__ is truthy and holds classmethod objects; the transport's
    json.dumps refuses them, so every V3 node call would die on the wire.
    """
    import json

    _invoke(_build(md, is_v3, tmp_path))

    sent = pool_stub["worker"].seen[-1]
    json.dumps(sent["self_state"])          # must not raise
    if is_v3:
        assert sent["self_state"] is None


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_patchers_are_registered_even_when_the_call_raises(md, monkeypatch,
                                                           pool_stub, tmp_path,
                                                           is_v3):
    """A model whose weights reached the GPU must reach the ledger."""
    pool_stub["worker"].boom = True

    with pytest.raises(RuntimeError, match="transport exploded"):
        _invoke(_build(md, is_v3, tmp_path))

    assert pool_stub["registered"] == [42], (
        "auto-detected models were not registered after a failed call -- "
        "they are GPU-resident with no ledger entry and can never be evicted"
    )
    assert pool_stub["removed"], "a transport error must drop the worker"


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_result_is_returned_on_the_happy_path(md, pool_stub, tmp_path, is_v3):
    assert _invoke(_build(md, is_v3, tmp_path)) == ("ok",)
    assert pool_stub["registered"] == [42]
    assert pool_stub["removed"] == []
