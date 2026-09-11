"""check_lazy_status crosses the boundary -- but only when the author wrote one.

Two properties, both load-bearing:

* Forwarded: the proxy carries a `check_lazy_status` that runs the pack's
  real method in the worker, so upstream's ask-then-compute loop gets a real
  answer instead of silence. Without it the pruned lazy inputs are never
  promoted and the node runs with every one of them None.
* Conditional: a node that did not define one gets no forwarder. Attaching
  it anyway would cost every isolated node a round-trip to be told "no
  questions" -- and would hand a node that declares `lazy` without a policy a
  behaviour plain ComfyUI does not have, since upstream's own default is
  unreachable there too.
"""

import sys
from pathlib import Path

import pytest

from test_proxy_call_body import _build, _meta, md, pool_stub  # noqa: F401

from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def real_worker_pool(monkeypatch, md):  # noqa: F811
    """Route the proxy's calls to a REAL worker instead of the recorder."""
    from comfy_env.isolation import pool

    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="lazy-test-worker")
    monkeypatch.setattr(pool, "_get_or_create_worker", lambda *a, **k: (w, 1))
    monkeypatch.setattr(pool, "_register_new_patchers", lambda *a, **k: None)
    monkeypatch.setattr(pool, "_remove_worker", lambda *a, **k: None)
    yield w
    w.shutdown()


def _lazy_meta():
    m = _meta(is_v3=False)
    m.update({
        "module_name": "lazy_node", "class_name": "V1Switch", "function": "pick",
        "has_check_lazy": True,
        "input_types": {"required": {
            "select":   ("BOOLEAN", {}),
            "on_true":  ("*", {"lazy": True}),
            "on_false": ("*", {"lazy": True}),
        }},
    })
    return m


def test_check_lazy_status_reaches_the_real_node(md, real_worker_pool, tmp_path):  # noqa: F811
    """Round one asks for the taken branch; round two, with it present, asks
    for nothing. That two-round shape is upstream's loop (execution.py:507-520),
    and the worker's answer has to be the pack's answer both times."""
    Proxy = md.build_proxy_class(
        node_name="V1Switch", meta=_lazy_meta(), env_dir=tmp_path,
        package_root=tmp_path, sys_path=[], env_vars={},
    )
    obj = Proxy()
    # upstream's V1 presence test, execution.py:506
    assert getattr(obj, "check_lazy_status", None) is not None

    first = obj.check_lazy_status(select=True, on_true=None, on_false=None)
    assert first == ["on_true"], "did not ask for the taken branch"

    second = obj.check_lazy_status(select=True, on_true="X", on_false=None)
    assert second == [], "kept asking after the branch was supplied"

    # and the real call still works on the same instance
    assert obj.pick(select=True, on_true="X", on_false=None) == ["X"] \
        or obj.pick(select=True, on_true="X", on_false=None) == ("X",)


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_forwarder_attaches_only_when_the_author_wrote_one(md, pool_stub, tmp_path, is_v3):  # noqa: F811
    # absent by default
    Proxy = _build(md, is_v3, tmp_path)
    assert "check_lazy_status" not in Proxy.__dict__
    assert pool_stub["worker"].seen == [], "a round-trip was paid for nothing"

    # present when the scan saw one -- in the proxy's OWN __dict__, which is
    # what first_real_override walks the MRO for on the V3 path
    m = _meta(is_v3); m["has_check_lazy"] = True
    Proxy = md.build_proxy_class(
        node_name="MyNode", meta=m, env_dir=tmp_path,
        package_root=tmp_path, sys_path=[], env_vars={},
    )
    assert "check_lazy_status" in Proxy.__dict__
    if is_v3:
        assert isinstance(Proxy.__dict__["check_lazy_status"], classmethod)
    else:
        assert getattr(Proxy(), "check_lazy_status", None) is not None
