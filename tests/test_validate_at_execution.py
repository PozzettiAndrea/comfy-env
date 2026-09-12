"""Contract: a pack's VALIDATE_INPUTS body runs, and its message reaches the user.

Upstream runs the body at submit, in the host. Under isolation the body lives
in the worker, so the host's stand-in carried only the signature (the
exemptions upstream reads) and returned True: ~200 real validate bodies in
~80 corpus packs never ran, and a rejected value failed deeper with a
traceback instead of the author's sentence.

Now the stand-in records what it was handed, keyed by the executing context
upstream wraps around both the validate call and the later FUNCTION call; the
FUNCTION call ships the record; the worker runs the real body against it
right before the function. Deterministic, no submit-time socket traffic.
"""

import sys
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md
from comfy_env.isolation.workers import WorkerError
from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


# --- the host's stand-in and its record ------------------------------------

class _Ctx:
    def __init__(self, prompt_id, node_id, list_index=0):
        self.prompt_id, self.node_id, self.list_index = prompt_id, node_id, list_index


@pytest.fixture()
def ctx(monkeypatch):
    holder = {"ctx": None}
    monkeypatch.setattr(md, "_executing_context", lambda: holder["ctx"])
    md._VALIDATE_RECORDS.clear()
    return holder


def test_stand_in_records_what_it_was_handed(ctx):
    cm, names = md._make_named_validate(["mesh", "x"], varkw=False, record=True)
    ctx["ctx"] = _Ctx("p1", "7")
    assert cm.__func__(object, mesh="a.obj", x=3) is True
    assert md._take_validate_kwargs() == {"mesh": "a.obj", "x": 3}


def test_signature_is_unchanged_by_recording():
    import inspect
    cm, _ = md._make_named_validate(["mesh", "x"], varkw=True, record=True)
    spec = inspect.getfullargspec(cm.__func__)
    assert spec.args == ["cls", "mesh", "x"] and spec.varkw == "kwargs"


def test_kwargs_form_records_the_catch_all_too(ctx):
    cm, _ = md._make_named_validate(["x"], varkw=True, record=True)
    ctx["ctx"] = _Ctx("p1", "7")
    cm.__func__(object, x=1, extra="e")
    assert md._take_validate_kwargs() == {"x": 1, "extra": "e"}


def test_no_body_means_no_record(ctx):
    """A stand-in that exists only for combo exemptions has nothing to run."""
    cm, _ = md._make_named_validate(["mesh"], varkw=False, record=False)
    ctx["ctx"] = _Ctx("p1", "7")
    cm.__func__(object, mesh="a.obj")
    assert md._take_validate_kwargs() is None


def test_outside_a_context_nothing_is_recorded(ctx):
    cm, _ = md._make_named_validate(["x"], record=True)
    ctx["ctx"] = None
    cm.__func__(object, x=1)
    assert md._VALIDATE_RECORDS == {}


def test_the_store_is_bounded_by_prompts(ctx):
    """A prompt rejected on some other node never executes; its records
    must not accumulate."""
    cm, _ = md._make_named_validate(["x"], record=True)
    for i in range(10):
        ctx["ctx"] = _Ctx(f"p{i}", "1")
        cm.__func__(object, x=i)
    assert len(md._VALIDATE_RECORDS) == md._VALIDATE_MAX_PROMPTS
    ctx["ctx"] = _Ctx("p9", "1")
    assert md._take_validate_kwargs() == {"x": 9}
    ctx["ctx"] = _Ctx("p0", "1")
    assert md._take_validate_kwargs() is None


def test_every_list_index_sees_the_one_record(ctx):
    cm, _ = md._make_named_validate(["x"], record=True)
    ctx["ctx"] = _Ctx("p1", "7", 0)
    cm.__func__(object, x=1)
    ctx["ctx"] = _Ctx("p1", "7", 2)
    assert md._take_validate_kwargs() == {"x": 1}


# --- the worker runs the real body ------------------------------------------

@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES, name="validate-worker")
    yield w
    w.shutdown()


def _run(worker, cls, x, validate):
    return worker.call_method(module_name="validate_node", class_name=cls, method_name="run",
                              kwargs={"x": x}, validate_kwargs=validate, timeout=60.0)


def test_the_authors_message_is_the_node_error(worker):
    with pytest.raises(WorkerError) as ei:
        _run(worker, "Ranged", -1, {"x": -1, "mesh": "a.obj"})
    assert ei.value.error_kind == "validation"
    assert ei.value.args[0] == "Custom validation failed for node: x must be non-negative, got -1"
    # what the host raises on the node: upstream's own wording, no traceback
    from comfy_env.isolation.errors import translate_error
    t = translate_error(ei.value)
    assert type(t) is ValueError and str(t) == ei.value.args[0]
    assert t.__cause__ is ei.value


def test_a_passing_validate_runs_the_node(worker):
    assert _run(worker, "Ranged", 5, {"x": 5, "mesh": "a.obj"}) == [5] or \
           _run(worker, "Ranged", 5, {"x": 5, "mesh": "a.obj"}) == (5,)


def test_kwargs_are_filtered_to_the_real_argspec(worker):
    """The host's stand-in exempts every combo, so it records `mesh` too; the
    author's validate takes only `x`. Without filtering: TypeError."""
    _run(worker, "Ranged", 1, {"x": 1, "mesh": "a.obj", "unexpected": True})


def test_bare_false_rejects(worker):
    with pytest.raises(WorkerError) as ei:
        _run(worker, "Blanket", 13, {"x": 13})
    assert ei.value.error_kind == "validation"
    assert ei.value.args[0] == "Custom validation failed for node"


def test_an_async_body_is_awaited_not_skipped(worker):
    with pytest.raises(WorkerError) as ei:
        _run(worker, "Async", 7, {"x": 7})
    assert "async says no" in str(ei.value)
    _run(worker, "Async", 8, {"x": 8})


def test_no_record_means_no_validate_call(worker):
    """validate_kwargs=None is the pre-2026-09-12 behaviour: run the node."""
    _run(worker, "Ranged", -1, None)


# --- upstream's context is the key ------------------------------------------

import os
COMFYUI_DIR = os.environ.get("COMFYUI_DIR")


@pytest.mark.comfyui
@pytest.mark.skipif(not COMFYUI_DIR, reason="COMFYUI_DIR not set")
def test_upstream_validate_prompt_records_under_the_executing_context(tmp_path):
    """Run ComfyUI's own validate_prompt on a proxy built from a scan, then
    read the record back under the CurrentNodeContext the executor will use
    for the FUNCTION call. If upstream stops wrapping validate in that
    context, or changes the key, this is where it shows."""
    import asyncio
    import shutil
    sys.path.insert(0, COMFYUI_DIR)
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        sys.argv = [sys.argv[0], "--cpu"]
        import execution
        import nodes
        from comfy_execution.utils import CurrentNodeContext
    finally:
        pass
    # scan the fixture pack through the real script, build the V1 proxy
    pkg = tmp_path / "vpack"
    pkg.mkdir()
    shutil.copy(FIXTURES / "validate_node.py", pkg / "node.py")
    (pkg / "__init__.py").write_text("from .node import *\n", encoding="utf-8")
    env_dir = tmp_path / "env"; (env_dir / "bin").mkdir(parents=True)
    (env_dir / "bin" / "python").symlink_to(sys.executable)
    payload = md.fetch_metadata(env_dir, "vpack", tmp_path)
    meta = dict(payload["nodes"]["Ranged"])
    meta["output_node"] = True   # validate_prompt walks back from output nodes only
    Proxy = md.build_proxy_class(node_name="Ranged", meta=meta,
                                 env_dir=env_dir, package_root=tmp_path, sys_path=[], env_vars={})
    nodes.NODE_CLASS_MAPPINGS["_CEV_Ranged"] = Proxy
    md._VALIDATE_RECORDS.clear()
    try:
        prompt = {"7": {"class_type": "_CEV_Ranged", "inputs": {"x": -1, "mesh": "zzz.obj"}}}
        valid, err, good, errs = asyncio.run(execution.validate_prompt("prompt-xyz", prompt, None))
        # the stand-in accepted (exemption for the live combo, body deferred)
        assert valid, err
        with CurrentNodeContext("prompt-xyz", "7", 0):
            rec = md._take_validate_kwargs()
        assert rec == {"x": -1, "mesh": "zzz.obj"}, rec
    finally:
        nodes.NODE_CLASS_MAPPINGS.pop("_CEV_Ranged", None)
        sys.path.remove(COMFYUI_DIR)
