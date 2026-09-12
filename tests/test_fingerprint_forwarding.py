"""Contract: a pack's own IS_CHANGED / fingerprint_inputs runs under isolation.

A node that defined a fingerprint used to be cached forever under isolation:
the proxy carried no method, so ComfyUI fell to its constant False
(execution.py:82-84) and served the cached output until restart. The
fingerprint is now forwarded over the same no-spawn ladder as live
dropdowns (`_forward_fingerprint`), with the miss answer turned upside
down: where a dropdown miss keeps the cached list, a fingerprint miss
answers `float("nan")`, which ComfyUI reads as "changed, re-run".

Three surfaces, each pinned here: the host ladder with a fake pool, the two
proxies' attach and kwargs shaping, and the worker handler over a REAL
`SubprocessWorker`. Plus source-level seam guards, because the worker is
exec'd by a foreign interpreter (ADR-0006) and its dispatch placement can
only be checked as text. Every test says which wrong implementation it
catches. None skip.
"""

import ast
import inspect
import json
import math
import sys
import threading
import time
from pathlib import Path

import pytest

import comfy_env.isolation.pool as _pool

from test_call_scope import _worker_source
from test_live_combo_options import _Worker, pool  # noqa: F401
from test_metadata_json_payload import _run_scan
from test_proxy_call_body import _meta, md, pool_stub  # noqa: F401

from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"

OK = {"status": "ok", "value": "mtime:42"}


def _isnan(v):
    return isinstance(v, float) and math.isnan(v)


def _forward(md, **over):  # noqa: F811
    kw = dict(env_dir="/env", module_name="m", class_name="C",
              method_name="IS_CHANGED", kwargs={"path": "a.obj"}, hidden=None)
    kw.update(over)
    return md._forward_fingerprint(**kw)


# --- the ladder -------------------------------------------------------------

def test_rung_1_returns_the_packs_own_answer(md, pool):  # noqa: F811
    """Catches a ladder that answers "changed" even when the worker spoke."""
    pool["/env"] = _pool.WorkerRecord(_Worker(OK), 1)
    assert _forward(md) == "mtime:42"


def test_rung_2_busy_answers_changed_not_unchanged(md, pool):  # noqa: F811
    """Catches the dropdown ladder's miss answer (None, or False) copied
    over: here a miss must be NaN, never a stable key."""
    pool["/env"] = _pool.WorkerRecord(_Worker("busy"), 1)
    got = _forward(md)
    assert _isnan(got)
    assert got is not None and got is not False


def test_rung_2_dead_answers_changed(md, pool):  # noqa: F811
    pool["/env"] = _pool.WorkerRecord(_Worker("dead"), 1)
    assert _isnan(_forward(md))


def test_rung_3_no_worker_answers_changed_and_never_spawns_one(md, pool, monkeypatch):  # noqa: F811
    """Catches a ladder that reaches for _get_or_create_worker on a miss: a
    cold spawn per isolated node on every prompt submit, on the event loop."""
    from comfy_env.isolation import pool as pool_mod

    def _spawn(*a, **k):
        raise AssertionError("a fingerprint spawned a worker")
    monkeypatch.setattr(pool_mod, "_get_or_create_worker", _spawn)
    assert _isnan(_forward(md))
    assert pool == {}


def test_a_raising_transport_answers_changed(md, pool):  # noqa: F811
    """Catches an implementation that lets a socket death out into
    IsChangedCache.get."""
    pool["/env"] = _pool.WorkerRecord(_Worker(RuntimeError("socket died")), 1)
    assert _isnan(_forward(md))


def test_an_error_frame_answers_changed(md, pool):  # noqa: F811
    pool["/env"] = _pool.WorkerRecord(_Worker({"status": "error", "error": "boom"}), 1)
    assert _isnan(_forward(md))


def test_the_lock_wait_is_short_and_the_frame_is_exact(md, pool):  # noqa: F811
    """Catches a second lock budget drifting from the dropdown ladder's, and
    a frame the worker's _handle_fingerprint would not understand."""
    w = _Worker(OK)
    pool["/env"] = _pool.WorkerRecord(w, 1)
    _forward(md, kwargs={"path": "a"}, hidden=[["UNIQUE_ID", "node_id", "7"]])
    (method, lock_timeout, params) = w.calls[0]
    assert method == "fingerprint"
    assert 0 < lock_timeout <= 0.5
    assert lock_timeout == md._REFRESH_LOCK_TIMEOUT
    assert params == {"module": "m", "class_name": "C",
                      "method_name": "IS_CHANGED", "kwargs": {"path": "a"},
                      "hidden": [["UNIQUE_ID", "node_id", "7"]]}


class _Untouchable:
    """Any read is a failure: the rung-0 check must reject by type alone."""

    def __getattr__(self, name):
        raise AssertionError(f"read attribute {name!r} of a non-primitive input")

    def __len__(self):
        raise AssertionError("took len() of a non-primitive input")

    def __iter__(self):
        raise AssertionError("iterated a non-primitive input")

    def __repr__(self):
        raise AssertionError("repr() of a non-primitive input")


def test_a_non_primitive_input_is_changed_and_never_sent(md, pool):  # noqa: F811
    """Catches an implementation that serializes first and asks second, or
    that runs prepare_for_ipc_recursive over the inputs."""
    w = _Worker(OK)
    pool["/env"] = _pool.WorkerRecord(w, 1)
    assert _isnan(_forward(md, kwargs={"image": _Untouchable(), "seed": 1}))
    assert w.calls == []


def test_a_non_primitive_hidden_value_is_changed_and_never_sent(md, pool):  # noqa: F811
    w = _Worker(OK)
    pool["/env"] = _pool.WorkerRecord(w, 1)
    assert _isnan(_forward(md, hidden=[["DYNPROMPT", None, _Untouchable()]]))
    assert w.calls == []


def test_a_non_primitive_reply_is_changed(md, pool):  # noqa: F811
    """Catches a host that trusts the reply shape: primitives only, in both
    directions."""
    pool["/env"] = _pool.WorkerRecord(_Worker({"status": "ok", "value": {"h": 1}}), 1)
    assert _isnan(_forward(md))


def test_changed_flag_beats_value(md, pool):  # noqa: F811
    pool["/env"] = _pool.WorkerRecord(_Worker({"status": "ok", "value": "x", "changed": True}), 1)
    assert _isnan(_forward(md))


def test_none_is_a_legitimate_unchanged_answer(md, pool):  # noqa: F811
    """Catches an implementation that keys on falsiness: None is a stable
    cache key natively (None == None), so it must come back as None."""
    pool["/env"] = _pool.WorkerRecord(_Worker({"status": "ok", "value": None}), 1)
    got = _forward(md)
    assert got is None


def test_nested_json_inputs_are_primitive_enough(md, pool):  # noqa: F811
    """Catches a rung-0 check that refuses dicts: a PROMPT hidden value is a
    dict of dicts and must reach the worker."""
    w = _Worker(OK)
    pool["/env"] = _pool.WorkerRecord(w, 1)
    prompt = {"7": {"class_type": "KSampler", "inputs": {"seed": 42, "l": [1, "a", None]}}}
    assert _forward(md, hidden=[["PROMPT", "prompt", prompt]]) == "mtime:42"
    assert len(w.calls) == 1


# --- proxy attach and kwargs shaping ---------------------------------------

def _build_with(md, is_v3, tmp_path, **over):  # noqa: F811
    m = _meta(is_v3)
    m.update(over)
    return md.build_proxy_class(
        node_name="MyNode", meta=m, env_dir=tmp_path,
        package_root=tmp_path, sys_path=[], env_vars={},
    )


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_forwarder_attaches_only_when_the_scan_saw_a_fingerprint(md, pool_stub, tmp_path, is_v3):  # noqa: F811
    """Catches an unconditional attach (every isolated node would answer NaN
    and the output cache would be gone) and the wrong spelling per proxy
    kind (execution.py:76 walks for lowercase, :79 hasattr for uppercase)."""
    Proxy = _build_with(md, is_v3, tmp_path)
    assert "IS_CHANGED" not in Proxy.__dict__
    assert "fingerprint_inputs" not in Proxy.__dict__
    assert not hasattr(Proxy, "IS_CHANGED")

    Proxy = _build_with(md, is_v3, tmp_path, fingerprint_args=[])
    if is_v3:
        assert isinstance(Proxy.__dict__["fingerprint_inputs"], classmethod)
        assert "IS_CHANGED" not in Proxy.__dict__
    else:
        assert isinstance(Proxy.__dict__["IS_CHANGED"], classmethod)
        assert "fingerprint_inputs" not in Proxy.__dict__
    assert pool_stub["worker"].seen == [], "a round trip was paid at build time"


def test_the_v1_fallback_of_a_v3_node_asks_for_fingerprint_inputs(md, pool, tmp_path):  # noqa: F811
    """Catches a method name keyed on which proxy was built instead of on
    the scan's view of the real class: a V3 node behind a V1 proxy defines
    fingerprint_inputs, not IS_CHANGED."""
    w = _Worker(OK)
    pool[str(tmp_path)] = _pool.WorkerRecord(w, 1)
    m = _meta(True)
    m.pop("node_info_v1")          # forces the V1 builder
    m["fingerprint_args"] = ["x"]
    Proxy = md.build_proxy_class(
        node_name="MyNode", meta=m, env_dir=tmp_path,
        package_root=tmp_path, sys_path=[], env_vars={},
    )
    assert "IS_CHANGED" in Proxy.__dict__
    Proxy.IS_CHANGED(x=1)
    assert w.calls[0][2]["method_name"] == "fingerprint_inputs"


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_the_proxy_accepts_inputs_the_author_did_not_name(md, pool, tmp_path, is_v3):  # noqa: F811
    """Catches a named-arg synthesis (the _make_named_validate shape): ComfyUI
    passes every declared input as f(**inputs) and never reads the argspec,
    so the host must not raise TypeError on an unnamed one."""
    w = _Worker(OK)
    pool[str(tmp_path)] = _pool.WorkerRecord(w, 1)
    Proxy = _build_with(md, is_v3, tmp_path, fingerprint_args=["path"])
    fn = Proxy.fingerprint_inputs if is_v3 else Proxy.IS_CHANGED
    assert fn(path="a.obj", seed=1) == "mtime:42"
    assert w.calls[0][2]["kwargs"] == {"path": "a.obj", "seed": 1}


@pytest.mark.parametrize("is_v3", [False, True], ids=["v1", "v3"])
def test_the_proxy_never_calls_get_or_create_worker(md, pool, monkeypatch, tmp_path, is_v3):  # noqa: F811
    """Catches a fingerprint factory that reuses _call_in_worker."""
    from comfy_env.isolation import pool as pool_mod

    def _spawn(*a, **k):
        raise AssertionError("the fingerprint proxy spawned a worker")
    monkeypatch.setattr(pool_mod, "_get_or_create_worker", _spawn)
    Proxy = _build_with(md, is_v3, tmp_path, fingerprint_args=["x"])
    fn = Proxy.fingerprint_inputs if is_v3 else Proxy.IS_CHANGED
    assert _isnan(fn(x=1))
    assert pool == {}


def test_v1_hidden_inputs_ride_the_hidden_field_not_kwargs(md, pool, tmp_path):  # noqa: F811
    """Catches a V1 fingerprint that ships hidden values as kwargs (the
    worker would then hand a V3 fallback node prompt= and die), and one that
    keys on the spelling `unique_id` instead of the sentinel."""
    w = _Worker(OK)
    pool[str(tmp_path)] = _pool.WorkerRecord(w, 1)
    Proxy = _build_with(
        md, False, tmp_path, fingerprint_args=["x"],
        input_types={"required": {"x": ("INT", {})},
                     "hidden": {"prompt": "PROMPT", "node_id": "UNIQUE_ID"}})
    Proxy.IS_CHANGED(x=1, prompt={"7": {"class_type": "KSampler"}}, node_id="7")
    sent = w.calls[0][2]
    forwarded = {s: v for s, _n, v in sent["hidden"]}
    assert forwarded == {"PROMPT": {"7": {"class_type": "KSampler"}},
                         "UNIQUE_ID": "7"}, "hidden inputs were dropped"
    names = {s: n for s, n, _v in sent["hidden"]}
    assert names["UNIQUE_ID"] == "node_id"
    assert "prompt" not in sent["kwargs"]
    assert "node_id" not in sent["kwargs"]


def test_v1_dotted_dynamiccombo_keys_are_nested_like_the_real_call(md, pool, tmp_path):  # noqa: F811
    """Catches a fingerprint factory that skips _shape_v1_kwargs: the real
    function receives `backend` as a nested dict, so the fingerprint must."""
    w = _Worker(OK)
    pool[str(tmp_path)] = _pool.WorkerRecord(w, 1)
    combo = ("COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "grid", "inputs": {"required": {"smooth_normals": ("BOOLEAN", {})}}}]})
    Proxy = _build_with(
        md, False, tmp_path, fingerprint_args=["backend"],
        input_types={"required": {"backend": combo}})
    Proxy.IS_CHANGED(**{"backend": "grid", "backend.smooth_normals": "true"})
    assert w.calls[0][2]["kwargs"] == {
        "backend": {"backend": "grid", "smooth_normals": "true"}}


def test_v3_hidden_are_read_off_the_clone(md, pool, tmp_path):  # noqa: F811
    """Catches a V3 fingerprint that ignores the HiddenHolder ComfyUI hung
    on the per-call clone, or ships dynprompt (a live object)."""
    import types

    w = _Worker(OK)
    pool[str(tmp_path)] = _pool.WorkerRecord(w, 1)
    Proxy = _build_with(md, True, tmp_path, fingerprint_args=["x"])
    Clone = type("Clone", (Proxy,), {})
    Clone.hidden = types.SimpleNamespace(
        prompt={"7": {}}, unique_id="7", dynprompt=_Untouchable())
    assert Clone.fingerprint_inputs(x=1) == "mtime:42"
    assert w.calls[0][2]["hidden"] == [["PROMPT", None, {"7": {}}],
                                       ["UNIQUE_ID", None, "7"]]


def test_v1_and_v3_share_one_kwargs_shaper(md):  # noqa: F811
    """Catches the third copy of the hidden-lift plus DynamicCombo-nesting
    block that the _shape_v1_kwargs extraction exists to prevent: exactly
    one function body in metadata.py both pops from the hidden map and
    splits a key on '.'."""
    tree = ast.parse(Path(md.__file__).read_text(encoding="utf-8"))
    hits = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        pops = splits = False
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                if n.func.attr == "pop":
                    pops = True
                if n.func.attr == "split" and n.args \
                        and isinstance(n.args[0], ast.Constant) and n.args[0].value == '.':
                    splits = True
        if pops and splits:
            hits.append(fn.name)
    assert hits == ["_shape_v1_kwargs"], hits


# --- the worker, for real ---------------------------------------------------

@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="fingerprint-test-worker")
    yield w
    w.shutdown()


def _started(worker):
    assert worker.call_module(module="fingerprint_node", func="ping") == "pong"
    return worker


def _ask(worker, cls, method="IS_CHANGED", kwargs=None, hidden=None):
    return worker.send_command_no_spawn(
        "fingerprint", lock_timeout=0.25, module="fingerprint_node",
        class_name=cls, method_name=method, kwargs=kwargs or {},
        hidden=hidden or [])


def test_a_never_started_worker_is_dead_not_spawned(worker):
    """Catches a no-spawn transport that starts the process to answer."""
    assert worker.is_alive() is False
    assert _ask(worker, "Mtime", kwargs={"path": "a"}) == "dead"
    assert worker.is_alive() is False


def test_worker_calls_the_real_classmethod_with_the_forwarded_kwargs(worker):
    r = _ask(_started(worker), "Mtime", kwargs={"path": "a"})
    assert r["status"] == "ok" and r["value"] == "a"
    r = _ask(worker, "Everything", kwargs={"b": 1, "a": 2})
    assert r["value"] == "a,b"


def test_a_subset_signature_raises_in_the_worker_not_the_host(worker):
    """Catches a handler that filters kwargs to the captured arg names: the
    pack's own signature must be the judge, as it is natively."""
    with pytest.raises(RuntimeError, match="TypeError"):
        _ask(_started(worker), "Picky", kwargs={"path": "a", "seed": 1})
    assert worker.is_alive()


def test_worker_maps_a_non_primitive_return_to_changed(worker):
    r = _ask(_started(worker), "Opaque")
    assert r["status"] == "ok"
    assert r.get("changed") is True and r["value"] is None


def test_worker_maps_nan_to_the_flag_not_the_token(worker):
    """Catches a handler that ships NaN as a float and leans on json's
    non-standard NaN token to carry it."""
    r = _ask(_started(worker), "Nan")
    assert r.get("changed") is True and r["value"] is None
    assert not any(_isnan(v) for v in r.values())
    json.dumps(r, allow_nan=False)


def test_worker_maps_a_raise_to_an_error_frame_and_survives(worker):
    """Catches a handler that lets a pack's exception kill the worker: the
    next real call would then lose the model cache."""
    with pytest.raises(RuntimeError, match="fingerprint exploded"):
        _ask(_started(worker), "Boom")
    assert _ask(worker, "Mtime", kwargs={"path": "b"})["value"] == "b"


def test_worker_maps_an_async_fingerprint_to_an_error_frame(worker):
    """Catches a handler that ships a coroutine object (json refuses it and
    the frame dies) instead of an error frame."""
    with pytest.raises(RuntimeError, match="async"):
        _ask(_started(worker), "Async")
    assert _ask(worker, "Mtime", kwargs={"path": "c"})["value"] == "c"


def test_worker_puts_v1_hidden_back_under_the_authors_name(worker):
    r = _ask(_started(worker), "V1Hidden",
             hidden=[["UNIQUE_ID", "node_id", "7"]])
    assert r["value"] == "7"


def test_worker_hands_v3_hidden_to_the_clone_and_not_the_class(worker):
    """Catches a handler that sets `hidden` on the real class: a credential
    would then outlive the fingerprint that needed it."""
    r = _ask(_started(worker), "V3Shaped", method="fingerprint_inputs",
             hidden=[["UNIQUE_ID", None, "9"]])
    assert r["value"] == "9"
    assert worker.call_module(module="fingerprint_node",
                              func="real_class_hidden") is None


def test_a_busy_worker_answers_busy_within_the_budget(worker):
    """Catches a fingerprint that queues behind a running node instead of
    giving up after the lock budget."""
    _started(worker)
    t = threading.Thread(target=lambda: worker.call_method(
        module_name="fingerprint_node", class_name="Slow", method_name="run",
        kwargs={"seconds": 1.5}, timeout=60.0))
    t.start()
    try:
        time.sleep(0.2)                      # let the real call take the lock
        t0 = time.monotonic()
        r = _ask(worker, "Mtime", kwargs={"path": "a"})
        elapsed = time.monotonic() - t0
    finally:
        t.join()
    assert r == "busy"
    assert elapsed < 0.5


# --- seam guards ------------------------------------------------------------

def _worker_tree():
    tree = ast.parse(_worker_source())
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    return tree


def _callee_name(call):
    f = call.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def test_fingerprint_is_dispatched_in_the_method_branch_before_the_type_branch():
    """Catches a handler wired into the call_method path, where the marks
    preamble, shm reconstruction and state sync run: a fingerprint asked
    before the prompt's first node would retire the previous prompt's pins."""
    tree = _worker_tree()
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    dispatch = [n.lineno for n in ast.walk(main)
                if isinstance(n, ast.Compare)
                and isinstance(n.left, ast.Call)
                and _callee_name(n.left) == "request.get"
                and n.left.args and isinstance(n.left.args[0], ast.Constant)
                and n.left.args[0].value == "method"
                and any(isinstance(c, ast.Constant) and c.value == "fingerprint"
                        for c in n.comparators)]
    request_type = [n.lineno for n in ast.walk(main)
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "request_type"
                            for t in n.targets)]
    assert len(dispatch) == 1, dispatch
    assert request_type, "the type branch's request_type assignment moved"
    assert dispatch[0] < min(request_type), (
        f"fingerprint dispatched at {dispatch[0]}, after the type branch at "
        f"{min(request_type)}")


def test_fingerprint_handler_touches_no_call_method_machinery():
    """Catches a handler that reconstructs shm, seeds an instance or diffs
    state for what is a class-level JSON question."""
    tree = _worker_tree()
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_handle_fingerprint")
    banned = {"_prompt_marks_preamble", "_from_shm", "_deserialize_isolated_objects",
              "_infer_mode", "object.__new__", "decode_state", "diff_state"}
    called = {_callee_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert not (called & banned), called & banned


def test_forward_fingerprint_never_names_a_spawning_call(md):  # noqa: F811
    """Catches a ladder rung that starts a worker, queues on a real call, or
    runs the IPC serializer over an input."""
    tree = ast.parse(Path(md.__file__).read_text(encoding="utf-8"))
    banned = {"_get_or_create_worker", "send_command", "call_method",
              "_call_in_worker", "_ensure_started", "prepare_for_ipc_recursive"}
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("_forward_fingerprint", "_make_v3_fingerprint", "_make_v1_fingerprint"):
        assert name in fns, f"{name} is gone"
        called = {_callee_name(n).split(".")[-1]
                  for n in ast.walk(fns[name]) if isinstance(n, ast.Call)}
        assert not (called & banned), (name, called & banned)


def test_the_startup_warning_that_called_this_a_loss_is_gone(md):  # noqa: F811
    """Catches the stale WARNING surviving: it told users the fingerprint is
    not forwarded, which is now false."""
    src = inspect.getsource(md._warn_node_conformance)
    assert "fingerprint_args" not in src
    assert "NOT forwarded" not in src


# --- the scan ---------------------------------------------------------------

def test_scan_records_fingerprint_args_for_a_v1_fingerprint_and_none_without(tmp_path):
    """Catches a scan that records [] for a node with no fingerprint: [] is
    truthy for the attach gate (`is not None`) and would put a NaN-answering
    method on every isolated node."""
    proc, out = _run_scan(tmp_path, """
        class Stamped:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"path": ("STRING", {})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
            @classmethod
            def IS_CHANGED(cls, path):
                return path
        class Plain:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"x": ("INT", {})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"Stamped": Stamped, "Plain": Plain}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["nodes"]["Stamped"]["fingerprint_args"] == ["path"]
    assert payload["nodes"]["Plain"]["fingerprint_args"] is None
