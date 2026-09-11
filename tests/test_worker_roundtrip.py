"""Contract: data sent through a worker comes back equal; failures are loud.

These tests run the REAL worker (_persistent_worker.py) in a subprocess using
the current interpreter -- no pixi env needed. This is the don't-break-
userspace test of the IPC layer: it survives any serialization rewrite and
fails only when the wire contract actually breaks.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker, _PERSISTENT_WORKER_SCRIPT

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    w = SubprocessWorker(
        python=sys.executable,
        working_dir=FIXTURES,
        name="test-worker",
    )
    yield w
    w.shutdown()


def test_roundtrip_primitives_and_nesting(worker):
    payload = {"a": [1, 2.5, "x", True, None], "b": {"c": "d"}, "empty": []}
    assert worker.call_module(module="echo_node", func="echo", value=payload) == payload


def test_roundtrip_torch_tensor(worker):
    import torch
    result = worker.call_module(module="echo_node", func="make_tensor", rows=4, cols=8)
    expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    assert torch.equal(result, expected)


def test_roundtrip_numpy_values(worker):
    import numpy as np
    sent = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = worker.call_module(module="echo_node", func="echo", value=sent)
    got = np.asarray(result)
    assert got.shape == sent.shape
    assert np.array_equal(got, sent)


def test_worker_error_propagates(worker):
    from comfy_env.isolation.workers import WorkerError
    with pytest.raises(WorkerError):
        worker.call_module(module="echo_node", func="no_such_function")


def test_serialize_failure_is_loud(worker):
    # A result pickle can't handle (lambda) must produce a named error
    # pointing at the type -- never leak the raw object into the JSON
    # message (the old fallback crashed two layers away with
    # "not JSON serializable").
    from comfy_env.isolation.workers import WorkerError
    with pytest.raises(WorkerError, match="cannot serialize"):
        worker.call_module(module="echo_node", func="make_unpicklable")


def test_parent_serialize_failure_is_loud():
    from comfy_env.isolation.workers.subprocess import _to_shm
    with pytest.raises(TypeError, match="cannot serialize 'function'"):
        _to_shm(lambda: 1, [])


def test_crash_is_loud(worker):
    with pytest.raises(RuntimeError, match="died|closed"):
        worker.call_module(module="echo_node", func="crash")
    assert not worker.is_alive()


def test_timeout_kills_worker(worker):
    with pytest.raises(TimeoutError):
        worker.call_module(module="echo_node", func="slow", seconds=60, timeout=3)


def test_worker_imports_shared_constants():
    # The worker takes the faulthandler filename and retention window FROM
    # _ipc_shared rather than repeating the literals.
    assert "import _ipc_shared" in _PERSISTENT_WORKER_SCRIPT
    assert "_ipc_shared.WORKER_FAULTHANDLER_BASENAME" in _PERSISTENT_WORKER_SCRIPT
    assert "_ipc_shared.TENSOR_KEEPER_TTL" in _PERSISTENT_WORKER_SCRIPT


def test_no_keeper_hardcodes_its_retention_window():
    """The check above greps the WORKER script, so it cannot see the others.

    It carried a comment claiming the 30s-vs-60s drift was "structurally gone".
    It was not: tensor_utils.TensorKeeper still defaulted to a literal 30.0
    while the other three keepers took TENSOR_KEEPER_TTL (60.0) -- a regression
    test asserting a property it is structurally incapable of observing.
    """
    import ast
    from pathlib import Path

    import comfy_env

    root = Path(comfy_env.__file__).parent
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for arg, default in zip(reversed(node.args.args),
                                    reversed(node.args.defaults)):
                if arg.arg == "retention_seconds" and isinstance(default, ast.Constant):
                    offenders.append(
                        (path.relative_to(root).as_posix(), node.lineno, default.value))

    assert not offenders, (
        f"keeper(s) hardcode a retention window instead of TENSOR_KEEPER_TTL: "
        f"{offenders}"
    )


def test_node_state_survives_with_its_types(worker):
    """The suite's first real-worker `call_method`, and the reason it exists.

    The shippability gate asked `pickle.dumps` while the wire was `json.dumps`
    with no `default=`, so a picklable-but-not-JSON attribute was promised as
    shippable and then killed the call. Values that did cross were retyped in
    silence: a tuple came back a list, an int key came back a string.

    Two calls, because the defect is only visible on the second: state has to
    make the round trip out to the parent and back before anything can be
    wrong about it.
    """
    common = dict(module_name="state_node", class_name="StateNode",
                  method_name="run")

    first = worker.call_method(**common, self_state={}, state_id="s1")
    assert first["calls"] == 1

    # what the parent now holds, handed straight back as the next call's state
    carried = worker._last_state_out["set"]

    second = worker.call_method(**common, self_state=carried, state_id="s1")
    assert second["calls"] == 2, "self.x did not survive the boundary"
    assert second["tup"] == [1, 2, 3] or second["tup"] == (1, 2, 3)
    assert second["tup_type"] == "tuple", "a tuple came back as a list"
    assert second["by_int_keys"] == ["int"], "an int key came back as a string"
    assert second["blob_type"] == "bytes", "bytes did not survive the wire"
    assert second["tags_type"] == "set", "a set did not survive the wire"


def _restart_in_place(worker):
    """The socket-unhealthy path: `_ensure_started` kills and respawns
    without telling anyone. No pool generation bump, no parent-side signal;
    the only thing that changes is which process answers the next call."""
    worker._kill_worker()
    worker._process = None


def test_init_reruns_after_restart_when_state_was_process_bound(worker):
    """A node whose __init__ builds a thread pool survives a worker restart.

    Before: the parent's seed flag was set once and never cleared, so the
    new process got the old markers and no __init__, then raised "its value
    is gone and will be recomputed" on every call until ComfyUI restarted.
    Nothing ever recomputed it.
    """
    common = dict(module_name="restart_node", class_name="RestartNode",
                  method_name="run")
    parent = {}

    def call(**kw):
        from comfy_env import state_sync
        r = worker.call_method(**common, self_state=dict(parent), state_id="r1", kwargs=kw)
        state_sync.apply_state_out(parent, worker._last_state_out)
        return r

    assert call(key="a")["calls"] == 1
    assert call(key="b")["calls"] == 2
    # the parent holds markers for the two unpicklables, live values for the rest
    from comfy_env.state_sync import is_overflow_marker
    assert is_overflow_marker(parent["executor"]) and is_overflow_marker(parent["lock"])
    assert parent["calls"] == 2 and sorted(parent["cache"]) == ["a", "b"]
    old_gen = parent["executor"]["gen"]

    _restart_in_place(worker)

    third = call(key="c")
    # __init__ re-ran (a fresh executor answered), and because a marker had
    # to be dropped the fresh instance is the whole truth: calls restarts
    # at 1 and the old cache is gone, not overlaid next to a reset counter
    assert third["calls"] == 1, "__init__ did not re-run after the restart"
    assert third["squared"] == 1
    assert third["cache_keys"] == ["c"], "stale state was overlaid on the fresh __init__"
    # the parent's copy was repaired, not left holding a dead marker
    assert parent["executor"]["gen"] != old_gen
    assert parent["calls"] == 1 and sorted(parent["cache"]) == ["c"]

    # and the new process now remembers the instance: no second __init__
    assert call(key="d")["calls"] == 2


def test_init_reruns_after_restart_but_plain_state_is_kept(worker):
    """A restart costs a node exactly what it cannot carry, and nothing more:
    with no process-bound state, the parent's copy is overlaid on the fresh
    __init__ and the counter continues."""
    common = dict(module_name="restart_node", class_name="PlainNode",
                  method_name="run")
    parent = {}

    def call():
        from comfy_env import state_sync
        r = worker.call_method(**common, self_state=dict(parent), state_id="p1")
        state_sync.apply_state_out(parent, worker._last_state_out)
        return r

    assert call()["calls"] == 1
    assert call()["calls"] == 2
    _restart_in_place(worker)
    assert call()["calls"] == 3, "a restart reset state that could have been carried"


def test_in_place_mutation_ships(worker):
    """`self.cache[k] = v` reaches the parent.

    Found by the restart test above. The worker fingerprinted the inbound
    state at diff time, after the call, against the very dict the instance
    had just mutated: identical bytes on both sides, nothing shipped. The
    unit test that claimed to cover this copied the inner dict itself, which
    production never did. A counter (`self.calls += 1` rebinds) shipped; a
    dict or list mutated in place (the common shape of a node cache) did not.
    """
    common = dict(module_name="restart_node", class_name="RestartNode",
                  method_name="run")
    parent = {}
    from comfy_env import state_sync
    for key in ("a", "b", "c"):
        worker.call_method(**common, self_state=dict(parent), state_id="m1",
                           kwargs={"key": key})
        state_sync.apply_state_out(parent, worker._last_state_out)
    assert sorted(parent["cache"]) == ["a", "b", "c"]
