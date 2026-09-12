"""Contract: a busy worker still answers cheap questions.

The main lane is one request at a time and its reader is whoever holds the
call lock, so while a node ran nobody was listening: dropdown refreshes and
fingerprints answered "busy" for the length of the node, and /object_info
during a run showed the startup list. Native ComfyUI runs INPUT_TYPES and
VALIDATE_INPUTS on the server thread concurrently with execution; the
single lane was a comfy-env transport artifact.

The worker now opens a second connection after its ready frame, read by a
daemon thread that serves an allowlist (ping, refresh_input_types,
fingerprint) and nothing else, never imports a pack module, and never
touches the main socket. The host's send_side has its own lock, monotonic
ids, a back-off after a timeout, and no way to kill the worker.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md
import comfy_env.isolation.pool as pool
from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    (tmp_path / "a.obj").write_text("")
    monkeypatch.setenv("SLOW_NODE_DIR", str(tmp_path))
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES, name="side-lane")
    w._ensure_started()
    yield w, tmp_path
    w.shutdown()


def _run_slow(w, cls="Slow", seconds=2.0):
    t = threading.Thread(target=lambda: w.call_method(
        module_name="slow_node", class_name=cls, method_name="run",
        kwargs={"seconds": seconds}, timeout=60.0))
    t.start()
    time.sleep(0.3)          # let the main lane become busy
    return t


def test_lane_is_connected_and_idle_ping_is_fast(worker):
    w, _ = worker
    assert w._side_transport is not None
    t0 = time.perf_counter()
    assert w.send_side("ping")["status"] == "pong"
    assert time.perf_counter() - t0 < 0.1


def test_side_answers_while_the_main_lane_is_busy(worker):
    """The contract. The same instant the main lane says busy, the side
    lane answers a real INPUT_TYPES re-run with fresh options."""
    w, d = worker
    # warm the module through a real call first: the side lane never imports
    w.call_method(module_name="slow_node", class_name="Slow", method_name="run",
                  kwargs={"seconds": 0.0}, timeout=30.0)
    t = _run_slow(w)
    try:
        assert w.send_command_no_spawn("ping", lock_timeout=0.05) == "busy"
        (d / "b.obj").write_text("")
        t0 = time.perf_counter()
        r = w.send_side("refresh_input_types", module="slow_node", class_name="Slow")
        assert time.perf_counter() - t0 < 0.5, "a side request waited on the main lane"
        assert r["status"] == "ok" and r["options"]["required"]["mesh"] == ["a.obj", "b.obj"]
    finally:
        t.join()


def test_side_answers_under_a_cpu_bound_node(worker):
    """Pure-Python spin on the main thread: the side reply rides the GIL
    switch interval, tens of milliseconds, not the node's duration."""
    w, _ = worker
    w.call_method(module_name="slow_node", class_name="Spinner", method_name="run",
                  kwargs={"seconds": 0.0}, timeout=30.0)
    t = _run_slow(w, cls="Spinner", seconds=1.5)
    try:
        t0 = time.perf_counter()
        assert w.send_side("ping")["status"] == "pong"
        assert time.perf_counter() - t0 < 0.5
    finally:
        t.join()


def test_a_module_not_loaded_by_a_real_call_is_a_miss(worker):
    """The side lane never imports: import is the expensive, side-effecting
    half and belongs to the main lane's first call."""
    w, _ = worker
    r = w.send_side("refresh_input_types", module="slow_node", class_name="Slow")
    assert r["status"] == "miss"


def test_only_the_allowlist_is_served(worker):
    w, _ = worker
    r = w.send_side("full_release")
    assert r["status"] == "error" and "not served on the side lane" in r["error"]
    # and the main loop still serves it
    assert isinstance(w.send_command_no_spawn("ping", lock_timeout=2.0), dict)


def test_a_timeout_neither_kills_nor_poisons_the_lane(worker, monkeypatch):
    """After one abandoned request the next one must not read the stale
    reply, and the worker must be alive throughout."""
    w, _ = worker
    w.call_method(module_name="slow_node", class_name="Slow", method_name="run",
                  kwargs={"seconds": 0.0}, timeout=30.0)
    # a fingerprint that sleeps 0.5 s, asked with a 0.1 s cap: abandoned
    r = w.send_side("fingerprint", reply_timeout=0.1, module="slow_node",
                    class_name="Sleeper", method_name="IS_CHANGED", kwargs={}, hidden=[])
    assert r == "timeout"
    assert w.is_alive()
    assert w.send_side("ping") == "slow", "back-off after a timeout"
    monkeypatch.setattr(w, "_side_unresponsive_until", 0.0)
    # the abandoned reply arrives first on the wire; it must be discarded
    r2 = w.send_side("ping")
    assert r2["status"] == "pong" and r2["side_id"] == w._side_id, "a stale reply was returned"


def test_send_side_never_spawns_and_never_names_the_pool_lock():
    import ast, inspect, textwrap
    src = textwrap.dedent(inspect.getsource(SubprocessWorker.send_side))
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"_ensure_started", "_get_or_create_worker", "_POOL_LOCK", "_kill_tree", "_kill_worker", "kill"}


def test_a_never_started_worker_is_dead_on_the_side_lane():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES, name="cold")
    try:
        assert w.send_side("ping") == "dead"
    finally:
        w.shutdown()


def test_call_parent_refuses_other_threads():
    """The main socket has one reader per side; a callback from a second
    thread would desync it. Pinned by AST on the worker source."""
    import ast
    from comfy_env.isolation.workers.subprocess import _PERSISTENT_WORKER_SCRIPT
    tree = ast.parse(_PERSISTENT_WORKER_SCRIPT)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_call_parent")
    src = ast.unparse(fn)
    assert "main_thread()" in src and "raise RuntimeError" in src


# --- the ladders use the lane, and /object_info batches per env -------------

class _SideWorker:
    def __init__(self):
        self.side_calls = []
        self.main_calls = []
        self.side_reply = {"status": "ok", "options": {}}

    def send_side(self, method, lock_timeout=0.25, reply_timeout=1.0, **params):
        self.side_calls.append((method, params))
        return self.side_reply

    def send_command_no_spawn(self, method, lock_timeout=2.0, **params):
        self.main_calls.append((method, params))
        return {"status": "ok", "options": {"required": {"mesh": ["main"]}}}


@pytest.fixture()
def fake_pool(monkeypatch):
    monkeypatch.setattr(pool, "_WORKER_POOL", {})
    monkeypatch.setattr(md, "_ENV_CLASSES", {})
    monkeypatch.setattr(md, "_PASS_MEMO", {})
    w = _SideWorker()
    pool._WORKER_POOL["/env"] = pool.WorkerRecord(w, 1)
    return w


def test_refresh_prefers_the_side_lane(fake_pool):
    fake_pool.side_reply = {"status": "ok", "options": {"required": {"mesh": ["side"]}}}
    assert md._refresh_combo_options("/env", "m", "C") == {"required": {"mesh": ["side"]}}
    assert fake_pool.main_calls == []


def test_nolane_falls_back_to_the_main_lane(fake_pool):
    fake_pool.side_reply = "nolane"
    assert md._refresh_combo_options("/env", "m", "C") == {"required": {"mesh": ["main"]}}


def test_any_other_side_sentinel_is_a_miss_not_a_fallback(fake_pool):
    for sentinel in ("busy", "slow", "timeout", "dead"):
        fake_pool.side_reply = sentinel
        assert md._refresh_combo_options("/env", "m", "C") is None
    assert fake_pool.main_calls == [], "a side miss must not queue on the main lane"


def test_an_object_info_pass_asks_once_per_env(fake_pool, monkeypatch):
    """node_info calls INPUT_TYPES twice per node; forty isolated nodes used
    to be forty round trips on the event loop. Inside a pass (upstream's
    cache_helper.active) it is one request per env."""
    import types
    fp = types.SimpleNamespace(cache_helper=types.SimpleNamespace(active=True))
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    md._ENV_CLASSES["/env"] = {("m", "A"), ("m", "B"), ("n", "C")}
    fake_pool.side_reply = {"status": "ok", "options": {
        "m:A": {"required": {"x": ["1"]}}, "m:B": {"required": {"y": ["2"]}}, "n:C": {}}}
    assert md._refresh_combo_options("/env", "m", "A") == {"required": {"x": ["1"]}}
    assert md._refresh_combo_options("/env", "m", "B") == {"required": {"y": ["2"]}}
    assert md._refresh_combo_options("/env", "n", "C") is None
    assert md._refresh_combo_options("/env", "m", "A") == {"required": {"x": ["1"]}}
    assert len(fake_pool.side_calls) == 1
    method, params = fake_pool.side_calls[0]
    assert method == "refresh_input_types"
    assert sorted(map(tuple, params["classes"])) == [("m", "A"), ("m", "B"), ("n", "C")]


def test_builders_register_their_classes_per_env(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "_ENV_CLASSES", {})
    meta = {"function": "run", "category": "t", "output_node": False, "return_types": ["INT"],
            "return_names": [], "input_types": {"required": {"x": ("INT", {})}},
            "module_name": "mod", "class_name": "Cls", "is_v3": False}
    md.build_proxy_class(node_name="N", meta=meta, env_dir=tmp_path, package_root=tmp_path,
                         sys_path=[], env_vars={})
    assert md._ENV_CLASSES[str(tmp_path)] == {("mod", "Cls")}
