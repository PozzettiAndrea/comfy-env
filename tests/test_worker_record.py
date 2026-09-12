"""Contract: when a worker process is replaced, nothing the pool knew about
that process survives it.

Three paths replace a process: _remove_worker after a crash, the dead-worker
branch of _get_or_create_worker, and the socket-unhealthy restart inside
_ensure_started (same SubprocessWorker object, fires _on_restart). They used
to clear different subsets of the pool's per-env ledgers. _LAST_PROMPT and
_PIN_REGRESSION_SEEN were cleared by nobody, so a replacement under the same
key inherited a stale "last logged" figure; the dead branch left the old
process's reserve charge on the new one; the restart path left the last VRAM
report on the worker object, booking phantom residency until the next frame.

The test is structural on purpose: it asserts the key is gone from EVERY
module-level dict or set in pool, so the next ledger someone adds is covered
without editing this file.
"""

import threading

import pytest

import comfy_env.isolation.pool as pool


class _FakeWorker:
    def __init__(self, alive=True):
        self._alive = alive
        self._on_restart = None
        self._last_vram_report = {"held": 5}
        self._last_held_bytes = 5
        self._pin_release_deferred = 7
        self._calls_in_flight = 0
        self._mem_lock = threading.Lock()
        self.memory_manager = {"manager": "ledger"}
        self.name = "fake"

    def is_alive(self):
        return self._alive

    def shutdown(self):
        self._alive = False

    def register_callback(self, *a, **k):
        pass

    def verify_transport(self):
        pass


def _ledgers():
    """Every module-level dict or set in pool, by name."""
    return {n: v for n, v in vars(pool).items()
            if isinstance(v, (dict, set)) and n.startswith("_") and n.isupper()}


def _seed(key, worker):
    """Put the key into every ledger the way the running code would."""
    pool._WORKER_POOL[key] = pool.WorkerRecord(worker, 1, held=123, last_activity=1.0,
                                               last_prompt="p1", mm_reported=True)
    pool._WORKER_PATCHERS[key] = {"m": object()}
    pool._PIN_REPORTS[key] = {"pinned": 1}
    pool._OVERHEAD_REPORTS[key] = {"excess": 1}
    pool._PIN_REGRESSION_SEEN[key] = 1


def _keyed_anywhere(key):
    return sorted(n for n, v in _ledgers().items() if key in v)


@pytest.fixture()
def clean(monkeypatch):
    for n, v in _ledgers().items():
        monkeypatch.setattr(pool, n, type(v)())
    monkeypatch.setattr(pool, "_STALE_PATCHERS", [])
    yield


def test_crash_path_forgets_everything(clean):
    w = _FakeWorker()
    _seed("/env/a", w)
    pool._remove_worker("/env/a")
    assert _keyed_anywhere("/env/a") == []
    assert w._last_vram_report is None and w._pin_release_deferred is None


def test_restart_path_forgets_everything(clean):
    """The socket-unhealthy restart keeps the worker OBJECT and fires
    _on_restart; the process behind it is new."""
    w = _FakeWorker()
    _seed("/env/a", w)
    w._on_restart = lambda: pool._cleanup_stale_patchers("/env/a", w)
    w._on_restart()
    # the pool entry itself is kept (same object, new process); every
    # per-process ledger is gone, and the patchers moved to the stale list
    assert _keyed_anywhere("/env/a") == ["_WORKER_POOL"]
    rec = pool._WORKER_POOL["/env/a"]
    assert (rec.held, rec.last_activity, rec.last_prompt, rec.mm_reported) == (0, None, None, False)
    assert rec.worker is w, "the record is replaced, the worker object is kept"
    assert len(pool._STALE_PATCHERS) == 1
    assert w._last_vram_report is None and w._last_held_bytes is None


def test_dead_branch_forgets_everything(clean, monkeypatch, tmp_path):
    """_get_or_create_worker finds a dead worker and replaces it."""
    dead = _FakeWorker(alive=False)
    _seed(str(tmp_path), dead)
    fresh = _FakeWorker()
    monkeypatch.setattr(pool, "_create_worker", lambda *a, **k: fresh)
    for fn in ("_check_host_contract", "_start_idle_sweep", "_install_pressure_hook",
               "_report_memory_manager"):
        monkeypatch.setattr(pool, fn, lambda *a, **k: None)
    worker, gen = pool._get_or_create_worker(tmp_path, tmp_path, [], {}, 5.0)
    assert worker is fresh
    assert _keyed_anywhere(str(tmp_path)) == ["_WORKER_POOL"], \
        "the replacement inherited the dead process's ledgers"
    rec = pool._WORKER_POOL[str(tmp_path)]
    assert rec.worker is fresh and rec.held == 0 and rec.mm_reported is False


def test_record_unpacks_like_the_old_tuple(clean):
    w = _FakeWorker()
    pool._WORKER_POOL["k"] = pool.WorkerRecord(w, 3)
    worker, gen = pool._WORKER_POOL["k"]
    assert worker is w and gen == 3 and pool._WORKER_POOL["k"][0] is w


def test_respawn_replaces_the_record_object(clean):
    """Readers that snapshotted the pool keep a consistent pair; the new
    process gets a new row."""
    w = _FakeWorker()
    old = pool.WorkerRecord(w, 1, held=9)
    pool._WORKER_POOL["k"] = old
    pool._cleanup_stale_patchers("k", w)
    assert pool._WORKER_POOL["k"] is not old
    assert old.held == 9, "the retired row is left alone, not zeroed under a reader"


def test_metadata_reaches_the_pool_only_through_the_accessor():
    import ast
    from pathlib import Path
    import comfy_env.isolation.metadata as md
    tree = ast.parse(Path(md.__file__).read_text(encoding="utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "pool":
            names |= {a.name for a in n.names}
    assert not any(x.startswith("_WORKER") for x in names), names
    assert "worker_for" in names


def test_worker_for_never_creates(clean):
    assert pool.worker_for("/nope") is None and pool._WORKER_POOL == {}


# --- which prompt is running -------------------------------------------------

class _Queue:
    def __init__(self):
        self.currently_running = {}


@pytest.fixture()
def host_queue(monkeypatch):
    import types
    q = _Queue()
    mod = types.ModuleType("server")
    mod.PromptServer = type("PromptServer", (), {"instance": types.SimpleNamespace(prompt_queue=q)})
    monkeypatch.setitem(__import__("sys").modules, "server", mod)
    return q


def test_current_prompt_is_the_queues_running_item(host_queue):
    host_queue.currently_running[0] = (1, "prompt-abc", {}, {}, [])
    assert pool._current_prompt() == "prompt-abc"


def test_current_prompt_is_none_when_nothing_runs_even_if_the_registry_is_stale(host_queue, monkeypatch):
    """The bug: the progress registry keeps the last prompt id forever, so
    the idle sweep believed a finished prompt was still running."""
    import types
    stale = types.SimpleNamespace(prompt_id="finished-long-ago")
    monkeypatch.setitem(__import__("sys").modules, "comfy_execution.progress",
                        types.SimpleNamespace(get_progress_state=lambda: stale))
    assert pool._current_prompt() is None


import os
COMFYUI_DIR = os.environ.get("COMFYUI_DIR")


@pytest.mark.comfyui
@pytest.mark.skipif(not COMFYUI_DIR, reason="COMFYUI_DIR not set")
def test_upstream_queue_fills_and_empties_currently_running():
    """Pins the upstream structure _current_prompt reads: get() populates
    currently_running with (number, prompt_id, ...) and task_done() pops."""
    import sys
    import types
    sys.path.insert(0, COMFYUI_DIR)
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        if "--cpu" not in sys.argv:
            sys.argv = [sys.argv[0], "--cpu"]
        import execution
        q = execution.PromptQueue(types.SimpleNamespace(queue_updated=lambda: None))
        q.put((0, "prompt-xyz", {}, {}, []))
        item, item_id = q.get(timeout=1)
        assert item[1] == "prompt-xyz"
        assert [v[1] for v in q.currently_running.values()] == ["prompt-xyz"]
        q.task_done(item_id, {}, status=None)
        assert q.currently_running == {}
    finally:
        sys.path.remove(COMFYUI_DIR)
