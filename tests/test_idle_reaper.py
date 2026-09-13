"""Contract: a worker idle past the window, holding nothing and serving no
model stand-in, loses its process (ADR-0019). Its next use is a cold start.
"""
import ast
import inspect
import sys
import textwrap
import time
from pathlib import Path

import pytest

import comfy_env.isolation.pool as pool
from comfy_env import state_sync
from test_worker_record import _FakeWorker, _keyed_anywhere, _seed, clean  # noqa: F401

FIXTURES = Path(__file__).parent / "fixtures"


def _state(**over):
    base = {"alive": True, "in_flight": False, "idle_since": 0.0, "holding": False,
            "last_prompt": "p1", "registered_models": False}
    base.update(over)
    return base


class TestPlanner:
    def test_quiet_empty_unregistered_worker_is_reaped(self):
        assert state_sync.plan_idle_reap({"a": _state()}, now=100.0, min_idle=50) == ["a"]

    def test_not_yet_idle_enough(self):
        assert state_sync.plan_idle_reap({"a": _state(idle_since=80.0)}, now=100.0, min_idle=50) == []

    @pytest.mark.parametrize("field", ["holding", "in_flight", "registered_models"])
    def test_holding_busy_or_registered_is_spared(self, field):
        assert state_sync.plan_idle_reap({"a": _state(**{field: True})}, now=100.0, min_idle=50) == []

    def test_dead_and_never_seen_are_skipped(self):
        w = {"a": _state(alive=False), "b": _state(idle_since=None)}
        assert state_sync.plan_idle_reap(w, now=100.0, min_idle=50) == []

    def test_the_running_prompts_worker_is_spared(self):
        w = {"a": _state(last_prompt="p1"), "b": _state(last_prompt="p0")}
        assert state_sync.plan_idle_reap(w, now=100.0, min_idle=50, current_prompt="p1") == ["b"]

    @pytest.mark.parametrize("window", [0, -1, None])
    def test_disabled(self, window):
        assert state_sync.plan_idle_reap({"a": _state()}, now=100.0, min_idle=window) == []


@pytest.fixture()
def no_prompt(monkeypatch):
    monkeypatch.setattr(pool, "_current_prompt", lambda: None)


def test_reap_removes_the_record_shuts_the_worker_and_forgets_its_ledgers(clean, no_prompt, monkeypatch, tmp_path):
    w = _FakeWorker()
    key = str(tmp_path)
    _seed(key, w)
    pool._WORKER_PATCHERS[key] = {}           # nothing registered
    rec = pool._WORKER_POOL[key]
    rec.held = 0
    rec.last_activity = time.monotonic() - 100
    monkeypatch.setenv(pool.IDLE_REAP_ENV_VAR, "10")
    pool._reap_idle_workers()
    assert key not in pool._WORKER_POOL and not w.is_alive()
    assert _keyed_anywhere(key) == [], "the reaped process left ledgers behind"


def test_a_worker_with_a_stand_in_on_the_books_is_never_reaped(clean, no_prompt, monkeypatch, tmp_path):
    w = _FakeWorker()
    key = str(tmp_path)
    _seed(key, w)                              # _seed registers one patcher
    rec = pool._WORKER_POOL[key]
    rec.held = 0
    rec.last_activity = time.monotonic() - 100
    monkeypatch.setenv(pool.IDLE_REAP_ENV_VAR, "10")
    pool._reap_idle_workers()
    assert pool._WORKER_POOL[key] is rec and w.is_alive()


def test_zero_disables_and_garbage_disables(clean, no_prompt, monkeypatch, tmp_path):
    w = _FakeWorker()
    key = str(tmp_path)
    pool._WORKER_POOL[key] = pool.WorkerRecord(w, 1, held=0, last_activity=time.monotonic() - 1e6)
    for raw in ("0", "banana"):
        monkeypatch.setenv(pool.IDLE_REAP_ENV_VAR, raw)
        pool._reap_idle_workers()
        assert pool._WORKER_POOL[key] is not None and w.is_alive()


def test_default_window_is_half_an_hour(monkeypatch):
    monkeypatch.delenv(pool.IDLE_REAP_ENV_VAR, raising=False)
    assert pool._idle_reap_seconds() == 1800.0


def test_release_runs_before_reap_every_sweep():
    src = textwrap.dedent(inspect.getsource(pool._idle_sweep_loop))
    calls = [n.func.id for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert calls.index("_release_idle_workers") < calls.index("_reap_idle_workers")


def test_reaping_a_real_worker_and_the_next_call_starts_a_fresh_one(clean, no_prompt, monkeypatch, tmp_path):
    """End to end through the pool with a real process: the reap kills the
    process, the next _get_or_create_worker spawns a new one under a bumped
    generation, and it answers."""
    from comfy_env.isolation.workers.subprocess import SubprocessWorker
    monkeypatch.setattr(pool, "_create_worker", lambda env_dir, working_dir, *a, **k:
                        SubprocessWorker(python=sys.executable, working_dir=working_dir, name="reap"))
    for fn in ("_check_host_contract", "_start_idle_sweep", "_install_pressure_hook",
               "_report_memory_manager"):
        monkeypatch.setattr(pool, fn, lambda *a, **k: None)
    key = str(tmp_path)
    w1, g1 = pool._get_or_create_worker(tmp_path, FIXTURES, [], {}, 30.0)
    try:
        pid1 = w1._process.pid
        rec = pool._WORKER_POOL[key]
        rec.held = 0
        rec.last_activity = time.monotonic() - 100
        monkeypatch.setenv(pool.IDLE_REAP_ENV_VAR, "10")
        pool._reap_idle_workers()
        assert key not in pool._WORKER_POOL and not w1.is_alive()
        deadline = time.time() + 10
        while time.time() < deadline and _pid_alive(pid1):
            time.sleep(0.05)
        assert not _pid_alive(pid1), "the reaped process is still running"
        w2, g2 = pool._get_or_create_worker(tmp_path, FIXTURES, [], {}, 30.0)
        assert w2 is not w1 and g2 > g1
        assert w2.send_side("ping")["status"] == "pong"
    finally:
        for rec in list(pool._WORKER_POOL.values()):
            rec.worker.shutdown()


def _pid_alive(pid):
    import os
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/status") as f:
            return "zombie" not in f.read().split("State:")[1].split("\n")[0].lower()
    except OSError:
        return True
