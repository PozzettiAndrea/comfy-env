"""Contract: the pool lock guards membership, not the spawn.

_get_or_create_worker used to hold _POOL_LOCK across process start, the
ready frame and the transport canary, so a prompt touching two cold envs
started them one after the other, and a caller for a warm env could not
even look its worker up until the other env's spawn finished. The spawn
now runs with the lock released; a concurrent caller for the SAME env
waits on the owner's future in _WARMING instead of spawning a twin.
"""
import ast
import inspect
import textwrap
import threading
import time

import pytest

import comfy_env.isolation.pool as pool
from test_worker_record import _FakeWorker, clean  # noqa: F401  (fixture)


@pytest.fixture()
def quiet(monkeypatch):
    for fn in ("_check_host_contract", "_start_idle_sweep", "_install_pressure_hook",
               "_report_memory_manager"):
        monkeypatch.setattr(pool, fn, lambda *a, **k: None)


def _get(path):
    return pool._get_or_create_worker(path, path, [], {}, 5.0)


def test_two_envs_spawn_at_the_same_time(clean, quiet, monkeypatch, tmp_path):
    """Both fake spawns must be inside _create_worker together. If the lock
    were still held across the spawn the barrier would time out."""
    gate = threading.Barrier(2, timeout=5.0)

    def create(env_dir, *a, **k):
        gate.wait()
        return _FakeWorker()

    monkeypatch.setattr(pool, "_create_worker", create)
    results = {}

    def run(name):
        results[name] = _get(tmp_path / name)

    threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert set(results) == {"a", "b"} and results["a"][0] is not results["b"][0]
    assert results["a"][1] != results["b"][1]
    assert pool._WARMING == {}


def test_same_env_spawns_once_and_the_waiter_gets_the_same_worker(clean, quiet, monkeypatch, tmp_path):
    calls = []

    def create(env_dir, *a, **k):
        calls.append(1)
        time.sleep(0.3)
        return _FakeWorker()

    monkeypatch.setattr(pool, "_create_worker", create)
    results = []
    threads = [threading.Thread(target=lambda: results.append(_get(tmp_path))) for _ in range(3)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join(10)
    assert len(calls) == 1, "concurrent callers for one env spawned twins"
    assert len(results) == 3 and all(r == results[0] for r in results)
    assert pool._WARMING == {}


def test_a_warm_env_is_not_queued_behind_another_envs_spawn(clean, quiet, monkeypatch, tmp_path):
    """The other half of the win: looking up a live worker never waits on a
    spawn in progress for a different env."""
    warm = _FakeWorker()
    pool._WORKER_POOL[str(tmp_path / "warm")] = pool.WorkerRecord(warm, 1)
    release = threading.Event()

    def create(env_dir, *a, **k):
        release.wait(5.0)
        return _FakeWorker()

    monkeypatch.setattr(pool, "_create_worker", create)
    t = threading.Thread(target=lambda: _get(tmp_path / "cold"))
    t.start()
    time.sleep(0.1)                      # the cold spawn is now in progress
    t0 = time.perf_counter()
    worker, gen = _get(tmp_path / "warm")
    assert worker is warm and time.perf_counter() - t0 < 0.5
    release.set()
    t.join(10)


def test_a_failed_spawn_is_raised_to_the_waiter_and_forgotten(clean, quiet, monkeypatch, tmp_path):
    calls = []

    def create(env_dir, *a, **k):
        calls.append(1)
        time.sleep(0.2)
        raise RuntimeError("no such interpreter")

    monkeypatch.setattr(pool, "_create_worker", create)
    errors = []

    def run():
        try:
            _get(tmp_path)
        except RuntimeError as e:
            errors.append(str(e))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
        time.sleep(0.02)
    for t in threads:
        t.join(10)
    assert errors == ["no such interpreter"] * 2
    assert pool._WARMING == {} and str(tmp_path) not in pool._WORKER_POOL
    # and the next caller tries again rather than inheriting the failure
    monkeypatch.setattr(pool, "_create_worker", lambda *a, **k: _FakeWorker())
    assert _get(tmp_path)[0].is_alive()


def test_the_lock_is_never_held_across_the_spawn():
    """Structural guard: no `with _POOL_LOCK:` body in the pool calls
    _create_worker, _spawn_worker or verify_transport."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(pool)))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any(isinstance(i.context_expr, ast.Name) and i.context_expr.id == "_POOL_LOCK"
                   for i in node.items):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                f = inner.func
                name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                if name in ("_create_worker", "_spawn_worker", "verify_transport"):
                    offenders.append((name, inner.lineno))
    assert offenders == [], f"spawn work under _POOL_LOCK: {offenders}"
