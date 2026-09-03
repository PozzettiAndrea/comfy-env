"""Tests for the senior-audit concurrency fixes (dict snapshots, the leaf
mutex, the install lock).

Context the docstrings assume: on CPython 3.11 to 3.13 with the GIL, a
single-statement read-modify-write is incidentally atomic (the eval breaker
checks only at RESUME and back edges), so a timing-based race test cannot
fail there. The guards below are therefore structural (ast) plus
deterministic (mutation-driven fakes) plus brute force (which catches an
unlocked regression on 3.9/3.10 and free-threaded builds where the statement
IS interruptible). Every test names the wrong implementation it catches.
"""

import ast
import threading
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
POOL = SRC / "isolation" / "pool.py"
SUBPROCESS = SRC / "isolation" / "workers" / "subprocess.py"

SHARED_DICTS = {"_WORKER_POOL", "_WORKER_PATCHERS", "_PIN_REPORTS",
                "_OVERHEAD_REPORTS", "_PIN_GRANTS"}


class TestSharedDictIteration:
    def test_shared_dict_iteration_is_always_materialized(self):
        """Bytecode-level iteration over a shared module dict dies with
        "dictionary changed size during iteration" when the aiohttp thread
        registers a patcher mid-comprehension; the callback wrapper swallows
        it into a FAILED budget reply, no host eviction runs, and the load
        OOMs, on exactly the no-NVML hosts the ledger path serves. list()
        and dict() are single C-level calls (GIL-atomic snapshots).
        Module-wide scope so the planned held_from_snapshot refactor stays
        covered without a function-name list."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        offenders = []

        def names_shared(expr):
            return any(isinstance(n, ast.Name) and n.id in SHARED_DICTS
                       for n in ast.walk(expr))

        def is_materialized(expr):
            return (isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Name)
                    and expr.func.id in ("list", "sorted", "dict", "set"))

        iterables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                iterables.append(node.iter)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                                   ast.GeneratorExp)):
                iterables.extend(gen.iter for gen in node.generators)
        for it in iterables:
            if not names_shared(it):
                continue
            # `set(_WORKER_POOL)` style single-call materializations and
            # snapshots taken under _POOL_LOCK still count as materialized;
            # what is banned is a bare .items()/.values()/name iteration.
            if is_materialized(it):
                continue
            # sorted(reports)/list-local iterations do not name the dict.
            offenders.append(it.lineno)
        assert not offenders, (
            f"bare iteration over a shared pool dict at pool.py:{offenders}; "
            f"a concurrent setdefault raises mid-comprehension and the "
            f"budget reply silently fails")

    def test_allocate_pin_budgets_never_receives_the_live_ledger(self):
        """allocate_pin_budgets iterates its reports in pure Python; handing
        it the live _PIN_REPORTS reintroduces the crash one call down."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "allocate_pin_budgets"):
                for arg in node.args:
                    assert not (isinstance(arg, ast.Name)
                                and arg.id == "_PIN_REPORTS"), (
                        f"live _PIN_REPORTS passed to allocate_pin_budgets "
                        f"at pool.py:{node.lineno}; pass dict(_PIN_REPORTS)")

    def test_worker_held_bytes_survives_mid_iteration_registration(self):
        """Deterministic, no thread timing: a patchers value whose __bool__
        inserts a new env key into the outer dict mid-comprehension. The old
        bare iteration dies with RuntimeError on the next step; the
        snapshotted form returns an int."""
        from comfy_env.isolation import pool

        class Registering(dict):
            def __init__(self, outer):
                super().__init__()
                self._outer = outer

            def __bool__(self):
                self._outer.setdefault("late-env", {})
                return False

        outer = {}
        outer["env-a"] = Registering(outer)
        outer["env-b"] = {}
        saved = (pool._WORKER_POOL, pool._WORKER_PATCHERS,
                 pool._OVERHEAD_REPORTS, list(pool._DEVICE_TOTAL_CACHE))
        try:
            pool._WORKER_POOL = {}
            pool._WORKER_PATCHERS = outer
            pool._OVERHEAD_REPORTS = {}
            pool._DEVICE_TOTAL_CACHE.clear()
            pool._DEVICE_TOTAL_CACHE.append(None)
            held = pool._worker_held_bytes()
            assert isinstance(held, int)
        finally:
            (pool._WORKER_POOL, pool._WORKER_PATCHERS,
             pool._OVERHEAD_REPORTS, cache) = saved
            pool._DEVICE_TOTAL_CACHE.clear()
            pool._DEVICE_TOTAL_CACHE.extend(cache)


class TestInFlightCounter:
    def _bare_worker(self):
        from comfy_env.isolation.workers.subprocess import SubprocessWorker
        w = SubprocessWorker.__new__(SubprocessWorker)
        w._calls_in_flight = 0
        w._mem_lock = threading.Lock()
        return w

    def test_bracket_is_exact_under_thread_pressure(self):
        """8 threads x 20000 begin/end pairs must land on exactly 0, and an
        increments-only phase on exactly the sum. Passes incidentally on
        3.11+ even unlocked; catches the unlocked-RMW regression on
        3.9/3.10/free-threaded builds, where a lost decrement charges full
        size forever and a lost increment drops the OOM protection."""
        w = self._bare_worker()
        n, threads = 20000, 8

        def pairs():
            for _ in range(n):
                w.begin_call()
                w.end_call()
        ts = [threading.Thread(target=pairs) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert w._calls_in_flight == 0

        def only_up():
            for _ in range(n):
                w.begin_call()
        ts = [threading.Thread(target=only_up) for _ in range(threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert w._calls_in_flight == threads * n

    def test_counter_is_mutated_only_through_the_helpers(self):
        """Any bare `worker._calls_in_flight = ...` outside subprocess.py is
        the unlocked RMW sneaking back (the audit found the docstring's
        serialization claim was false for exactly that spelling)."""
        for path in SRC.rglob("*.py"):
            if path.name == "subprocess.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AugAssign):
                    targets = [node.target]
                for t in targets:
                    assert not (isinstance(t, ast.Attribute)
                                and t.attr == "_calls_in_flight"), (
                        f"{path.name}:{node.lineno} mutates _calls_in_flight "
                        f"directly; only begin_call/end_call may")


class TestInstallLock:
    def _stub_mm(self):
        import sys
        import types
        mod = types.ModuleType("comfy.model_management")

        def original(*a, **k):
            return "swept"
        mod.unload_all_models = original
        pkg = types.ModuleType("comfy")
        pkg.model_management = mod
        sys.modules["comfy"] = pkg
        sys.modules["comfy.model_management"] = mod
        return mod, original

    def test_double_install_wraps_exactly_once(self):
        """Two installs (sequential here; the lock makes concurrent identical)
        must leave ONE wrapper layer: the loser of the old check-then-set
        race captured the winner's wrapper as _original, nesting forever."""
        import sys
        from comfy_env.isolation import pool
        mod, original = self._stub_mm()
        saved = pool._FREE_WRAP_INSTALLED
        try:
            pool._FREE_WRAP_INSTALLED = False
            pool._install_free_broadcast()
            pool._install_free_broadcast()
            assert getattr(mod.unload_all_models, "_comfy_env_wrap", False)
            assert not getattr(original, "_comfy_env_wrap", False)
        finally:
            pool._FREE_WRAP_INSTALLED = saved
            sys.modules.pop("comfy.model_management", None)
            sys.modules.pop("comfy", None)

    def test_hammered_install_wraps_exactly_once(self):
        import sys
        from comfy_env.isolation import pool
        mod, original = self._stub_mm()
        saved = pool._FREE_WRAP_INSTALLED
        try:
            pool._FREE_WRAP_INSTALLED = False
            barrier = threading.Barrier(16)

            def race():
                barrier.wait()
                pool._install_free_broadcast()
            ts = [threading.Thread(target=race) for _ in range(16)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            wrapper = mod.unload_all_models
            assert getattr(wrapper, "_comfy_env_wrap", False)
            # exactly one layer: the wrapped original must be the sentinel-
            # free original, not another wrapper
            assert not getattr(original, "_comfy_env_wrap", False)
        finally:
            pool._FREE_WRAP_INSTALLED = saved
            sys.modules.pop("comfy.model_management", None)
            sys.modules.pop("comfy", None)

    def test_both_installers_hold_the_install_lock(self):
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        for fname in ("_install_free_broadcast", "_install_prompt_epoch"):
            fn = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == fname)
            src = ast.unparse(fn)
            assert "with _INSTALL_LOCK" in src, (
                f"{fname} lost its install lock; two concurrent first-worker "
                f"creations can nest the wrapper permanently")
            assert src.index("with _INSTALL_LOCK") < src.index("= True"), (
                f"{fname} sets its installed flag outside the lock")
