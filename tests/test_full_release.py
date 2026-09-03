"""Tests for the /free deep release (host wrap, broadcast, worker ladder).

The pure units are memory_manager.full_release (injected modules) and
state_sync.plan_release_broadcast; the ast guards pin the wiring bare CI
cannot execute. Every test names the wrong implementation it catches.
"""

import ast
from pathlib import Path

from comfy_env.memory_manager import full_release
from comfy_env.state_sync import plan_release_broadcast

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
POOL = SRC / "isolation" / "pool.py"
WORKER = SRC / "isolation" / "workers" / "_persistent_worker.py"
SUBPROCESS = SRC / "isolation" / "workers" / "subprocess.py"
MEMMGR = SRC / "memory_manager.py"


def _body_src(fn):
    """Unparsed function body EXCLUDING the docstring: these guards ban
    calls/references in code, and the docstrings legitimately name the very
    things they forbid (to say why)."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return "\n".join(ast.unparse(st) for st in body)


def _fakes(order):
    class FakeCuda:
        def is_initialized(self):
            return True

        def memory_reserved(self):
            return 8 * 1024 ** 3

        def synchronize(self):
            order.append("synchronize")

        def empty_cache(self):
            order.append("empty_cache")

        def ipc_collect(self):
            order.append("ipc_collect")

    class FakeC:
        def _host_emptyCache(self):
            order.append("host_empty_cache")

    class FakeTorch:
        cuda = FakeCuda()
        _C = FakeC()

    class FakeMM:
        TOTAL_PINNED_MEMORY = 2 * 1024 ** 3

        def reset_cast_buffers(self):
            order.append("reset_cast_buffers")

    return {"torch": FakeTorch(), "comfy.model_management": FakeMM()}


class TestFullReleaseLadder:
    def test_ladder_order_gc_then_caches_then_host_cache(self):
        """The dependency chain: gc returns cycle-held blocks to the device
        allocator and pinned buffers to torch's host cache, so empty_cache
        must follow gc and host_emptyCache must come LAST or it misses the
        buffers gc just returned. Catches: a host flush before gc (strands
        gc-returned buffers) and empty_cache before gc (blocks still
        'allocated' when the flush runs)."""
        order = []
        receipt = full_release(_modules=_fakes(order))
        gc_pos = next(i for i, s in enumerate(receipt["steps"])
                      if s["name"] == "gc_collect")
        names = [s["name"] for s in receipt["steps"]]
        assert names.index("empty_cache") > gc_pos
        assert names.index("host_empty_cache") == len(names) - 1
        assert order.index("host_empty_cache") > order.index("empty_cache")
        assert receipt["errors"] == []

    def test_receipt_reports_measured_numbers_not_intent(self):
        """The parent verifies with the receipt; a fabricated zero would mask
        a failed release."""
        receipt = full_release(_modules=_fakes([]))
        assert receipt["reserved_before"] == 8 * 1024 ** 3
        assert receipt["pinned_before"] == 2 * 1024 ** 3

    def test_torchless_worker_yields_receipt_not_exception(self):
        """A CPU pack's worker has no torch; pressing /free must not kill it.
        Catches: a direct import that turns the ladder into an ImportError."""
        receipt = full_release(_modules={})
        assert isinstance(receipt, dict)
        gc_steps = [s for s in receipt["steps"] if s["name"] == "gc_collect"]
        assert gc_steps and gc_steps[0]["ok"]

    def test_raising_rung_does_not_stop_later_rungs(self):
        """One try around the whole ladder would let a failed cast-buffer
        reset skip the cache flush entirely."""
        order = []
        modules = _fakes(order)

        def boom():
            raise RuntimeError("cast reset broke")
        modules["comfy.model_management"].reset_cast_buffers = boom
        receipt = full_release(_modules=modules)
        assert any("cast reset broke" in e for e in receipt["errors"])
        assert "empty_cache" in order and "host_empty_cache" in order

    def test_release_never_touches_keepers_or_ledgers(self):
        """The keepers' lifetime is the consumed-ack protocol; the overflow
        store is node STATE; the pin registration ledger drops through the
        real unload sweep. free_registrations here would desynchronize
        TOTAL_PINNED accounting."""
        tree = ast.parse(MEMMGR.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "full_release")
        src = _body_src(fn)
        for banned in ("free_registrations", "_shm_keeper", "_tensor_keeper",
                       "_overflow_store", "unpin_memory"):
            assert banned not in src, (
                f"full_release touches {banned}; that store has a correct "
                f"owner already")

    def test_full_release_never_raises_by_construction(self):
        tree = ast.parse(MEMMGR.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "full_release")
        assert not [n for n in ast.walk(fn) if isinstance(n, ast.Raise)], (
            "full_release contains a raise; a failed release must degrade, "
            "never kill the worker's request loop")


class TestPlanReleaseBroadcast:
    def test_every_key_lands_in_exactly_one_list(self):
        """Set equality, not lengths: a filtered-out worker cannot hide."""
        plan = plan_release_broadcast(
            {"a": {"alive": True, "advertises": True},
             "b": {"alive": False, "advertises": True},
             "c": {"alive": True, "advertises": False}},
            now=100.0, last_broadcast=0.0)
        assert plan["send"] == ["a"]
        assert plan["skip_dead"] == ["b"]
        assert plan["skip_unsupported"] == ["c"]
        everything = sum(plan.values(), [])
        assert sorted(everything) == ["a", "b", "c"]

    def test_dead_worker_never_lands_in_send(self):
        """The send path must never resurrect a worker to free its memory
        (send_command runs _ensure_started; the no-spawn path exists for
        exactly this)."""
        plan = plan_release_broadcast({"d": {"alive": False, "advertises": True}},
                                      now=100.0, last_broadcast=0.0)
        assert plan["send"] == [] and plan["skip_dead"] == ["d"]

    def test_debounce_suppresses_same_burst_duplicates_only(self):
        """The window is deliberately SHORT (0.5 s): the wrap site cannot
        tell a human /free from OOM recovery, so a long window would swallow
        a genuine press right after an OOM broadcast. Inside the window a
        nested duplicate is skipped; just outside it a fresh press sends."""
        plan = plan_release_broadcast({"a": {"alive": True, "advertises": True}},
                                      now=100.0, last_broadcast=99.7)
        assert plan["send"] == [] and plan["skip_debounced"] == ["a"]
        plan = plan_release_broadcast({"a": {"alive": True, "advertises": True}},
                                      now=100.0, last_broadcast=99.0)
        assert plan["send"] == ["a"]

    def test_empty_pool_is_a_noop(self):
        plan = plan_release_broadcast({}, now=1.0, last_broadcast=0.0)
        assert all(v == [] for v in plan.values())


class TestFreeSeamGuards:
    def test_no_spawn_send_path_contains_no_ensure_started(self):
        """send_command runs _ensure_started, so a broadcast through it would
        SPAWN a worker in order to free its memory."""
        tree = ast.parse(SUBPROCESS.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "send_command_no_spawn")
        src = _body_src(fn)
        assert "_ensure_started" not in src
        assert "is_alive" in src

    def test_wrap_calls_the_original_before_broadcasting(self):
        """The sweep must detach worker models (dropping their pins through
        the real unpatch path) BEFORE the ladder runs, or rung 4 flushes a
        cache the sweep is about to refill."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_wrapped_unload_all_models")
        src = ast.unparse(fn)
        assert src.index("_original(") < src.index("broadcast_release")

    def test_wrap_is_behind_its_kill_switch(self):
        """comfy-env's first host-side function wrap earns a revert path that
        needs no package rollback."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_install_free_broadcast")
        assert "FREE_BROADCAST_ENV_VAR" in ast.unparse(fn)

    def test_release_dispatch_is_main_loop_only(self):
        """A release executed from _call_parent's interleave would run gc and
        empty_cache under an ACTIVE forward (the interleave services commands
        while a node computes)."""
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        call_parent = next(n for n in ast.walk(tree)
                           if isinstance(n, ast.FunctionDef)
                           and n.name == "_call_parent")
        assert "full_release" not in ast.unparse(call_parent), (
            "full_release is dispatched from the _call_parent interleave; "
            "it would release memory mid-forward")
        src = WORKER.read_text(encoding="utf-8")
        assert 'request.get("method") == "full_release"' in src

    def test_broadcast_binds_and_ingests_every_reply(self):
        """Fire-and-forget masks failure, and harvested-but-not-ingested
        replies sit unconsumed until a next call that a released worker may
        never make."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_release_one")
        src = ast.unparse(fn)
        assert "send_command_no_spawn" in src
        assert "_ingest_worker_frames" in src, (
            "the broadcast no longer ingests the reply's census and pin "
            "scalar; a quiet released worker advertises stale pins forever")

    def test_ready_frame_advertises_the_capability(self):
        src = WORKER.read_text(encoding="utf-8")
        ready = src[src.index("_ready_frame = {"):src.index("transport.send(_ready_frame)")]
        assert '"full_release": True' in ready


class TestPinPressureSeam:
    """The RAM-pressure reclaim wiring (gap 6): the one honest trigger is
    should_free_pins_for_ram_pressure (single upstream caller, fires only
    under genuine pressure); the broadcast must never block the execution
    loop and never respawn the dead."""

    def test_wrap_calls_original_first_and_returns_it_verbatim(self):
        """The wrap is observability plus a side effect, never a behavior
        change: the host's own free_pins must still run on the True path."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_wrapped_should_free_pins")
        src = ast.unparse(fn)
        assert src.index("_original(") < src.index("broadcast_pin_release")
        assert "return result" in src

    def test_pressure_broadcast_never_joins(self):
        """broadcast_release may join (a human pressed /free and waits);
        the pressure sweep fires from the execution loop BETWEEN NODES and a
        join would stall every prompt under sustained pressure."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "broadcast_pin_release")
        src = _body_src(fn)
        assert ".join(" not in src, (
            "broadcast_pin_release joins its threads; the execution loop "
            "stalls under sustained pressure")
        assert "send_command_no_spawn" in src

    def test_pressure_installer_holds_the_install_lock(self):
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_install_pin_pressure")
        assert "with _INSTALL_LOCK" in ast.unparse(fn)

    def test_release_pins_never_touches_the_ladder_exclusions(self):
        """Same OUT list as full_release: keepers, overflow store, and it
        must go through mm.free_pins (the real tier ladder, prompt marks
        included), never a side-channel unpin."""
        memmgr = SRC / "memory_manager.py"
        tree = ast.parse(memmgr.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "release_pins")
        src = _body_src(fn)
        assert "free_pins" in src
        for banned in ("unpin_memory", "_shm_keeper", "_overflow_store",
                       "free_registrations"):
            assert banned not in src

    def test_worker_advertises_and_dispatches_release_pins(self):
        src = WORKER.read_text(encoding="utf-8")
        assert '"release_pins": True' in src
        assert 'request.get("method") == "release_pins"' in src
        call_parent = next(n for n in ast.walk(ast.parse(src))
                           if isinstance(n, ast.FunctionDef)
                           and n.name == "_call_parent")
        assert "release_pins" not in ast.unparse(call_parent), (
            "release_pins dispatched from the interleave; it would free "
            "pins under an active forward")
