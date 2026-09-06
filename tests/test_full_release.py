"""Tests for the /free deep release (host wrap, broadcast, worker ladder).

The pure unit is memory_manager.full_release (injected modules); the ast
guards pin the wiring bare CI cannot execute. Every test names the wrong
implementation it catches.
"""

import ast
from pathlib import Path

from comfy_env.memory_manager import full_release

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

    def test_ready_frame_advertises_the_capability(self):
        src = WORKER.read_text(encoding="utf-8")
        ready = src[src.index("_ready_frame = {"):src.index("transport.send(_ready_frame)")]
        assert '"full_release": True' in ready


class TestPinPressureSeam:
    """The RAM-pressure pin release lever. comfy-env does not patch the host,
    so nothing upstream triggers it; it is a lever comfy-env's own code may
    pull, and when pulled it must never block the execution loop and never
    respawn the dead."""


    def test_pressure_broadcast_never_joins(self):
        """The pressure sweep fires from the execution loop BETWEEN NODES;
        a join would stall every prompt under sustained pressure."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "broadcast_pin_release")
        src = _body_src(fn)
        assert ".join(" not in src, (
            "broadcast_pin_release joins its threads; the execution loop "
            "stalls under sustained pressure")
        assert "send_command_no_spawn" in src


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


GIB = 1024 ** 3


class TestFullReleaseUnloadsModels:
    def test_rung_zero_unloads_the_workers_own_models(self):
        """Catches the premise that expired when the proxy went away: this
        ladder was written to run AFTER the host's sweep had already detached
        the worker's models, so without rung 0 the Free button reaches the
        worker and frees only caches. Measured: a 6 GiB paged model survived
        the whole ladder untouched."""
        import types
        from comfy_env import memory_manager as mm_mod
        called = []
        mm = types.SimpleNamespace(
            unload_all_models=lambda: called.append("unload_all_models"),
            reset_cast_buffers=lambda: called.append("reset_cast_buffers"),
            TOTAL_PINNED_MEMORY=0)
        mm_mod.full_release(_modules={"comfy.model_management": mm})
        assert called[0] == "unload_all_models"

    def test_it_runs_before_the_cache_sweep(self):
        """Catches: unloading after empty_cache, which leaves the blocks the
        unload just returned sitting in torch's allocator."""
        import ast
        import inspect
        from comfy_env import memory_manager as mm_mod
        src = ast.unparse(ast.parse(inspect.getsource(mm_mod.full_release)))
        assert src.index("unload_all_models") < src.index("empty_cache")

    def test_an_old_comfy_without_it_is_skipped_not_crashed(self):
        import types
        from comfy_env import memory_manager as mm_mod
        mm = types.SimpleNamespace(TOTAL_PINNED_MEMORY=0)
        receipt = mm_mod.full_release(_modules={"comfy.model_management": mm})
        assert receipt["errors"] == []


class TestPartialReleaseSeam:
    """The admission time shrink. Unlike full_release it keeps the model
    loaded, so the pages come back from pinned RAM rather than from disk;
    that is what makes asking cheaper than evicting."""

    def test_it_asks_for_free_plus_the_shortfall(self):
        """Catches: passing the shortfall straight to free_memory. ComfyUI's
        loop evicts until `memory_required - get_free_memory` is met, so a
        target below the current free number evicts nothing at all."""
        import types
        from comfy_env import memory_manager as mm_mod
        asks = []
        mm = types.SimpleNamespace(
            get_torch_device=lambda: "cuda",
            get_free_memory=lambda dev: 1 * GIB,
            free_memory=lambda amount, dev: asks.append(amount))
        mm_mod.partial_release(4 * GIB, _modules={"comfy.model_management": mm})
        assert asks == [5 * GIB]

    def test_the_receipt_is_measured_not_assumed(self):
        """Catches: reporting the asked size as freed. The parent lowers a
        worker's high-water by this number, so an optimistic receipt hands
        the host space that never came back."""
        import types
        from comfy_env import memory_manager as mm_mod
        free = [1 * GIB]
        mm = types.SimpleNamespace(
            get_torch_device=lambda: "cuda",
            get_free_memory=lambda dev: free[0],
            free_memory=lambda amount, dev: free.__setitem__(0, 3 * GIB))
        receipt = mm_mod.partial_release(9 * GIB,
                                         _modules={"comfy.model_management": mm})
        assert receipt["freed_bytes"] == 2 * GIB

    def test_a_raising_manager_returns_a_receipt_not_an_exception(self):
        """Catches: letting a worker side failure surface as a failed budget
        reply, which the requester reads as a refused admission."""
        import types
        from comfy_env import memory_manager as mm_mod

        def boom(*a, **k):
            raise RuntimeError("no cuda")
        mm = types.SimpleNamespace(get_torch_device=boom,
                                   get_free_memory=boom, free_memory=boom)
        receipt = mm_mod.partial_release(1 * GIB,
                                         _modules={"comfy.model_management": mm})
        assert receipt["errors"] and "freed_bytes" not in receipt

    def test_dispatch_is_main_loop_only(self):
        """Catches: handling partial_release in the _call_parent interleave,
        which would drop pages while a forward is running in this worker."""
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        call_parent = next(n for n in ast.walk(tree)
                           if isinstance(n, ast.FunctionDef)
                           and n.name == "_call_parent")
        assert "partial_release" not in ast.unparse(call_parent)
        src = WORKER.read_text(encoding="utf-8")
        assert 'request.get("method") == "partial_release"' in src

    def test_ready_frame_advertises_the_capability(self):
        """Catches: shipping the handler without the flag, so the parent
        never asks, or the flag without the handler, so it asks a worker
        that answers with an error."""
        src = WORKER.read_text(encoding="utf-8")
        ready = src[src.index("_ready_frame = {"):src.index("transport.send(_ready_frame)")]
        assert '"partial_release": True' in ready
