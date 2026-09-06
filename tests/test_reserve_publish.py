"""Contract: what comfy-env publishes into ComfyUI's reserve is safe.

This is the preventive half of the floor, and every failure mode here is
silent: a compounding base, a reserve that drops before the memory is back,
or a dead worker still holding space. Each test names the wrong
implementation it exists to catch.
"""

import sys
import types

import pytest

GB = 1024 ** 3


@pytest.fixture()
def pool_mod(monkeypatch):
    """isolation.pool with a stubbed comfy.model_management."""
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cuda", index=0)
    mm.get_free_memory = lambda dev: 15 * GB
    mm.EXTRA_RESERVED_VRAM = 400 * 1024 ** 2
    mm.extra_reserved_memory = lambda: mm.EXTRA_RESERVED_VRAM
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    from comfy_env.isolation import pool
    monkeypatch.setattr(pool, "_RESERVE_BASE", None)
    monkeypatch.setattr(pool, "_RESERVE_PUBLISHED", 0)
    monkeypatch.setattr(pool, "_WORKER_HELD", {})
    monkeypatch.setattr(pool, "_WORKER_POOL", {})
    monkeypatch.setattr(pool, "_device_total_bytes", lambda: 24 * GB)
    monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: False)
    pool._WORKER_PATCHERS.clear()
    return pool, mm


class _Worker:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _Patcher:
    def __init__(self, loaded):
        self.model = types.SimpleNamespace(model_loaded_weight_memory=loaded)


def _add_worker(pool, key, loaded=0, alive=True):
    pool._WORKER_POOL[key] = (_Worker(alive), 1)
    if loaded:
        pool._WORKER_PATCHERS[key] = {"m": _Patcher(loaded)}


class TestPublish:
    def test_the_operators_base_is_preserved(self, pool_mod):
        """Catches: overwriting EXTRA_RESERVED_VRAM. That value is the user's
        own --reserve-vram and comfy-env is adding to it, not replacing it."""
        pool, mm = pool_mod
        _add_worker(pool, "a", loaded=0)
        value = pool._publish_reserve()
        assert value >= 400 * 1024 ** 2
        assert mm.EXTRA_RESERVED_VRAM == value

    def test_the_base_is_read_once_and_never_compounds(self, pool_mod):
        """Catches THE compounding bug: re-reading EXTRA_RESERVED_VRAM after
        comfy-env has written it makes every republish add the previous
        reserve back, and the card shrinks to nothing over a session."""
        pool, mm = pool_mod
        _add_worker(pool, "a", loaded=2 * GB)
        first = pool._publish_reserve()
        for _ in range(5):
            pool._publish_reserve()
        assert mm.EXTRA_RESERVED_VRAM == first

    def test_a_device_wide_host_is_told_nothing_at_all(self, pool_mod):
        """The whole rule, at the publish level: on Linux cudaMemGetInfo
        already excludes a worker's models AND its CUDA context, so there is
        nothing to declare and the published number is the operator's own
        --reserve-vram, untouched. Catches any charge creeping back in,
        including the old high water forecast and the context floor, both of
        which the host can already see."""
        pool, mm = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        assert pool._publish_reserve() == 400 * 1024 ** 2
        assert mm.EXTRA_RESERVED_VRAM == 400 * 1024 ** 2

    def test_residency_is_not_double_booked_on_device_wide_platforms(self, pool_mod):
        """Catches the 8.9 GiB bug: on Linux the host already sees resident
        worker VRAM through cudaMemGetInfo, so charging it again removes that
        much usable card for nothing."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        published = pool._publish_reserve()
        assert published < 400 * 1024 ** 2 + 6 * GB

    def test_windows_charges_what_the_worker_holds_now(self, pool_mod, monkeypatch):
        """Catches: applying the device-wide rule on WDDM, where the host
        cannot see the worker at all and would reserve nothing for it."""
        pool, _ = pool_mod
        monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: True)
        _add_worker(pool, "a", loaded=6 * GB)
        assert pool._publish_reserve() >= 6 * GB


class TestShrinkRule:
    def test_a_device_wide_charge_does_not_move_with_residency(self, pool_mod):
        """The old design made the charge RISE as a worker's residency fell,
        to keep protecting space it might want back. That was the forecast,
        and it is gone: the host sees residency directly, and takes memory
        back through the model proxies when it needs it. Catches the forecast
        returning in the form of a residency-sensitive charge."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        busy = pool._publish_reserve()
        pool._WORKER_PATCHERS["a"] = {"m": _Patcher(0)}
        idle = pool._publish_reserve()
        assert idle == busy

    def test_a_lower_proposal_is_refused_without_a_receipt(self, pool_mod):
        """The shrink rule itself: a proposal below what is published needs
        evidence the memory is actually back, because a reserve that drops
        early is space the host loads straight into."""
        pool, _ = pool_mod
        # Only the process-local branch publishes a charge at all now, so
        # that is where a shrink can be observed.
        pool._blind_free_is_process_local = lambda: True
        _add_worker(pool, "a", loaded=2 * GB)
        pool._publish_reserve()
        published = pool._RESERVE_PUBLISHED
        pool._WORKER_POOL.clear()
        pool._WORKER_PATCHERS.clear()
        assert pool._publish_reserve(shrink_allowed=False) == published
        assert pool._publish_reserve(shrink_allowed=True) < published

    def test_removal_is_what_returns_the_space(self, pool_mod):
        """The one legitimate shrink: the process is gone, so the memory is
        provably back. Catches a _forget_reserve that does not actually drop
        the high-water."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        pool._publish_reserve()
        pool._WORKER_POOL.pop("a")
        pool._WORKER_PATCHERS.pop("a", None)
        pool._forget_reserve("a")
        assert pool._publish_reserve(shrink_allowed=True) == 400 * 1024 ** 2
        assert "a" not in pool._WORKER_HELD


def test_publish_never_raises_without_comfy(monkeypatch):
    """Catches: letting an import error out of a boundary hook and taking a
    whole prompt down over a reporting concern."""
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    from comfy_env.isolation import pool
    monkeypatch.setattr(pool, "_RESERVE_BASE", None)
    assert pool._publish_reserve() == 0


class _HeldWorker:
    def __init__(self, held):
        self._last_vram_report = {"held": held}
        self._calls_in_flight = 0

    def is_alive(self):
        return True


@pytest.fixture()
def aimdo_stub(monkeypatch):
    """A host running the pager, with a recording headroom setter."""
    import sys as _sys
    import types as _types
    cmm = _types.ModuleType("comfy.memory_management")
    cmm.aimdo_enabled = True
    monkeypatch.setitem(_sys.modules, "comfy.memory_management", cmm)
    control = _types.ModuleType("comfy_aimdo.control")
    control.calls = []
    control.set_simple_vram_headroom = lambda b: control.calls.append(int(b))
    pkg = _types.ModuleType("comfy_aimdo")
    pkg.control = control
    monkeypatch.setitem(_sys.modules, "comfy_aimdo", pkg)
    monkeypatch.setitem(_sys.modules, "comfy_aimdo.control", control)
    cli = _types.ModuleType("comfy.cli_args")
    cli.args = types.SimpleNamespace(reserve_vram=None)
    monkeypatch.setitem(_sys.modules, "comfy.cli_args", cli)
    return cmm, control, cli


class TestPagerForward:
    """The paged path: EXTRA_RESERVED_VRAM is inert there (measured), the
    pager's own headroom is live at the next fault (measured), so the
    reserve has to be mirrored into it or the default NVIDIA host never
    backs off for a worker."""

    def test_publish_forwards_seed_plus_the_added_reserve(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: forwarding nothing (the P2 era), or forwarding the whole
        published number (base counted twice on the pager)."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub
        monkeypatch.setattr(pool, "_AIMDO_SEED", None)
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(2 * GB), 1)})
        published = pool._publish_reserve()
        from comfy_env.reserve import AIMDO_DEFAULT_HEADROOM
        assert published == mm.EXTRA_RESERVED_VRAM
        assert control.calls == [AIMDO_DEFAULT_HEADROOM + (published - 400 * 1024 ** 2)]

    def test_the_seed_is_the_operators_reserve_vram_when_set(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: seeding from EXTRA_RESERVED_VRAM, which ComfyUI does NOT
        pass to the pager; main.py seeds from --reserve-vram alone."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub
        cli.args.reserve_vram = 2.0
        monkeypatch.setattr(pool, "_AIMDO_SEED", None)
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(1 * GB), 1)})
        published = pool._publish_reserve()
        assert control.calls == [2 * GB + (published - 400 * 1024 ** 2)]

    def test_no_forward_when_the_host_is_not_paging(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: touching the pager on a legacy host, where the published
        reserve already does the job and aimdo may not even be initialised."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub
        cmm.aimdo_enabled = False
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(2 * GB), 1)})
        pool._publish_reserve()
        assert control.calls == []

    def test_falls_back_to_the_raw_export_before_aimdo_107(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: requiring the Python wrapper (comfy-aimdo #107). Every
        wheel since 0.4.10 carries the C export; the pinned 0.5.2 has no
        wrapper."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub
        del control.set_simple_vram_headroom
        control.lib = types.SimpleNamespace(
            set_simple_vram_headroom=lambda b: control.calls.append(("lib", int(b))))
        monkeypatch.setattr(pool, "_AIMDO_SEED", None)
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(2 * GB), 1)})
        pool._publish_reserve()
        assert len(control.calls) == 1 and control.calls[0][0] == "lib"

    def test_a_failing_setter_never_loses_the_published_reserve(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: letting the pager write take the ComfyUI write down with
        it. The legacy knob is written first and stays written."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub

        def boom(b):
            raise RuntimeError("comfy-aimdo is not initialized")
        control.set_simple_vram_headroom = boom
        monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: True)
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(2 * GB), 1)})
        published = pool._publish_reserve()
        assert published > 400 * 1024 ** 2
        assert mm.EXTRA_RESERVED_VRAM == published

    def test_forward_fires_only_when_the_reserve_moves(self, pool_mod, aimdo_stub, monkeypatch):
        """Catches: a pager write on every node boundary. Same rule as the
        ComfyUI write: only on change."""
        pool, mm = pool_mod
        cmm, control, cli = aimdo_stub
        monkeypatch.setattr(pool, "_WORKER_POOL", {"w": (_HeldWorker(2 * GB), 1)})
        pool._publish_reserve()
        pool._publish_reserve()
        assert len(control.calls) == 1


class TestIdleSweepTimer:
    """The sweep used to run only at worker call boundaries, so a host-only
    prompt after a worker prompt never released the worker."""

    def test_starts_exactly_one_daemon_thread(self, pool_mod, monkeypatch):
        """Catches: a non daemon thread (keeps ComfyUI alive at exit) and a
        thread per worker."""
        pool, mm = pool_mod
        started = []

        class _T:
            def __init__(self, target=None, name=None, daemon=None):
                started.append((target, daemon))

            def start(self):
                pass
        monkeypatch.setattr(pool, "_IDLE_SWEEP_STARTED", False)
        monkeypatch.setattr(pool.threading, "Thread", _T)
        pool._start_idle_sweep()
        pool._start_idle_sweep()
        assert started == [(pool._idle_sweep_loop, True)]

    def test_the_loop_body_is_the_sweep(self):
        """Catches: the timer calling something other than the boundary sweep,
        so the two paths drift."""
        import ast
        import inspect
        from comfy_env.isolation import pool
        tree = ast.parse(inspect.getsource(pool._idle_sweep_loop))
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_release_idle_workers" in called


class TestPressureHook:
    """The stand-in's eviction path is the pressure signal. There is no
    second entry in ComfyUI's list any more, and there must not be one."""

    def test_nothing_of_ours_is_planted_in_the_ledger(self, pool_mod):
        """Catches the design that was deleted: a listener appended to
        current_loaded_models to hear the Free button. The button already
        reaches workers through the stand-in's own detach, and a bare entry
        of ours in upstream's list is the surface both loud breaks came
        through. Installing the hook must touch nothing but our own module."""
        pool, mm = pool_mod
        mm.current_loaded_models = []
        pool._install_pressure_hook()
        assert mm.current_loaded_models == []

    def test_the_hook_lands_on_the_object_comfyui_already_calls(self, pool_mod):
        """Catches wiring it to anything other than the stand-in: the whole
        point is that no new object is introduced."""
        pool, _mm = pool_mod
        pool._install_pressure_hook()
        from comfy_env.isolation import model_patcher
        assert model_patcher._ON_PRESSURE is pool._on_host_pressure

    def test_it_never_blocks_the_host(self, pool_mod, monkeypatch):
        """Catches THE way this breaks a user's ComfyUI: it runs on the
        host's thread, inside free_memory, inside a node. A synchronous IPC
        round trip to a busy worker there stalls every host load."""
        import ast
        import inspect
        pool, _mm = pool_mod
        body = ast.unparse(ast.parse(
            inspect.getsource(pool._on_host_pressure).lstrip()))
        assert "Thread" in body and "daemon=True" in body
        assert "join" not in body

    def test_one_eviction_pass_is_one_ask(self, pool_mod, monkeypatch):
        """Catches asking once per stand-in: ComfyUI's loop walks every
        candidate in a single pass and each ask is the same news, so an
        undebounced hook fans one shortage into N sweeps of every worker."""
        pool, _mm = pool_mod
        asks = []
        monkeypatch.setattr(pool, "_ask_idle_workers",
                            lambda n, *a, **k: asks.append(n))
        monkeypatch.setattr(pool, "_LAST_PRESSURE_ASK", [0.0])
        for _ in range(5):
            pool._on_host_pressure(1 << 30)
        for t in list(__import__("threading").enumerate()):
            if t.name == "comfy-env-pressure":
                t.join(timeout=5.0)
        assert len(asks) == 1, asks


class TestForeignHeadroomWriter:
    """The pager's headroom is one process wide global with no owner.

    ComfyUI seeds it once and, today, never again. The obvious upstream fix
    for that is to start writing it per load, at which point there are two
    writers and the loser is silent. These pin the detector.
    """

    def _control(self, current, with_getter=True):
        import types
        c = types.SimpleNamespace()
        if with_getter:
            c.get_simple_vram_headroom = lambda: current
        return c

    def test_a_value_we_did_not_write_is_reported(self, pool_mod, monkeypatch):
        """Catches shipping the setter with no read back at all, which is what
        shipped until comfy-aimdo #107 gave us a getter: two writers on one
        global, last write wins, and nothing says which one lost."""
        pool, _mm = pool_mod
        said = []
        monkeypatch.setattr(pool, "_log", lambda m: said.append(m))
        monkeypatch.setattr(pool, "_AIMDO_HEADROOM_WRITTEN", 4 << 30)
        monkeypatch.setattr(pool, "_AIMDO_FOREIGN_WRITER_LOGGED", False)
        pool._warn_if_headroom_was_changed(self._control(9 << 30), 5 << 30)
        assert any("Something else writes" in m for m in said), said

    def test_our_own_value_is_silent(self, pool_mod, monkeypatch):
        """Catches a detector that fires on every forward: we write this knob
        on every publish, so reading back our own number is the normal case."""
        pool, _mm = pool_mod
        said = []
        monkeypatch.setattr(pool, "_log", lambda m: said.append(m))
        monkeypatch.setattr(pool, "_AIMDO_HEADROOM_WRITTEN", 4 << 30)
        monkeypatch.setattr(pool, "_AIMDO_FOREIGN_WRITER_LOGGED", False)
        pool._warn_if_headroom_was_changed(self._control(4 << 30), 4 << 30)
        assert said == []

    def test_no_getter_costs_nothing(self, pool_mod, monkeypatch):
        """Catches making the read back a precondition. The getter is #107 and
        the installed base is 0.4.13, so its absence must not block or warn."""
        pool, _mm = pool_mod
        said = []
        monkeypatch.setattr(pool, "_log", lambda m: said.append(m))
        monkeypatch.setattr(pool, "_AIMDO_HEADROOM_WRITTEN", 4 << 30)
        monkeypatch.setattr(pool, "_AIMDO_FOREIGN_WRITER_LOGGED", False)
        pool._warn_if_headroom_was_changed(
            self._control(9 << 30, with_getter=False), 5 << 30)
        assert said == []

    def test_it_says_so_once_not_every_publish(self, pool_mod, monkeypatch):
        """Catches a warning inside the publish path with no latch: publishing
        runs at every node boundary, so this would be one line per node."""
        pool, _mm = pool_mod
        said = []
        monkeypatch.setattr(pool, "_log", lambda m: said.append(m))
        monkeypatch.setattr(pool, "_AIMDO_HEADROOM_WRITTEN", 4 << 30)
        monkeypatch.setattr(pool, "_AIMDO_FOREIGN_WRITER_LOGGED", False)
        for _ in range(5):
            pool._warn_if_headroom_was_changed(self._control(9 << 30), 5 << 30)
        assert len(said) == 1, said
