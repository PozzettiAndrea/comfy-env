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
    monkeypatch.setattr(pool, "_RESERVE_HIGHWATER", {})
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

    def test_an_idle_worker_still_reserves_its_context(self, pool_mod):
        """Catches: charging nothing for a worker holding no model. Its CUDA
        context is real and the host taking that space OOMs its next call."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=0)
        assert pool._publish_reserve() > 400 * 1024 ** 2

    def test_a_dead_worker_is_not_charged(self, pool_mod):
        """Catches: reserving for a process that no longer exists, which
        shrinks the card permanently after any crash."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=4 * GB, alive=False)
        pool._publish_reserve()
        _add_worker(pool, "b", loaded=0)
        with_live_only = pool._publish_reserve(shrink_allowed=True)
        assert with_live_only < 4 * GB

    def test_residency_is_not_double_booked_on_device_wide_platforms(self, pool_mod):
        """Catches the 8.9 GiB bug: on Linux the host already sees resident
        worker VRAM through cudaMemGetInfo, so charging it again removes that
        much usable card for nothing."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        published = pool._publish_reserve()
        assert published < 400 * 1024 ** 2 + 6 * GB

    def test_windows_charges_the_whole_entitlement(self, pool_mod, monkeypatch):
        """Catches: applying the device-wide rule on WDDM, where the host
        cannot see the worker at all and would reserve nothing for it."""
        pool, _ = pool_mod
        monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: True)
        _add_worker(pool, "a", loaded=6 * GB)
        assert pool._publish_reserve() >= 6 * GB


class TestShrinkRule:
    def test_protection_is_constant_as_residency_moves(self, pool_mod):
        """The property that matters, which is NOT "the number never falls".

        On a device-wide platform the host already sees resident worker VRAM,
        so the charge is the headroom beyond it. As a worker's residency
        drops, the host starts seeing that space as free, and the charge must
        RISE by the same amount to keep the worker's entitlement protected.
        Catches: a charge that ignores residency (the double book) or one
        that falls with it (handing the host space the worker still needs).
        """
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        busy = pool._publish_reserve()
        pool._WORKER_PATCHERS["a"] = {"m": _Patcher(0)}
        idle = pool._publish_reserve()
        # busy: host sees 6 GB held, we add the context floor.
        # idle: host sees nothing held, we add the whole entitlement.
        assert idle - busy == 6 * GB

    def test_a_lower_proposal_is_refused_without_a_receipt(self, pool_mod):
        """The shrink rule itself: a proposal below what is published needs
        evidence the memory is actually back, because a reserve that drops
        early is space the host loads straight into."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=0)
        pool._publish_reserve()
        published = pool._RESERVE_PUBLISHED
        pool._WORKER_POOL.clear()
        pool._WORKER_PATCHERS.clear()
        assert pool._publish_reserve(shrink_allowed=False) == published
        assert pool._publish_reserve(shrink_allowed=True) < published

    def test_high_water_survives_a_drop_to_zero(self, pool_mod):
        """Catches: tracking current residency, which hands the host the
        worker's space between calls."""
        pool, _ = pool_mod
        _add_worker(pool, "a", loaded=6 * GB)
        pool._publish_reserve()
        pool._WORKER_PATCHERS["a"] = {"m": _Patcher(0)}
        pool._publish_reserve()
        assert pool._RESERVE_HIGHWATER["a"] >= 6 * GB

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
        assert "a" not in pool._RESERVE_HIGHWATER


def test_publish_never_raises_without_comfy(monkeypatch):
    """Catches: letting an import error out of a boundary hook and taking a
    whole prompt down over a reporting concern."""
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    from comfy_env.isolation import pool
    monkeypatch.setattr(pool, "_RESERVE_BASE", None)
    assert pool._publish_reserve() == 0
