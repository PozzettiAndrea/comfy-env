"""Contract: admission survives a blind `get_free_memory`.

On Windows/WDDM `torch.cuda.mem_get_info` reports the CALLING PROCESS's
budget, not device-wide free -- measured: a sibling allocated 13.0 GiB while
the parent's view moved 75 MB. ComfyUI decides how much to evict from
`memory_required - get_free_memory(device)`, so with that number stuck near
full-card the difference goes negative and `free_memory` evicts NOTHING.

The fix pre-compensates: add the parent's over-report (= worker-held bytes) to
the target passed to `free_memory`. The offset is constant across the eviction
loop, and parent-side unloads move the blind and true numbers together, so
ComfyUI's own arithmetic behaves as if it could see the whole device.

These tests pin that arithmetic and the zero-dependency fallback.
"""

import sys
import types

import pytest

from comfy_env import state_sync

GB = 1024 ** 3


@pytest.fixture()
def pool_mod(monkeypatch):
    """Import isolation.pool with a stubbed comfy.model_management."""
    calls = {"free_memory": []}

    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: types.SimpleNamespace(type="cuda", index=0)
    mm._blind_free = 15 * GB          # what a blind mem_get_info reports
    mm.get_free_memory = lambda dev: mm._blind_free
    mm.minimum_inference_memory = lambda: 1 * GB
    mm.free_memory = lambda amount, dev, **kw: calls["free_memory"].append(amount)
    mm.vram_state = types.SimpleNamespace(name="NORMAL_VRAM", value=3)
    mm.VRAMState = types.SimpleNamespace(LOW_VRAM=types.SimpleNamespace(value=1))
    mm.EXTRA_RESERVED_VRAM = 0
    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    from comfy_env.isolation import pool
    pool._WORKER_PATCHERS.clear()
    # The ledger fallback is a WDDM instrument: these scenarios model a
    # process-local blind number (mm._blind_free counts only this process).
    monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: True)
    return pool, mm, calls


class FakePatcher:
    def __init__(self, resident):
        self._resident = resident

    def loaded_size(self):
        return self._resident


def test_offset_compensates_the_blind_view(pool_mod, monkeypatch):
    """Parent believes 15GB free; a worker really holds 13GB. The target passed
    to free_memory must be inflated by that 13GB, or nothing is ever evicted."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 2 * GB)

    pool._handle_vram_budget({"total_size": 4 * GB})

    assert calls["free_memory"], "free_memory must be called"
    asked = calls["free_memory"][0]
    offset = 15 * GB - 2 * GB
    need = int(4 * GB * state_sync.WEIGHT_SLACK) + pool._WORKER_FIXED_VRAM_COST + 1 * GB
    assert asked == need + offset
    # Without compensation ComfyUI computes need - 15GB < 0 and evicts nothing.
    assert asked > mm._blind_free, (
        "target must exceed the blind free value, or free_memory's "
        "`memory_to_free > 0` guard never passes")


def test_ledger_fallback_when_nvml_unavailable(pool_mod, monkeypatch):
    """No pynvml and no nvidia-smi: reconstruct the offset from comfy-env's own
    books, which already know every worker model's residency."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: None)
    pool._WORKER_PATCHERS["envA"] = {"m1": FakePatcher(6 * GB),
                                     "m2": FakePatcher(2 * GB)}

    pool._handle_vram_budget({"total_size": 1 * GB})

    held = 8 * GB + pool._WORKER_FIXED_VRAM_COST      # one live worker
    need = int(1 * GB * state_sync.WEIGHT_SLACK) + pool._WORKER_FIXED_VRAM_COST + 1 * GB
    assert calls["free_memory"][0] == need + held


def test_headroom_is_additive_not_only_multiplicative(pool_mod, monkeypatch):
    """A per-process CUDA context is a constant, not a percentage: 1.1x on a
    small model does not cover ~250MB of context+handles."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 15 * GB)

    pool._handle_vram_budget({"total_size": 100 * 1024 ** 2})  # 100MB model

    asked = calls["free_memory"][0]
    assert asked >= 100 * 1024 ** 2 + pool._WORKER_FIXED_VRAM_COST
    assert asked > 100 * 1024 ** 2 * 1.1, "multiplicative-only headroom is too small"


def test_inference_reserve_is_included(pool_mod, monkeypatch):
    """An in-process load reserves minimum_inference_memory; a worker load must
    reserve it too, or worker models get ~1GB less headroom than host ones."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 15 * GB)
    pool._handle_vram_budget({"total_size": 2 * GB})
    assert calls["free_memory"][0] >= 2 * GB + 1 * GB


def test_worker_receives_true_device_free(pool_mod, monkeypatch):
    """The worker corrects its OWN blindness from this number: its
    get_free_memory minus device_free = what everyone else holds."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: 3 * GB)
    reply = pool._handle_vram_budget({"total_size": 1 * GB})
    assert reply["device_free_bytes"] == 3 * GB


def test_worker_held_bytes_counts_fixed_cost_per_worker(pool_mod):
    pool, mm, calls = pool_mod
    pool._WORKER_PATCHERS["a"] = {"m": FakePatcher(1 * GB)}
    pool._WORKER_PATCHERS["b"] = {"m": FakePatcher(2 * GB)}
    assert pool._worker_held_bytes() == 3 * GB + 2 * pool._WORKER_FIXED_VRAM_COST


def test_budget_branches_agree_for_identical_device_state(pool_mod, monkeypatch):
    """NVML and the ledger fallback are two coordinate systems for ONE device
    state: the offset translates blind free into device-wide free, and `need`
    re-books the requester's floor plus excess ONCE on top, in BOTH branches.
    Catches: any one-sided edit that treats the ledger offset as a second
    booking (subtracting the requester's booking out of held, an exclude_key
    on _worker_held_bytes, or stripping requester terms from need on one
    branch only). Each makes the two branches evict different amounts and
    report different device_free_bytes for the same card."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})

    def setup():
        pool._WORKER_PATCHERS.clear()
        pool._OVERHEAD_REPORTS.clear()
        pool._WORKER_PATCHERS["req"] = {"m": FakePatcher(2 * GB)}
        pool._OVERHEAD_REPORTS["req"] = {"excess": 700 * 1024 ** 2, "seq": 1}
        pool._WORKER_PATCHERS["other"] = {"m": FakePatcher(5 * GB)}

    setup()
    held = pool._worker_held_bytes()
    assert held > 7 * GB, "guard: books must be non-trivial or 0 == 0 passes"

    # NVML branch: an accurate driver sees exactly what the books record.
    monkeypatch.setattr(pool, "_true_device_free",
                        lambda dev: mm._blind_free - held)
    calls["free_memory"].clear()
    reply = pool._handle_vram_budget({"total_size": 8 * GB}, worker_key="req")
    assert calls["free_memory"], "guard: free_memory must be called"
    target_nvml, free_nvml = calls["free_memory"][0], reply["device_free_bytes"]

    # Ledger branch: same device state, NVML ladder exhausted.
    setup()
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: None)
    calls["free_memory"].clear()
    reply = pool._handle_vram_budget({"total_size": 8 * GB}, worker_key="req")
    assert calls["free_memory"], "guard: free_memory must be called"
    target_ledger = calls["free_memory"][0]
    free_ledger = reply["device_free_bytes"]

    assert target_nvml == target_ledger, (
        "branches must demand the same eviction for the same device state")
    assert free_nvml == free_ledger, (
        "branches must report the same device_free_bytes to the worker")
    assert target_nvml > mm._blind_free, (
        "guard: the scenario must actually demand eviction past the blind "
        "view, or the equality never exercises the offset")


def test_device_wide_blind_free_is_never_ledger_corrected(pool_mod, monkeypatch):
    """Linux/macOS with the NVML ladder exhausted: mem_get_info is already
    device-wide there, so the ledger subtraction double-books every worker
    byte. Catches: the B1 live verdict (free_memory asked for 16 GiB on a
    card with 10 GiB free; 8 GiB of executing worker models evicted for a
    4 GiB load). The two platforms must differ by exactly the ledger."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_WORKER_POOL", {})  # no leftover floors from other tests
    pool._WORKER_PATCHERS.clear()
    pool._WORKER_PATCHERS["other"] = {"m": FakePatcher(5 * GB)}
    held = pool._worker_held_bytes()
    assert 5 * GB < held < mm._blind_free, "guard: the offset must not clamp at zero"
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: None)
    req = {"total_size": 8 * GB}

    monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: True)
    calls["free_memory"].clear()
    wddm = pool._handle_vram_budget(req, worker_key="req")
    target_wddm = calls["free_memory"][0]

    monkeypatch.setattr(pool, "_blind_free_is_process_local", lambda: False)
    calls["free_memory"].clear()
    linux = pool._handle_vram_budget(req, worker_key="req")
    target_linux = calls["free_memory"][0]

    assert target_wddm - target_linux == held, (
        "device-wide platforms must not carry the ledger offset")
    assert linux["device_free_bytes"] == mm._blind_free
    assert wddm["device_free_bytes"] == max(0, mm._blind_free - held)


def test_platform_verdict_is_pure_and_windows_only():
    from comfy_env.state_sync import blind_free_is_process_local
    assert blind_free_is_process_local("win32") is True
    for plat in ("linux", "linux2", "darwin", "", None):
        assert blind_free_is_process_local(plat) is False


def test_ask_includes_the_hosts_own_reserve(pool_mod, monkeypatch):
    """Catches: dropping extra_reserved_memory() from the ask, which is the
    term upstream books and comfy-env did not. Without it the host frees
    strictly less for a worker load than it would for an identical
    in-process one, by exactly the operator's own --reserve-vram."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_WORKER_POOL", {})
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: mm._blind_free)
    pool._WORKER_PATCHERS.clear()

    mm.extra_reserved_memory = lambda: 0
    calls["free_memory"].clear()
    pool._handle_vram_budget({"total_size": 4 * GB}, worker_key="req")
    without = calls["free_memory"][0]

    mm.extra_reserved_memory = lambda: 4 * GB
    calls["free_memory"].clear()
    pool._handle_vram_budget({"total_size": 4 * GB}, worker_key="req")
    with_reserve = calls["free_memory"][0]

    # min_inference is 1 GB in the fixture, so a 4 GB reserve moves the max()
    # from the inference floor to the reserve term: +3 GB.
    assert with_reserve - without == 3 * GB


def test_ask_uses_upstreams_multiplier_on_the_weights(pool_mod, monkeypatch):
    """Catches: a slack below upstream's reappearing anywhere in the chain.
    Asserted on the delta between two model sizes, so it holds whatever the
    additive terms are."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_WORKER_POOL", {})
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: mm._blind_free)
    pool._WORKER_PATCHERS.clear()
    mm.extra_reserved_memory = lambda: 0

    asks = []
    for size in (4 * GB, 8 * GB):
        calls["free_memory"].clear()
        pool._handle_vram_budget({"total_size": size}, worker_key="req")
        asks.append(calls["free_memory"][0])
    assert asks[1] - asks[0] == int(8 * GB * 1.1) - int(4 * GB * 1.1)


def test_ask_excludes_the_requesters_own_reserve_charge(pool_mod, monkeypatch):
    """Catches: passing extra_reserved_memory() straight into the ask. The
    published reserve already holds the requester's own growth, and this
    load IS that growth, so the host would evict its models for the same
    bytes twice. A stranger's charge stays in."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: mm._blind_free)
    monkeypatch.setattr(pool, "_RESERVE_HIGHWATER", {})
    pool._WORKER_PATCHERS.clear()

    class _W:
        _last_vram_report = {"held": 2 * GB}
        _calls_in_flight = 0

        def is_alive(self):
            return True
    monkeypatch.setattr(pool, "_WORKER_POOL", {"req": (_W(), 1)})
    mm.extra_reserved_memory = lambda: 8 * GB

    calls["free_memory"].clear()
    pool._handle_vram_budget({"total_size": 4 * GB}, worker_key="stranger")
    stranger = calls["free_memory"][0]
    calls["free_memory"].clear()
    pool._handle_vram_budget({"total_size": 4 * GB}, worker_key="req")
    own = calls["free_memory"][0]

    # Fixture is process local (WDDM branch), so the charge is the whole
    # entitlement: context floor plus the 2 GB high water.
    expected_charge = pool._WORKER_FIXED_VRAM_COST + 2 * GB
    assert stranger - own == expected_charge


class _AskWorker:
    def __init__(self, freed, in_flight=0, advertises=True):
        self.supports_partial_release = advertises
        self._calls_in_flight = in_flight
        self._freed = freed
        self.sent = []

    def is_alive(self):
        return True

    def send_command_no_spawn(self, method, **kw):
        self.sent.append((method, kw.get("size")))
        return {"receipt": {"freed_bytes": self._freed}}


def _ask_fixture(pool, monkeypatch, workers, free_after_evict):
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_WORKER_POOL", {k: (w, 1) for k, w in workers.items()})
    monkeypatch.setattr(pool, "_RESERVE_HIGHWATER", {k: 8 * GB for k in workers})
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: free_after_evict)
    monkeypatch.setattr(pool, "_publish_reserve", lambda **kw: 0)
    pool._WORKER_PATCHERS.clear()


def test_admission_asks_idle_siblings_when_the_host_is_still_short(pool_mod, monkeypatch):
    """Catches: giving up after the host has evicted its own models. Nothing
    outside a process can free that process's VRAM, so the parent asks the
    processes it owns before the requester loads into a card with no room."""
    pool, mm, calls = pool_mod
    idle = _AskWorker(freed=4 * GB)
    _ask_fixture(pool, monkeypatch, {"idle": idle}, free_after_evict=2 * GB)

    pool._handle_vram_budget({"total_size": 6 * GB}, worker_key="req")

    assert idle.sent and idle.sent[0][0] == "partial_release"
    assert idle.sent[0][1] > 0


def test_admission_asks_nobody_when_the_card_already_has_room(pool_mod, monkeypatch):
    """Catches: an ask on every load. Round trips and page traffic for a
    shortfall that does not exist."""
    pool, mm, calls = pool_mod
    idle = _AskWorker(freed=4 * GB)
    _ask_fixture(pool, monkeypatch, {"idle": idle}, free_after_evict=50 * GB)

    pool._handle_vram_budget({"total_size": 6 * GB}, worker_key="req")

    assert idle.sent == []


def test_admission_never_asks_the_requester(pool_mod, monkeypatch):
    """Catches: the requester shrinking itself to make room for its own
    load, which is a round trip that cancels the admission."""
    pool, mm, calls = pool_mod
    req = _AskWorker(freed=4 * GB)
    _ask_fixture(pool, monkeypatch, {"req": req}, free_after_evict=1 * GB)

    pool._handle_vram_budget({"total_size": 6 * GB}, worker_key="req")

    assert req.sent == []


def test_a_freed_receipt_lowers_that_workers_high_water(pool_mod, monkeypatch):
    """Catches: a high-water that survives a proven release, which keeps the
    reserve holding space the worker has demonstrably given back."""
    pool, mm, calls = pool_mod
    idle = _AskWorker(freed=3 * GB)
    _ask_fixture(pool, monkeypatch, {"idle": idle}, free_after_evict=1 * GB)

    pool._ask_idle_workers(4 * GB, requester_key="req")

    assert pool._RESERVE_HIGHWATER["idle"] == 5 * GB


def test_the_reply_discounts_the_requesters_own_charge(pool_mod, monkeypatch):
    """The worker assigns this number to its OWN EXTRA_RESERVED_VRAM, where
    it shrinks the weight budget of the very load it is about to do. Catches
    handing over the raw published total, which made BOTH ends under load:
    the host freed for a reserve that already counted the requester, and the
    requester then reserved against itself."""
    pool, mm, calls = pool_mod
    monkeypatch.setattr(pool, "_OVERHEAD_REPORTS", {})
    monkeypatch.setattr(pool, "_true_device_free", lambda dev: mm._blind_free)
    monkeypatch.setattr(pool, "_RESERVE_HIGHWATER", {})
    pool._WORKER_PATCHERS.clear()

    class _W:
        _last_vram_report = {"held": 2 * GB}
        _calls_in_flight = 0

        def is_alive(self):
            return True
    monkeypatch.setattr(pool, "_WORKER_POOL", {"req": (_W(), 1)})
    mm.EXTRA_RESERVED_VRAM = 8 * GB
    mm.extra_reserved_memory = lambda: mm.EXTRA_RESERVED_VRAM

    own = pool._handle_vram_budget({"total_size": 1 * GB}, worker_key="req")
    stranger = pool._handle_vram_budget({"total_size": 1 * GB}, worker_key="other")

    charge = pool._WORKER_FIXED_VRAM_COST + 2 * GB
    assert stranger["extra_reserved_vram"] == 8 * GB
    assert own["extra_reserved_vram"] == 8 * GB - charge
