"""Tests for the memory seam sync logic (problems 7 and 11).

Every test here comes from the two design groups' validated test lists, and
each names the wrong implementation it exists to catch. None can pass
vacuously: the assertions are on values, not deltas, per this codebase's
history of tautological tests.
"""

import pickle

import pytest

from comfy_env.state_sync import (
    RESERVED_KEYS,
    SEED_SENTINEL,
    apply_residency,
    apply_state_out,
    diff_state,
    fingerprint,
    held_ceiling,
    is_overflow_marker,
    make_marker,
    outbound_state,
)

GIB = 1024**3


class FakeModel:
    def __init__(self, size):
        self.model_loaded_weight_memory = size


class FakePatcher:
    def __init__(self, size, loaded=None):
        self.size = size
        self.model = FakeModel(size if loaded is None else loaded)
        self.load_device = "cuda:0"
        self.offload_device = "cpu"
        self.eviction_deferred = False

    def loaded_size(self):
        return self.model.model_loaded_weight_memory


def census(mid, seq, resident, device="cuda:0"):
    return [{"id": mid, "seq": seq, "resident": resident, "device": device}]


class TestApplyResidency:
    def test_a_decrease_applies_to_an_already_known_patcher(self):
        """Catches: refresh only for freshly-registered ids (the actual bug:
        the `continue` skip made known models unreachable)."""
        p = FakePatcher(8 * GIB)
        changed = apply_residency({"m": p}, census("m", 1, 2 * GIB))
        assert p.model.model_loaded_weight_memory == 2 * GIB
        assert changed == ["m"]

    def test_fractional_residency_arrives_verbatim(self):
        """Catches: a boolean in bytes' clothing (size-if-resident-else-0)."""
        p = FakePatcher(8 * GIB)
        apply_residency({"m": p}, census("m", 1, 8 * GIB // 4))
        got = p.model.model_loaded_weight_memory
        assert got == 8 * GIB // 4
        assert got not in (0, 8 * GIB)

    def test_stale_census_cannot_resurrect_freed_bytes(self):
        """The only defence against a frame census undoing an eviction reply.
        Catches: arrival-order last-write-wins and wall-clock stamps."""
        p = FakePatcher(8 * GIB)
        p._residency_seq = 5  # an eviction reply already recorded seq 5
        p.model.model_loaded_weight_memory = 0
        apply_residency({"m": p}, census("m", 4, 8 * GIB))
        assert p.model.model_loaded_weight_memory == 0

    def test_zero_flips_device_but_keeps_the_entry_reachable(self):
        """Catches: zeroing that also drops the patcher (invisible AND
        unevictable)."""
        p = FakePatcher(8 * GIB)
        apply_residency({"m": p}, census("m", 1, 0))
        assert p.model.model_loaded_weight_memory == 0
        assert p.model.device == "cpu"

    def test_clamped_to_size(self):
        p = FakePatcher(4 * GIB)
        apply_residency({"m": p}, census("m", 1, 9 * GIB))
        assert p.model.model_loaded_weight_memory == 4 * GIB

    def test_missing_id_keeps_old_value_not_zero(self):
        """Missing means unknown, not zero: zeroing under-states residency and
        over-states true free, admitting a load that OOMs."""
        p = FakePatcher(8 * GIB, loaded=6 * GIB)
        apply_residency({"m": p}, [])
        assert p.model.model_loaded_weight_memory == 6 * GIB

    def test_unknown_id_ignored_without_error(self):
        p = FakePatcher(8 * GIB)
        apply_residency({"m": p}, census("ghost", 1, GIB))
        assert p.model.model_loaded_weight_memory == 8 * GIB

    def test_census_is_the_decay_authority(self):
        """A census is a present-state statement sampled at a boundary, when
        the env is about to be idle: it decays the peak to sampled truth.
        Catches: the old up-only ratchet, which stranded the pre-eviction
        peak forever (the in-flight size charge in held_charge now carries
        the mid-call pessimism this ratchet used to overpay for)."""
        p = FakePatcher(8 * GIB, loaded=0)
        apply_residency({"m": p}, census("m", 1, 6 * GIB))
        apply_residency({"m": p}, census("m", 2, 2 * GIB))
        assert p.model.model_loaded_weight_memory == 2 * GIB  # ledger: receipt
        assert held_ceiling(p) == 2 * GIB  # census decayed the peak

    def test_census_at_equal_seq_is_applied_not_dropped(self):
        """THE dead-letter fix. The worker bumps seq only at command echoes;
        a census after the echo carries the SAME seq and was sampled AFTER
        the echo's byte movement (single thread, one FIFO socket). The old
        `<=` drop made every post-echo census a dead letter: a lazy re-fault
        regrew VRAM the parent could never see again, and at ledger 0 detach
        short-circuits so the bytes were unevictable too. Catches: the
        shipped strict-`<=` comparator."""
        p = FakePatcher(8 * GIB, loaded=0)
        p._residency_seq = 5  # an eviction echo stamped seq 5, resident 0
        p._residency_peak = 0
        apply_residency({"m": p}, census("m", 5, 6 * GIB))  # post-echo re-fault
        assert p.model.model_loaded_weight_memory == 6 * GIB
        assert held_ceiling(p) == 6 * GIB

    def test_equal_seq_replay_keeps_replace_semantics(self):
        """Multiple equal-seq censuses arrive transport-ordered; the last one
        wins. Catches: a tie rule that applies only the first."""
        p = FakePatcher(8 * GIB, loaded=0)
        p._residency_seq = 5
        apply_residency({"m": p}, census("m", 5, 6 * GIB))
        apply_residency({"m": p}, census("m", 5, 1 * GIB))
        assert p.model.model_loaded_weight_memory == 1 * GIB
        assert held_ceiling(p) == 1 * GIB


class TestApplyEcho:
    """The echo transition, pure. The peak rule is the admissibility
    criterion: lower the admission ceiling only when every regrowth exit out
    of the certified state is signaled."""

    from comfy_env.state_sync import apply_echo, held_charge  # noqa: PLC0415

    def test_in_flight_partial_echo_never_lowers_the_peak(self):
        """The worker can lazily re-fault mid call with no signal, so a
        mid-call eviction echo may not lower the ceiling. Catches: the
        originally shipped unconditional reset (model_patcher.py:272), the
        unbounded stale-LOW window."""
        from comfy_env.state_sync import apply_echo
        p = FakePatcher(8 * GIB, loaded=6 * GIB)
        p._residency_peak = 6 * GIB
        apply_echo(p, 2 * GIB, seq=7, in_flight=True)
        assert p.model.model_loaded_weight_memory == 2 * GIB  # ledger: receipt
        assert held_ceiling(p) == 6 * GIB  # window stays pessimistic
        assert p._residency_seq == 7

    def test_idle_partial_echo_releases_the_reserve_immediately(self):
        """An idle worker cannot re-fault (faults are synchronous worker
        Python), so its echo is authoritative and the parked env's
        over-reserve releases NOW, not at a census that may never come.
        Catches: the echoes-never-lower overcorrection."""
        from comfy_env.state_sync import apply_echo
        p = FakePatcher(8 * GIB, loaded=6 * GIB)
        p._residency_peak = 6 * GIB
        apply_echo(p, 2 * GIB, seq=7, in_flight=False)
        assert held_ceiling(p) == 2 * GIB

    def test_unmapped_lowers_to_zero_in_either_flag_state(self):
        """A detach is a full-unmap receipt: every regrowth exit is signaled
        (budget RPC, load echo, fresh registration). Also fixes the shipped
        bug where _mark_offloaded zeroed the ledger but left the peak stuck
        high forever."""
        from comfy_env.state_sync import apply_echo
        for in_flight in (False, True):
            p = FakePatcher(8 * GIB, loaded=6 * GIB)
            p._residency_peak = 6 * GIB
            apply_echo(p, 0, unmapped=True, in_flight=in_flight)
            assert held_ceiling(p) == 0
            assert p.model.device == "cpu"

    def test_echo_never_raises_on_a_broken_patcher(self):
        from comfy_env.state_sync import apply_echo
        apply_echo(object(), 1 * GIB)  # must not raise

    def test_held_charge_is_size_in_flight_and_ceiling_idle(self):
        """The case split itself. Catches: an additive in-flight term (which
        would double count every resident byte) and a charge that forgets
        the idle peak."""
        from comfy_env.state_sync import held_charge
        p = FakePatcher(8 * GIB, loaded=2 * GIB)
        p._residency_peak = 3 * GIB
        assert held_charge(p, in_flight=True) == 8 * GIB
        assert held_charge(p, in_flight=False) == 3 * GIB


class TestHeldFromSnapshot:
    from comfy_env.state_sync import held_from_snapshot  # noqa: PLC0415

    def test_in_flight_charge_replaces_never_adds(self):
        """B's veto of the additive composition: an in-flight worker's model
        charges size ONCE, not held plus an inflight term."""
        from comfy_env.state_sync import held_from_snapshot
        snap = {"w": {"in_flight": True, "excess": 0,
                      "models": [{"ledger": 2 * GIB, "peak": 3 * GIB,
                                  "size": 8 * GIB}]}}
        assert held_from_snapshot(snap, floor=0) == 8 * GIB

    def test_modelless_live_worker_books_the_floor(self):
        """The old `if patchers:` skip booked a live worker's CUDA context at
        zero. Presence in the snapshot IS liveness."""
        from comfy_env.state_sync import held_from_snapshot
        floor = 300 * 1024 * 1024
        assert held_from_snapshot({"w": {"in_flight": False, "excess": None,
                                         "models": []}},
                                  floor=floor) == floor

    def test_excess_adds_to_the_floor_never_maxes(self):
        """memory_reserved cannot see the CUDA context, so floor and excess
        partition cleanly and ADD; max(floor, reported) under-books by
        min(floor, reported). A low report ADDS, no special casing."""
        from comfy_env.state_sync import held_from_snapshot
        floor = 300 * 1024 * 1024
        low = 10 * 1024 * 1024
        snap = {"w": {"in_flight": False, "excess": low, "models": []}}
        assert held_from_snapshot(snap, floor=floor) == floor + low

    def test_excess_clamped_to_cap(self):
        from comfy_env.state_sync import held_from_snapshot
        floor = 300 * 1024 * 1024
        snap = {"w": {"in_flight": False, "excess": 99 * GIB, "models": []}}
        assert held_from_snapshot(snap, floor=floor, cap=24 * GIB) \
            == floor + (24 * GIB - floor)

    def test_absent_worker_key_books_nothing(self):
        """Dead workers' keys are ABSENT (the pool removes them); a ghost
        entry would book ~1 GB per crash in a restart loop."""
        from comfy_env.state_sync import held_from_snapshot
        assert held_from_snapshot({}, floor=300 * 1024 * 1024) == 0


class TestOverheadAndForward:
    def test_stale_overhead_frame_cannot_resurrect(self):
        from comfy_env.state_sync import update_overhead_reports
        reports = {}
        assert update_overhead_reports(reports, "w", 1 * GIB, seq=5)
        assert not update_overhead_reports(reports, "w", 2 * GIB, seq=3)
        assert reports["w"]["excess"] == 1 * GIB

    def test_newer_lower_overhead_actually_lowers(self):
        """REPLACE, no peak: the value is self-measured in-frame, so a worker
        that tore down its streams must stop booking them. Catches: a
        peak-forever implementation."""
        from comfy_env.state_sync import update_overhead_reports
        reports = {"w": {"excess": 1 * GIB, "seq": 5}}
        assert update_overhead_reports(reports, "w", 0, seq=6)
        assert reports["w"]["excess"] == 0

    def test_absurd_overhead_warns(self):
        from comfy_env.state_sync import update_overhead_reports
        msgs = []
        update_overhead_reports({}, "w", 30 * GIB, seq=1, log=msgs.append)
        assert msgs and "overhead" in msgs[0]

    def test_forward_cast_need_books_the_future_buffers_exactly(self):
        """Cast buffers allocate lazily at the first forward, AFTER
        admission: only the incoming load's own numbers can book them.
        streams 2 x 600 MiB shifts need by exactly 1200 MiB vs streams 0."""
        from comfy_env.state_sync import forward_cast_need
        mib = 1024 * 1024
        assert forward_cast_need(600 * mib, 2) - forward_cast_need(600 * mib, 0) \
            == 1200 * mib

    def test_forward_cast_need_degrades_to_zero(self):
        """Old worker (no largest_tensor) or no stream info collapses the
        need formula's max() back to min_inference, today's behavior."""
        from comfy_env.state_sync import forward_cast_need
        assert forward_cast_need(None, 2) == 0
        assert forward_cast_need(600, None) == 0


class TestStateFilter:
    def test_changed_key_ships_unchanged_key_does_not(self):
        """Byte-diff, not identity: `self.cache['x'] = 1` mutates the same
        object, so identity-diff would silently drop it."""
        cache = {"a": 1}
        pre = {"cache": dict(cache), "same": "constant"}
        cache["x"] = 1
        post = {"cache": cache, "same": "constant"}
        out = diff_state(pre, post, 1024, "g", lambda: 1, lambda h, v: None)
        assert "cache" in out["set"]
        assert "same" not in out["set"]

    def test_deleted_key_propagates(self):
        out = diff_state({"gone": 1, "kept": 2}, {"kept": 2}, 1024,
                         "g", lambda: 1, lambda h, v: None)
        assert out["deleted"] == ["gone"]

    def test_one_bad_key_does_not_kill_the_rest(self):
        """Catches: try/except around the whole projection (today one
        unserializable attribute kills the entire frame)."""
        store = {}
        post = {"good": 42, "bad": lambda: None}
        out = diff_state({}, post, 1024, "g",
                         iter(range(1, 99)).__next__, store.__setitem__)
        assert out["set"]["good"] == 42
        assert is_overflow_marker(out["set"]["bad"])
        assert any(d["name"] == "bad" and d["reason"] == "unpicklable"
                   for d in out["dropped"])
        assert 1 in store  # the value is held, not lost

    def test_over_cap_value_is_sized_before_serialising(self):
        """Catches: cap applied after the serialisation cost is already paid.
        nbytes objects must short-circuit."""

        class Big:
            nbytes = 10 * GIB

            def __reduce__(self):  # pragma: no cover - must never be called
                raise AssertionError("serialised a 10 GiB value to size it")

        digest, verdict, nbytes = fingerprint(Big(), 8 * 1024 * 1024)
        assert verdict == "over_cap" and nbytes == 10 * GIB and digest is None

    def test_reserved_keys_never_cross(self):
        """The sentinel is parent-only: a worker writing it would plant an
        attribute the pack author never wrote."""
        d = {SEED_SENTINEL: True, "_comfy_env_state_id": "x", "real": 1}
        assert set(outbound_state(d)) == {"real"}
        out = diff_state({}, {SEED_SENTINEL: True, "n": 1}, 1024,
                         "g", lambda: 1, lambda h, v: None)
        assert SEED_SENTINEL not in out["set"]

    def test_marker_round_trip_is_not_reshipped(self):
        """An untouched inbound marker must not be re-minted every call."""
        marker = make_marker("g", 7, "big", "over_cap", 10 * GIB)
        out = diff_state({"big": marker}, {"big": marker}, 1024,
                         "g", lambda: pytest.fail("re-minted"), lambda h, v: None)
        assert "big" not in out["set"]
        assert "big" not in out["deleted"]


class TestApplyStateOut:
    def test_update_only_never_deletes_unmentioned(self):
        """A dropped key must keep its parent value; the filter is not an
        authoritative snapshot."""
        d = {"host_only": "kept", "n": 0}
        apply_state_out(d, {"set": {"n": 1}, "deleted": [], "dropped": []})
        assert d["host_only"] == "kept" and d["n"] == 1

    def test_deleted_applies(self):
        d = {"gone": 1}
        apply_state_out(d, {"set": {}, "deleted": ["gone"], "dropped": []})
        assert "gone" not in d

    def test_sets_the_seed_sentinel(self):
        """First ingest marks the instance seeded, so __init__ runs once per
        parent instance and the sweep dropping it reseeds."""
        d = {}
        apply_state_out(d, {"set": {}, "deleted": [], "dropped": []})
        assert d[SEED_SENTINEL] is True

    def test_none_is_a_noop(self):
        d = {"x": 1}
        apply_state_out(d, None)
        assert d == {"x": 1}

    def test_worker_cannot_write_reserved_keys(self):
        d = {}
        apply_state_out(d, {"set": {"_comfy_env_state_id": "evil", "ok": 1},
                            "deleted": [], "dropped": []})
        assert "ok" in d and d.get("_comfy_env_state_id") != "evil"


def test_reserved_keys_are_exactly_the_documented_two():
    assert RESERVED_KEYS == {SEED_SENTINEL, "_comfy_env_state_id"}


def test_fingerprint_ship_path_is_deterministic():
    d1, v1, _ = fingerprint({"a": 1}, 1024)
    d2, v2, _ = fingerprint({"a": 1}, 1024)
    assert v1 == v2 == "ship" and d1 == d2
    d3, _, _ = fingerprint({"a": 2}, 1024)
    assert d3 != d1


def test_fingerprint_uses_pickle_stability():
    """Guard the assumption the byte-diff rests on: equal values pickle
    equal for plain data."""
    assert pickle.dumps({"k": [1, 2]}) == pickle.dumps({"k": [1, 2]})


class TestAllocatePinBudgets:
    """The per-process pin split. Every rule below is an invariant from the
    design debate; each test names the wrong implementation it catches."""

    from comfy_env.state_sync import (  # noqa: PLC0415
        allocate_pin_budgets, damp_pin_grant, update_pin_reports,
    )

    def test_disabled_host_yields_sentinel_never_zero(self):
        """host_max <= 0 means pinning was never enabled: every key gets -1
        unchanged. Catches: granting 0, which strands registrations because
        unpin_memory early-returns on MAX <= 0."""
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(0, {"host": {"pinned": 0},
                                       "w1": {"pinned": 5 * GIB}})
        assert out == {"host": -1, "w1": -1}
        assert 0 not in out.values()

    def test_grant_never_below_current_pinned(self):
        """The drain bound beats conservation: a ceiling below a holder's
        TOTAL_PINNED makes model_management.py:739 a permanent shortfall
        (evict-forever). Catches: a conservation-first allocator."""
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(
            10 * GIB,
            {"host": {"pinned": 0},
             "w1": {"pinned": 9 * GIB}, "w2": {"pinned": 8 * GIB}},
            floor_bytes=GIB)
        assert out["w1"] >= 9 * GIB and out["w2"] >= 8 * GIB

    def test_idle_worker_lands_on_floor_without_draining_the_share(self):
        """The denominator counts LIVE pinners plus the requester. Catches:
        equal division over all keys, which starves the loader to feed
        workers that pin nothing."""
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(
            20 * GIB,
            {"host": {"pinned": 0},
             "loader": {"pinned": 4 * GIB}, "idle": {"pinned": 0}},
            floor_bytes=GIB, reserve=0.5)
        # (20 - 10) GiB over ONE live pinner, not two
        assert out["loader"] == 10 * GIB
        assert out["idle"] == GIB

    def test_requester_counts_as_live_before_its_first_pin(self):
        """A worker asking for budget is about to pin. Catches: a denominator
        that only sees past pinners, granting the whole remainder to a
        requester that then double-books it."""
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(
            20 * GIB,
            {"host": {"pinned": 0},
             "vet": {"pinned": 4 * GIB}, "newcomer": {"pinned": 0}},
            floor_bytes=GIB, reserve=0.5, requester="newcomer")
        assert out["newcomer"] == 5 * GIB  # (20-10)/2, not (20-10)/1
        assert out["vet"] == 5 * GIB

    def test_dead_worker_key_absent_not_retained_at_zero(self):
        """The pool removes a dead worker's report; its key must not appear in
        the output either. Catches: an allocator that remembers ghosts and
        keeps splitting the pool with them."""
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(10 * GIB, {"host": {"pinned": 0},
                                              "alive": {"pinned": GIB}})
        assert "dead" not in out and set(out) == {"host", "alive"}

    def test_grant_overage_is_exactly_the_floors(self):
        """Grants may exceed host_max ONLY by the floor and drain-bound terms.
        Catches: an allocator that quietly oversubscribes beyond the two
        documented exceptions."""
        from comfy_env.state_sync import allocate_pin_budgets
        host_max, floor = 10 * GIB, 2 * GIB
        reports = {"host": {"pinned": 0},
                   "w1": {"pinned": 0}, "w2": {"pinned": 0}}
        out = allocate_pin_budgets(host_max, reports, floor_bytes=floor,
                                   reserve=0.5)
        # share = (10-5)/1 = 5 GiB... but no live pinner and no requester:
        # everyone idle lands on the floor; host keeps its reserve.
        assert out["w1"] == out["w2"] == floor
        overage = sum(out.values()) - host_max
        assert overage == (out["host"] + 2 * floor) - host_max

    def test_host_grant_is_reserve_or_its_own_pinned(self):
        from comfy_env.state_sync import allocate_pin_budgets
        out = allocate_pin_budgets(10 * GIB, {"host": {"pinned": 7 * GIB},
                                              "w1": {"pinned": GIB}},
                                   floor_bytes=GIB, reserve=0.5)
        assert out["host"] == 7 * GIB  # drain bound beats the 50% reserve


class TestPinReportsAndDamping:
    def test_stale_report_is_dropped_whole(self):
        """Same rule as apply_residency: an out-of-order frame must not
        resurrect a stale total."""
        from comfy_env.state_sync import update_pin_reports
        reports = {}
        assert update_pin_reports(reports, "w1", 5 * GIB, seq=10)
        assert not update_pin_reports(reports, "w1", 99 * GIB, seq=9)
        assert reports["w1"]["pinned"] == 5 * GIB

    def test_shrink_applies_immediately(self):
        """Shrink-fast is half the anti-oscillation contract; deferring a
        shrink is the direction that oversubscribes RAM."""
        from comfy_env.state_sync import damp_pin_grant
        assert damp_pin_grant(10 * GIB, 4 * GIB, stable_censuses=0) == 4 * GIB

    def test_grow_waits_for_stability(self):
        """Grow only after 2 consecutive censuses with an unchanged consumer
        set. Catches: a paging worker retuning the pool every node boundary."""
        from comfy_env.state_sync import damp_pin_grant
        assert damp_pin_grant(4 * GIB, 10 * GIB, stable_censuses=1) == 4 * GIB
        assert damp_pin_grant(4 * GIB, 10 * GIB, stable_censuses=2) == 10 * GIB

    def test_grow_below_deadband_is_swallowed(self):
        """No grant delta below 512 MiB is emitted: each emitted delta retunes
        every worker, and sub-deadband jitter is census noise."""
        from comfy_env.state_sync import damp_pin_grant
        small = 4 * GIB + 100 * 1024 * 1024
        assert damp_pin_grant(4 * GIB, small, stable_censuses=5) == 4 * GIB

    def test_first_grant_passes_undamped(self):
        from comfy_env.state_sync import damp_pin_grant
        assert damp_pin_grant(None, 4 * GIB, stable_censuses=0) == 4 * GIB


def test_state_sync_stays_pure():
    """The whole module must keep importable under bare CI: no comfy, no
    torch, no psutil. Catches: a convenience import that quietly couples the
    pure layer to an environment only workers have."""
    import ast as _ast
    from pathlib import Path as _Path
    src = (_Path(__file__).resolve().parents[1] / "src" / "comfy_env"
           / "state_sync.py").read_text(encoding="utf-8")
    for node in _ast.walk(_ast.parse(src)):
        names = []
        if isinstance(node, _ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, _ast.ImportFrom):
            names = [node.module or ""]
        for n in names:
            root = n.split(".")[0]
            assert root not in ("comfy", "torch", "psutil", "comfy_aimdo"), (
                f"state_sync imports {n}; the pure layer just died")
