"""Contract: the number comfy-env publishes is right in both directions.

Pure unit tests, no torch and no comfy. Every test names the wrong
implementation it exists to catch, and the priority ones are the mistakes
that produce a plausible-looking but silently wrong number: the platform
split, entitlement versus current residency, and the shrink rule.
"""

import ast
from pathlib import Path

from comfy_env import reserve as R

GIB = 1024 ** 3
SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"


class TestEntitlement:
    def test_books_the_context_floor_even_with_nothing_resident(self):
        """Catches: booking zero for an idle worker. A live worker holds its
        CUDA context whether or not it holds a model, measured at 276 to
        300 MiB, and a host that takes that space OOMs the worker's next
        call."""
        assert R.entitlement(0) == R.CONTEXT_FLOOR_BYTES

    def test_uses_high_water_not_current_residency(self):
        """Catches: tracking current residency, which hands the host the
        worker's space between calls and takes it back by OOM on the next
        one."""
        assert R.entitlement(4 * GIB) == R.CONTEXT_FLOOR_BYTES + 4 * GIB

    def test_negative_and_none_are_floored_not_subtracted(self):
        """Catches: a bad reading reducing the entitlement below the floor."""
        assert R.entitlement(-5) == R.CONTEXT_FLOOR_BYTES
        assert R.entitlement(None) == R.CONTEXT_FLOOR_BYTES


class TestCharge:
    def test_device_wide_charges_only_the_unheld_headroom(self):
        """THE double-book. On Linux cudaMemGetInfo already reports worker
        VRAM, so charging residency again removed 8.9 GiB of usable card in
        measurement. Only what the worker will take BEYOND what it holds is
        new information to the host."""
        entitled = R.entitlement(6 * GIB)
        assert R.charge(entitled, 6 * GIB, process_local_free=False) == \
            R.CONTEXT_FLOOR_BYTES

    def test_windows_charges_the_whole_entitlement(self):
        """Catches: applying the Linux rule everywhere. On WDDM the host's
        reading is per process, so the worker is invisible and nothing would
        be reserved at all."""
        entitled = R.entitlement(6 * GIB)
        assert R.charge(entitled, 6 * GIB, process_local_free=True) == entitled

    def test_an_idle_worker_still_charges_on_both_platforms(self):
        """Catches: charging nothing once a worker has released. The context
        is still held and the worker will need its high-water again."""
        entitled = R.entitlement(6 * GIB)
        assert R.charge(entitled, 0, False) == entitled
        assert R.charge(entitled, 0, True) == entitled

    def test_residency_above_entitlement_never_goes_negative(self):
        """Catches: a negative charge quietly reducing another worker's."""
        assert R.charge(R.entitlement(1 * GIB), 99 * GIB, False) == 0


class TestTotalReserve:
    def test_preserves_the_operators_own_reserve(self):
        """Catches: overwriting EXTRA_RESERVED_VRAM. That value is the
        operator's --reserve-vram instruction; comfy-env adds to it."""
        assert R.total_reserve(2 * GIB, [1 * GIB]) == 3 * GIB

    def test_caps_below_the_whole_card(self):
        """Catches: a reserve at or above the device total, which converts an
        over-book into every host load failing outright."""
        assert R.total_reserve(0, [100 * GIB], device_total=24 * GIB) == \
            int(24 * GIB * 0.75)

    def test_no_workers_publishes_exactly_the_base(self):
        """Catches: a floor that leaks in with no workers running, which
        would shrink the card for a user who has no packs loaded."""
        assert R.total_reserve(400 * 1024 ** 2, []) == 400 * 1024 ** 2


class TestNextReserve:
    def test_growing_never_needs_permission(self):
        """The invariant: raise before the memory is taken."""
        assert R.next_reserve(1 * GIB, 5 * GIB, shrink_allowed=False) == 5 * GIB

    def test_shrinking_without_a_receipt_is_refused(self):
        """Catches the dangerous direction: lowering the reserve because a
        worker is EXPECTED to have released hands the host space that is
        still occupied, and the host loads straight into it."""
        assert R.next_reserve(5 * GIB, 1 * GIB, shrink_allowed=False) == 5 * GIB

    def test_shrinking_with_a_receipt_applies(self):
        assert R.next_reserve(5 * GIB, 1 * GIB, shrink_allowed=True) == 1 * GIB

    def test_equal_is_not_a_shrink(self):
        """Catches: an off-by-one that treats a no-op republish as a shrink
        and so requires a receipt for it."""
        assert R.next_reserve(5 * GIB, 5 * GIB, shrink_allowed=False) == 5 * GIB


class TestAskTarget:
    def test_reproduces_upstreams_expression(self):
        """Catches: comfy-env inventing its own admission formula again. The
        host frees weights * 1.1 + max(inference, required + reserved); a
        smaller shape makes the host free less for a worker load than for an
        identical in-process one."""
        got = R.ask_target(weights=10 * GIB, slack=1.1,
                           min_inference=1 * GIB, extra_reserved=400 * 1024 ** 2,
                           want_inference=0)
        assert got == int(10 * GIB * 1.1) + 1 * GIB

    def test_the_reserve_term_wins_when_it_exceeds_the_inference_floor(self):
        """Catches: dropping extra_reserved from the max, which under-asks by
        exactly the operator's own reserve."""
        got = R.ask_target(weights=0, slack=1.1, min_inference=1 * GIB,
                           extra_reserved=4 * GIB, want_inference=1 * GIB)
        assert got == 5 * GIB

    def test_slack_below_upstreams_is_visible_as_a_shortfall(self):
        """Guards the constant's direction rather than its value: a smaller
        multiplier must produce a smaller ask, so the regression is
        detectable if someone tunes it."""
        low = R.ask_target(12 * GIB, 1.02, 0, 0)
        high = R.ask_target(12 * GIB, 1.1, 0, 0)
        assert high - low > 600 * 1024 ** 2


def test_module_imports_nothing_at_top_level():
    """Catches: an import creeping into a module staged into workers and
    exercised by bare CI with no torch."""
    tree = ast.parse((SRC / "reserve.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
            "reserve.py imports at module scope: " + ast.unparse(node))


class TestAimdoHeadroom:
    """The paged path only sees the reserve through the pager's own headroom,
    and the pager's budget is seed plus what comfy-env added, never the base
    twice."""

    def test_forwards_seed_plus_what_comfy_env_added(self):
        """Catches: forwarding the published reserve as is, which books the
        operator's --reserve-vram a second time on the paged path."""
        from comfy_env.reserve import aimdo_headroom
        seed, base = 2 * GIB, 400 * 1024 ** 2
        published = base + 3 * GIB
        assert aimdo_headroom(seed, published, base) == seed + 3 * GIB

    def test_no_seed_means_aimdos_own_default(self):
        """Catches: treating an absent --reserve-vram as zero. aimdo starts
        at its compile time floor, and a forward below it would LOWER the
        host's headroom the moment a worker appears."""
        from comfy_env.reserve import aimdo_headroom, AIMDO_DEFAULT_HEADROOM
        assert aimdo_headroom(None, 1 * GIB, 1 * GIB) == AIMDO_DEFAULT_HEADROOM
        assert aimdo_headroom(None, 0, 0) == AIMDO_DEFAULT_HEADROOM

    def test_a_reserve_below_the_base_adds_nothing(self):
        """Catches: a negative addition when the published number is capped
        or has not been written yet."""
        from comfy_env.reserve import aimdo_headroom
        assert aimdo_headroom(1 * GIB, 100, 4 * GIB) == 1 * GIB


class TestReserveForRequester:
    """A worker's own growth charge must not be asked for on top of the load
    that IS that growth."""

    def test_excludes_the_requesters_own_charge(self):
        """Catches: passing extra_reserved_memory() straight into the ask,
        which evicts host models for the same bytes twice."""
        from comfy_env.reserve import reserve_for_requester
        assert reserve_for_requester(6 * GIB, 2 * GIB) == 4 * GIB

    def test_a_charge_larger_than_the_reserve_floors_at_zero(self):
        """Catches: a negative reserve term, which would SHRINK the ask below
        what an identical in-process load gets."""
        from comfy_env.reserve import reserve_for_requester
        assert reserve_for_requester(1 * GIB, 3 * GIB) == 0

    def test_no_charge_leaves_the_reserve_whole(self):
        from comfy_env.reserve import reserve_for_requester
        assert reserve_for_requester(6 * GIB, 0) == 6 * GIB
        assert reserve_for_requester(6 * GIB, None) == 6 * GIB
