"""Contract: one ordered switch, and its refusals are visible.

Pure unit tests. The failure that matters most is a SILENT demotion: four
worker environments on the development machine ran a different memory
manager than their host and nothing said so.
"""

import ast
from pathlib import Path

from comfy_env import memlevel as M

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"

ALL = {"aimdo_available": True, "marks_available": True,
       "pressure_available": True}
NO_AIMDO = dict(ALL, aimdo_available=False)
NO_MARKS = dict(ALL, marks_available=False)
NO_PRESSURE = dict(ALL, pressure_available=False)


class TestAuto:
    def test_picks_the_best_available(self):
        assert M.resolve("auto", ALL) == (M.SHARED, None)

    def test_stops_where_the_host_stops_and_says_why(self):
        """Catches the silent demotion: an environment running the legacy
        ledger against a paging host with nothing reported."""
        level, note = M.resolve("auto", NO_AIMDO)
        assert level == M.LEDGER
        assert note and "comfy-aimdo" in note

    def test_missing_marks_stops_at_ledger_not_paged(self):
        """Marks are not a feature above paging, they are what makes paging
        safe in a worker: upstream's first eviction tier matches every model
        not marked as the current prompt's. Catches: treating them as
        optional and paging without them."""
        level, note = M.resolve("auto", NO_MARKS)
        assert level == M.LEDGER
        assert note and "2026-07-28" in note

    def test_missing_pressure_only_costs_the_top_level(self):
        level, note = M.resolve("auto", NO_PRESSURE)
        assert level == M.PAGED
        assert note and "2026-07-09" in note

    def test_empty_and_none_mean_auto(self):
        """Catches: an unset variable resolving to off, which would silently
        disable memory management for every user who never set it."""
        assert M.resolve(None, ALL)[0] == M.SHARED
        assert M.resolve("", ALL)[0] == M.SHARED
        assert M.resolve("  ", ALL)[0] == M.SHARED


class TestExplicit:
    def test_a_reachable_request_is_honoured_silently(self):
        """A deliberate downgrade is a decision, not a fault. Catches:
        warning on every explicit choice, which trains the operator to
        ignore the channel that carries real demotions."""
        assert M.resolve("ledger", ALL) == (M.LEDGER, None)
        assert M.resolve("off", ALL) == (M.OFF, None)

    def test_an_unreachable_request_runs_lower_and_says_so(self):
        """Catches: refusing to start. Failing a whole pack because one
        memory feature is unavailable is failing on availability, which the
        house rules forbid."""
        level, note = M.resolve("shared", NO_PRESSURE)
        assert level == M.PAGED
        assert note and "shared was requested" in note

    def test_case_and_whitespace_are_tolerated(self):
        assert M.resolve("  PAGED \n", ALL)[0] == M.PAGED

    def test_an_unknown_level_does_not_silently_become_off(self):
        """Catches: a typo disabling memory management with no message."""
        level, note = M.resolve("pagd", ALL)
        assert level == M.SHARED
        assert note and "pagd" in note and "Valid" in note


class TestDerivedFlags:
    def test_marks_switch_with_paging_never_apart(self):
        """The half-off state that must be impossible: marks on with paging
        off, or paging on with marks off."""
        for level in M.ORDER:
            flags = M.derived_flags(level)
            assert flags["aimdo"] == flags["marks"], level

    def test_each_level_is_a_superset_of_the_one_below(self):
        """Catches: a level that turns something OFF that a lower level had,
        which would make the ordering meaningless."""
        previous = None
        for level in M.ORDER:
            flags = M.derived_flags(level)
            if previous is not None:
                for key, was in previous.items():
                    assert flags[key] or not was, (level, key)
            previous = flags

    def test_off_enables_nothing(self):
        assert not any(M.derived_flags(M.OFF).values())

    def test_pressure_is_the_top_level_only(self):
        assert M.derived_flags(M.PAGED)["pressure"] is False
        assert M.derived_flags(M.SHARED)["pressure"] is True


def test_module_imports_nothing_at_top_level():
    tree = ast.parse((SRC / "memlevel.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
            "memlevel.py imports at module scope: " + ast.unparse(node))
