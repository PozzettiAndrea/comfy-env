"""Contract: what comfy-env requires of its host is written down and checked.

Every test names the wrong implementation it exists to catch. The unit under
test is pure and needs no torch and no comfy, which is the point: the
contract is data, so it can be evaluated anywhere, including by a static
sweep over ComfyUI's history.
"""

import ast
from pathlib import Path

import pytest

from comfy_env import contract as C

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"


class _Stub:
    def __init__(self, **attrs):
        for key, value in attrs.items():
            setattr(self, key, value)


class TestEvaluate:
    def test_a_missing_fatal_entry_refuses(self):
        """Catches: treating every absence as a degrade, which is how a
        missing symbol becomes a wrong number instead of a refusal."""
        present = {k: True for k in C.required_keys(C.HOST, (C.FLOOR,))}
        present["comfy.model_management.EXTRA_RESERVED_VRAM"] = False
        ok, failures, notes = C.evaluate(present, C.HOST, (C.FLOOR,))
        assert ok is False
        assert any("EXTRA_RESERVED_VRAM" in f for f in failures)

    def test_a_missing_degrade_entry_does_not_refuse_but_is_named(self):
        """Catches: refusing on availability, which would take a whole pack
        out of ComfyUI because one optional feature is unavailable."""
        present = {k: True for k in C.required_keys(C.HOST, (C.FLOOR,))}
        present["comfy.cli_args.args"] = False
        ok, failures, notes = C.evaluate(present, C.HOST, (C.FLOOR,))
        assert ok is True and not failures
        assert any("cli_args.args" in n for n in notes)

    def test_unknown_is_never_a_verdict(self):
        """Catches: scoring an unimportable module as missing. A checker that
        cannot see the module must not manufacture a failure, or every
        torch-less CI run would refuse."""
        present = {k: None for k in C.required_keys(C.HOST, (C.FLOOR,))}
        ok, failures, notes = C.evaluate(present, C.HOST, (C.FLOOR,))
        assert ok is True and not failures and not notes

    def test_a_failure_names_the_version_that_introduced_the_symbol(self):
        """Catches: a bare "attribute missing" message. An operator needs to
        read "needs ComfyUI <x>", not go and bisect it themselves."""
        present = {k: True for k in C.required_keys(C.HOST, (C.FLOOR,))}
        present["comfy.model_management.extra_reserved_memory"] = False
        _, failures, _ = C.evaluate(present, C.HOST, (C.FLOOR,))
        assert any("2024-09-01" in f for f in failures)

    def test_tiers_are_not_evaluated_unless_requested(self):
        """Catches: checking paged entries on a ledger worker, which would
        report a missing pager as a problem when it is a deliberate mode."""
        present = {"comfy_aimdo.control.get_total_vram_usage": False}
        ok, failures, _ = C.evaluate(present, C.WORKER, (C.FLOOR,))
        assert ok is True and not failures
        ok, failures, _ = C.evaluate(present, C.WORKER, (C.FLOOR, C.PAGED))
        assert ok is False

    def test_host_only_entries_are_not_required_of_a_worker(self):
        """Catches: one flat list applied to both sides, which would make a
        worker refuse over a symbol only the host ever calls."""
        keys = C.required_keys(C.WORKER, (C.FLOOR,))
        assert "comfy.model_management.free_memory" not in keys
        assert "comfy.model_management.get_free_memory" in keys


class TestProbe:
    def test_presence_is_checked_by_name_not_by_a_default(self):
        """Catches: getattr(module, name, default), which is exactly how a
        renamed upstream symbol turns into a silently wrong number."""
        stub = _Stub(get_free_memory=lambda *a: 0)
        present = C.probe_present(
            ["comfy.model_management.get_free_memory",
             "comfy.model_management.EXTRA_RESERVED_VRAM"],
            modules={"comfy.model_management": stub})
        assert present["comfy.model_management.get_free_memory"] is True
        assert present["comfy.model_management.EXTRA_RESERVED_VRAM"] is False

    def test_a_callable_entry_that_is_not_callable_fails(self):
        """Catches: hasattr alone. Upstream turning a function into a plain
        attribute passes a presence check and raises at the call site."""
        stub = _Stub(free_memory=42)
        present = C.probe_present(
            ["comfy.model_management.free_memory"],
            modules={"comfy.model_management": stub})
        assert present["comfy.model_management.free_memory"] is False

    def test_an_absent_module_is_unknown_not_missing(self):
        present = C.probe_present(
            ["comfy.model_management.free_memory"],
            modules={"comfy.model_management": None})
        assert present["comfy.model_management.free_memory"] is None

    def test_check_against_a_stub_host_refuses_on_a_fatal_gap(self):
        """End to end through the real entry point, so the wiring between
        required_keys, probe_present and evaluate is exercised together."""
        stub = _Stub(get_torch_device=lambda: None, get_free_memory=lambda d: 0,
                     free_memory=lambda *a: None,
                     minimum_inference_memory=lambda: 0,
                     vram_state=None)
        ok, failures, _ = C.check(
            C.HOST, (C.FLOOR,), modules={"comfy.model_management": stub,
                                         "comfy.cli_args": _Stub(args=None)})
        assert ok is False
        assert any("EXTRA_RESERVED_VRAM" in f for f in failures)


class TestContractShape:
    def test_every_entry_is_complete_and_uses_known_values(self):
        """Catches: a half-written entry, which silently never evaluates."""
        for entry in C.CONTRACT:
            assert set(entry) == {"module", "attr", "kind", "severity",
                                  "tier", "side", "why", "since"}, entry
            assert entry["kind"] in ("attr", "callable")
            assert entry["severity"] in (C.FATAL, C.DEGRADE)
            assert entry["tier"] in (C.FLOOR, C.PAGED, C.SHARED)
            assert entry["side"] in (C.HOST, C.WORKER, C.BOTH)
            assert entry["why"] and len(entry["why"]) > 20

    def test_the_module_imports_nothing_at_top_level(self):
        """Catches: an import creeping in. This module is staged into workers
        and must be importable in bare CI with no torch and no comfy."""
        tree = ast.parse((SRC / "contract.py").read_text(encoding="utf-8"))
        for node in tree.body:
            assert not isinstance(node, (ast.Import, ast.ImportFrom)), (
                "contract.py imports at module scope: " + ast.unparse(node))

    def test_keys_are_unique(self):
        """Catches: a duplicated entry, where one severity silently shadows
        the other depending on iteration order."""
        keys = ["{}.{}".format(e["module"], e["attr"]) for e in C.CONTRACT]
        assert len(keys) == len(set(keys))


@pytest.mark.parametrize("tier", [C.FLOOR, C.PAGED, C.SHARED])
def test_each_tier_has_entries(tier):
    """Catches: a tier that exists in the level switch but requires nothing,
    so its availability is never actually checked."""
    assert any(e["tier"] == tier for e in C.CONTRACT)


class TestWorkerSideIsActuallyEvaluated:
    """Every comfy_aimdo entry is WORKER side and PAGED tier. The only caller
    was the host at FLOOR, so none of them was evaluated by anything: the
    static sweep marks them absent by design and the module is staged into
    every worker where nothing imported it. These pin the wiring that makes
    the entries mean something."""

    def test_the_worker_program_runs_the_check(self):
        """Catches: entries that exist to be read by humans. A contract that
        nothing evaluates is a comment with a data structure around it."""
        import pathlib
        src = pathlib.Path(
            "src/comfy_env/isolation/workers/_persistent_worker.py"
        ).read_text(encoding="utf-8")
        assert "import contract as _contract" in src
        assert "side=_contract.WORKER" in src

    def test_the_paged_tier_is_only_asked_for_when_paging(self):
        """Catches: reporting missing aimdo symbols on a ledger worker, where
        their absence is correct and expected."""
        import pathlib
        src = pathlib.Path(
            "src/comfy_env/isolation/workers/_persistent_worker.py"
        ).read_text(encoding="utf-8")
        i = src.index("side=_contract.WORKER")
        window = src[i - 600:i]
        assert '_mem_info.get("manager") == "aimdo"' in window
        assert "_contract.PAGED" in window

    def test_the_worker_entries_are_reachable_at_all(self):
        """Catches the state this fixes: required_keys(WORKER, PAGED) coming
        back empty, or the aimdo entries being on a side nothing checks."""
        from comfy_env import contract
        keys = contract.required_keys(contract.WORKER,
                                      (contract.FLOOR, contract.PAGED))
        assert any(k.startswith("comfy_aimdo.") for k in keys)

    def test_a_missing_aimdo_symbol_is_reported_not_fatal_to_startup(self):
        """Catches: a worker refusing to start over a DEGRADE entry. A missing
        symbol means this worker pages worse; the answer is to say so."""
        from comfy_env import contract
        present = {k: True for k in contract.required_keys(
            contract.WORKER, (contract.FLOOR, contract.PAGED))}
        present["comfy_aimdo.model_vbar.vbars_reset_watermark_limits"] = False
        ok, failures, notes = contract.evaluate(
            present, side=contract.WORKER,
            tiers=(contract.FLOOR, contract.PAGED))
        assert ok is True
        assert any("vbars_reset_watermark_limits" in n for n in notes)
