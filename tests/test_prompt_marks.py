"""Tests for the prompt-epoch pin marks (worker current_prompt protection).

The bug: a worker runs no executor, so upstream's PromptModelTracker never
marks its models and pin-eviction tier 1 (cp=False, active NOT consulted)
evicts this prompt's warm models like stale leftovers -- including a model
mid-load, whose unregistered staging pages can be decommitted under an
in-flight async copy. Every test names the wrong implementation it catches.
"""

import ast
from pathlib import Path

from comfy_env.memory_manager import apply_prompt_marks
from comfy_env.state_sync import (
    PROMPT_MARK_DECAY_CALLS,
    clear_on_epoch_change,
    mark_on_load,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
WORKER = SRC / "isolation" / "workers" / "_persistent_worker.py"
SUBPROCESS = SRC / "isolation" / "workers" / "subprocess.py"
POOL = SRC / "isolation" / "pool.py"
PROXY = SRC / "isolation" / "model_patcher.py"


class TestClearOnEpochChange:
    def test_untouched_same_epoch_mark_survives_the_call(self):
        """THE bug row: the VAE marked in call N must stay protected through
        call N+1's unet load in the same prompt. Catches: clearing at every
        node boundary, which leaves the between-nodes window unprotected."""
        marks, to_clear = clear_on_epoch_change({"A": (1, 1)}, 1, 2,
                                                ["A", "B"])
        assert to_clear == [] and marks["A"] == (1, 1)

    def test_epoch_change_clears_the_previous_prompts_marks(self):
        """Catches: never clearing, so a two-pack worker protects pack A's
        stale models during pack B's prompt (priority inversion)."""
        marks, to_clear = clear_on_epoch_change({"A": (1, 3)}, 2, 4,
                                                ["A", "B"])
        assert to_clear == ["A"] and "A" not in marks

    def test_none_epoch_retains_within_decay_then_clears(self):
        """The sticky-with-decay fallback. Dark (no marks) would silently
        reopen the corruption window on a host-patch failure; indefinite
        sticky violates the mandatory-decay invariant. Catches both."""
        marks, to_clear = clear_on_epoch_change(
            {"A": (None, 1)}, None, 1 + PROMPT_MARK_DECAY_CALLS - 1, ["A"])
        assert to_clear == [] and "A" in marks
        marks, to_clear = clear_on_epoch_change(
            {"A": (None, 1)}, None, 1 + PROMPT_MARK_DECAY_CALLS, ["A"])
        assert to_clear == ["A"] and "A" not in marks

    def test_vanished_id_is_pruned_silently(self):
        """A flip emitted for a dead id would make the applier chase ghosts
        (and a KeyError there could mask the node result)."""
        marks, to_clear = clear_on_epoch_change({"Z": (1, 1)}, 1, 2, ["A"])
        assert "Z" not in marks and to_clear == []


class TestMarkOnLoad:
    def test_restart_marks_from_an_empty_table(self):
        """Catches: requiring a previously seen epoch to mark, which leaves
        the first call after a worker restart unprotected for the whole
        prompt."""
        marks, to_set = mark_on_load({}, 7, 1, ["A"])
        assert to_set == ["A"] and marks["A"] == (7, 1)

    def test_reused_model_migrates_epochs_without_a_flip(self):
        """A model already marked (flag True) that appears in the new
        prompt's first load keeps its flag and re-stamps: a clear/set pair
        would open a one-tier gap mid-walk."""
        marks, to_set = mark_on_load({"A": (1, 3)}, 2, 4, ["A"])
        assert to_set == [] and marks["A"] == (2, 4)

    def test_remark_after_a_preamble_clear_flips_again(self):
        """After the epoch change cleared A (flag now False), a reload of A
        in the new prompt must flip it back True."""
        marks, to_clear = clear_on_epoch_change({"A": (1, 3)}, 2, 4, ["A"])
        assert to_clear == ["A"]
        marks, to_set = mark_on_load(marks, 2, 4, ["A"])
        assert to_set == ["A"]

    def test_sticky_fallback_marks_too(self):
        """No token still marks (with the decay clock running): the
        corruption window must stay closed on legacy paths."""
        marks, to_set = mark_on_load({}, None, 5, ["A"])
        assert to_set == ["A"] and marks["A"] == (None, 5)


class _DynPatcher:
    def __init__(self):
        self.flag = None
        self.model = object()

    def is_dynamic(self):
        return True

    def set_in_use_by_current_prompt(self, v):
        self.flag = v


class TestApplyPromptMarks:
    def test_flips_set_and_clear(self):
        a, b = _DynPatcher(), _DynPatcher()
        n = apply_prompt_marks({"a": a, "b": b}, ["a"], ["b"])
        assert n == 2 and a.flag is True and b.flag is False

    def test_ledger_mode_object_is_skipped_not_crashed(self):
        """A base ModelPatcher has no dynamic_pins and no setter; an
        aimdo-off worker must be structurally inert. Catches: marking that
        AttributeErrors ledger workers."""

        class Ledger:
            def is_dynamic(self):
                return False
        assert apply_prompt_marks({"m": Ledger()}, ["m"], []) == 0
        assert apply_prompt_marks({"m": object()}, ["m"], []) == 0

    def test_old_comfy_without_the_setter_is_skipped(self):
        class Dyn:
            def is_dynamic(self):
                return True
        assert apply_prompt_marks({"m": Dyn()}, ["m"], []) == 0

    def test_missing_id_in_registry_is_a_noop(self):
        assert apply_prompt_marks({}, ["ghost"], ["ghost2"]) == 0

    def test_a_raising_patcher_does_not_kill_the_rest(self):
        class Bomb:
            def is_dynamic(self):
                raise RuntimeError("boom")
        ok = _DynPatcher()
        n = apply_prompt_marks({"bomb": Bomb(), "ok": ok}, ["bomb", "ok"], [])
        assert n == 1 and ok.flag is True


class TestPromptMarkSeam:
    def test_mark_call_precedes_the_original_load(self):
        """The pressure fires DURING the load (ensure_pin_registerable inside
        pin_memory), and pin tier 1 ignores `active`: end-of-call marking
        leaves every model's first loaded call exposed to the
        decommit-under-async-copy window."""
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_shimmed_load_models_gpu")
        src = ast.unparse(fn)
        assert "_prompt_marks_on_load" in src
        assert src.index("_prompt_marks_on_load") \
            < src.index("_original_load_models_gpu(models"), (
                "marking happens after the real load; the first call's "
                "corruption window is open")

    def test_epoch_clear_precedes_method_execution(self):
        """A dead prompt's marks alive during this call's load would demote a
        genuinely stale model behind them in the tier walk."""
        src = WORKER.read_text(encoding="utf-8")
        assert src.index("_prompt_marks_preamble(request)") \
            < src.index("with _infer_mode():")

    def test_epoch_rides_both_request_types(self):
        """The shim installs globally at worker start, so function-style
        packs calling comfy loaders create marks that only a call_module
        preamble clear can retire. Catches: token on call_method only."""
        tree = ast.parse(SUBPROCESS.read_text(encoding="utf-8"))
        for rtype in ("call_method", "call_module"):
            dicts = [n for n in ast.walk(tree)
                     if isinstance(n, ast.Dict)
                     and any(isinstance(k, ast.Constant) and k.value == "type"
                             for k in n.keys)
                     and any(isinstance(v, ast.Constant) and v.value == rtype
                             for v in n.values)]
            assert dicts, f"{rtype} request dict not found"
            assert any(
                any(isinstance(k, ast.Constant) and k.value == "prompt_gen"
                    for k in d.keys)
                for d in dicts), f"prompt_gen missing from {rtype} requests"

    def test_proxy_stays_out_of_the_dynamic_paths(self):
        """current_prompt's only consumer is worker-side pin tiering; a mark
        on the host proxy would be a write with no reader, and flipping
        is_dynamic would enter the proxy into every dynamic pin path."""
        src = PROXY.read_text(encoding="utf-8")
        tree = ast.parse(src)
        touches = [n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute) and n.attr == "dynamic_pins"]
        assert not touches, f"the proxy touches dynamic_pins at {touches}"
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "is_dynamic")
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        assert all(isinstance(r.value, ast.Constant) and r.value.value is False
                   for r in returns)

    def test_one_switch_gates_both_ends(self):
        """COMFY_ENV_PIN_MARKS off must silence the host patch AND the worker
        writes; a half-off state would be sticky marks with no epoch source,
        permanently."""
        pool_src = POOL.read_text(encoding="utf-8")
        worker_src = WORKER.read_text(encoding="utf-8")
        assert "PIN_MARKS_ENV_VAR" in pool_src
        assert "COMFY_ENV_PIN_MARKS" in worker_src

    def test_host_patch_bumps_before_calling_the_original(self):
        """The epoch must be visible to requests dispatched DURING the prompt
        it names; bumping after the original start would race the first
        node."""
        tree = ast.parse(POOL.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_epoch_start")
        src = ast.unparse(fn)
        assert src.index("PROMPT_GEN[0] += 1") < src.index("_orig_start")

    def test_counter_wrapper_calls_the_original_unconditionally(self):
        """The eviction counters are observability only: the wrapper must
        return the original's result verbatim, never gate or alter it."""
        memmgr = SRC / "memory_manager.py"
        tree = ast.parse(memmgr.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_counted_free_model_pins")
        src = ast.unparse(fn)
        assert "_orig(size, subsets, current_prompt, active" in src
        assert "return freed" in src

    def test_sweep_runs_at_both_node_end_finallys(self):
        """The catch-up for loads that bypass the shim must cover both call
        paths, like release_node_boundary does."""
        src = WORKER.read_text(encoding="utf-8")
        assert src.count("_prompt_marks_sweep()") >= 2


class TestSweepScope:
    """The node-end sweep is catch-up for shim-bypassing loads only. It must
    mark models NEWLY resident this call, never survivors of a previous call,
    or it becomes the blanket-marking the contract forbade (and would re-mark
    survivors to the current epoch immediately after the preamble cleared
    them, defeating the epoch clear)."""

    def test_sweep_marks_only_newly_present_not_survivors(self):
        """Simulates the preamble/finally delta the worker computes. A model
        resident since before the call (a survivor of the previous prompt,
        cleared by the preamble) must NOT be re-marked; a model that appeared
        during the call (a bypassing load) must be."""
        resident_at_start = {"survivor"}
        resident_now = {"survivor", "bypassed_load"}
        newly = resident_now - resident_at_start
        marks = {}  # preamble cleared everything
        missing = [k for k in newly if k not in marks]
        assert missing == ["bypassed_load"]
        assert "survivor" not in missing, (
            "the sweep re-marks a survivor, re-stamping it to the current "
            "epoch right after the preamble cleared it")

    def test_sweep_is_scoped_by_the_preamble_snapshot(self):
        """Source guard: the sweep must diff against the residency snapshot
        the preamble took, not the raw registry."""
        src = WORKER.read_text(encoding="utf-8")
        assert "_resident_at_call_start[0] = set(" in src, (
            "the preamble no longer snapshots residency; the sweep cannot "
            "tell a newly-loaded model from a survivor")
        fn_start = src.index("def _prompt_marks_sweep")
        fn = src[fn_start:src.index("\n    def ", fn_start + 10)]
        assert "_resident_at_call_start[0]" in fn, (
            "the sweep marks all resident models, not the newly present ones")
