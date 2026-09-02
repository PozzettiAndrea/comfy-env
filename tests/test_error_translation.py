"""Tests for the typed OOM/interrupt translation at the worker IPC frontier.

Every test names the wrong implementation it exists to catch. The pure unit
under test is ``translate_error`` with an injected registry, so bare CI (no
torch, no comfy) drives the whole decision table; the ast guards pin the seam
sites that CI cannot execute.
"""

import ast
from pathlib import Path

from comfy_env.isolation.errors import translate_error
from comfy_env.isolation.workers.base import InterruptRequested, WorkerError

SRC = Path(__file__).resolve().parents[1] / "src" / "comfy_env"
ERRORS = SRC / "isolation" / "errors.py"
METADATA = SRC / "isolation" / "metadata.py"
WORKER = SRC / "isolation" / "workers" / "_persistent_worker.py"
POOL = SRC / "isolation" / "pool.py"


class FakeOOM(RuntimeError):
    pass


class FakeInterrupt(BaseException):
    pass


REGISTRY = {"oom": lambda: FakeOOM, "interrupt": lambda: FakeInterrupt}


class TestTranslateError:
    def test_oom_kind_yields_registry_class_chained_to_the_original(self):
        """Catches: translation that loses the WorkerError (and with it the
        worker traceback) instead of chaining it."""
        we = WorkerError("CUDA out of memory. Tried to allocate 20.00 GiB",
                         traceback="worker stack", error_kind="oom")
        out = translate_error(we, registry=REGISTRY)
        assert isinstance(out, FakeOOM)
        assert out.__cause__ is we
        assert "worker stack" in str(out)

    def test_no_kind_returns_the_same_object(self):
        """Old worker, new parent: no error_kind key means today's behavior,
        the identical WorkerError, not a copy."""
        we = WorkerError("boom", error_kind=None)
        assert translate_error(we, registry=REGISTRY) is we

    def test_unknown_kind_returns_the_same_object(self):
        """A future worker with a richer vocabulary must degrade, not KeyError."""
        we = WorkerError("boom", error_kind="thermal_runaway")
        assert translate_error(we, registry=REGISTRY) is we

    def test_resolver_returning_none_passes_through(self):
        """The torch-less host trap: never synthesize a stand-in class, since
        comfy's OOM_EXCEPTION = Exception fallback would then make is_oom true
        for every exception on that host."""
        we = WorkerError("CUDA out of memory", error_kind="oom")
        assert translate_error(we, registry={"oom": lambda: None}) is we

    def test_resolver_raising_passes_through_not_up(self):
        """translate_error returns, never raises: an ImportError here must not
        replace the real error the user needs to see."""
        def bad():
            raise ImportError("no torch")
        we = WorkerError("CUDA out of memory", error_kind="oom")
        assert translate_error(we, registry={"oom": bad}) is we

    def test_kind_wins_over_a_contradicting_message(self):
        """Catches: any implementation that sniffs message text. A filename can
        contain the words; a real OOM can be localized with none of them."""
        false_positive = WorkerError(
            "FileNotFoundError: llama-out-of-memory-bench.safetensors")
        assert translate_error(false_positive, registry=REGISTRY) is false_positive
        localized = WorkerError("显存不足", error_kind="oom")
        assert isinstance(translate_error(localized, registry=REGISTRY), FakeOOM)

    def test_oom_stats_ride_an_attribute_never_the_message(self):
        """execution.py:651 formats str(ex) into the UI error; numbers belong
        on an attribute where tools read them and users are not spammed."""
        we = WorkerError("CUDA out of memory", error_kind="oom",
                         oom_stats={"allocated": 123456789, "reserved": 2,
                                    "largest_free_block": 3})
        out = translate_error(we, registry=REGISTRY)
        assert out.comfy_env_worker == {"allocated": 123456789, "reserved": 2,
                                        "largest_free_block": 3}
        assert "123456789" not in str(out)

    def test_interrupt_kind_resolves_via_its_own_entry(self):
        we = WorkerError("Processing interrupted by user",
                         error_kind="interrupt")
        assert isinstance(translate_error(we, registry=REGISTRY), FakeInterrupt)

    def test_default_registry_is_honest_about_torch(self):
        """With torch absent the default registry must pass through (never
        synthesize); with torch present it must produce the REAL class that
        ComfyUI's own is_oom accepts."""
        we = WorkerError("CUDA out of memory", error_kind="oom")
        out = translate_error(we)
        try:
            import torch
        except ImportError:
            assert out is we
        else:
            assert isinstance(out, torch.cuda.OutOfMemoryError)
            assert out.__cause__ is we

    def test_legacy_constructor_signature_still_works(self):
        """Old call sites build WorkerError(message, traceback=...). Breaking
        that signature would break every existing raise site at once."""
        we = WorkerError("boom", traceback="tb")
        assert we.error_kind is None and we.oom_stats is None


class TestSeamGuards:
    def test_default_oom_resolver_names_the_real_torch_class(self):
        """Catches: the resolver quietly switched to a comfy-env subclass,
        which would leak into the frontend's exception_type via
        full_type_name at execution.py:630."""
        tree = ast.parse(ERRORS.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_resolve_oom")
        assert "torch.cuda.OutOfMemoryError" in ast.unparse(fn)

    def test_no_verdict_from_message_text_in_errors_module(self):
        """Bans any comparison against a string literal in errors.py: the
        verdict comes from error_kind alone. (Building the message WITH
        str(exc) is fine; deciding FROM it is the defect.)"""
        tree = ast.parse(ERRORS.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            operands = [node.left, *node.comparators]
            for op in operands:
                assert not (isinstance(op, ast.Constant)
                            and isinstance(op.value, str)), (
                    f"string comparison at errors.py:{node.lineno}; the "
                    f"verdict must come from error_kind, never message text")

    def test_translation_sits_outside_the_worker_teardown_clause(self):
        """The load-bearing placement: torch.OutOfMemoryError IS a
        RuntimeError, and metadata.py's except (RuntimeError, ConnectionError)
        removes the worker. translate_error must be called from an except
        WorkerError handler of that same try, never from the try body, or
        every worker OOM tears the worker down."""
        tree = ast.parse(METADATA.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_call_in_worker")
        tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        hosting = [t for t in tries
                   if any("WorkerError" in ast.unparse(h.type)
                          for h in t.handlers if h.type is not None)]
        assert hosting, "_call_in_worker no longer handles WorkerError"
        t = hosting[0]
        handler_types = [ast.unparse(h.type) for h in t.handlers if h.type]
        assert any("RuntimeError" in ht and "ConnectionError" in ht
                   for ht in handler_types), (
            "the WorkerError handler moved away from the teardown try; the "
            "ordering guarantee is gone")
        body_src = "\n".join(ast.unparse(s) for s in t.body)
        assert "translate_error" not in body_src, (
            "translate_error is called inside the try body: the translated "
            "OOM is a RuntimeError and will tear the worker down")
        we_handler = next(h for h in t.handlers
                          if h.type is not None
                          and "WorkerError" in ast.unparse(h.type))
        assert "translate_error" in "\n".join(ast.unparse(s)
                                              for s in we_handler.body)

    def test_both_worker_error_frames_carry_the_typed_verdict(self):
        """Catches: a new error frame added (or one refactored) without the
        error_kind stamp, silently reverting that path to untyped."""
        src = WORKER.read_text(encoding="utf-8")
        assert src.count("_frame.update(_error_kind_fields(e))") >= 2, (
            "fewer than two error frames stamp error_kind; the echo or call "
            "path lost its typed verdict")

    def test_worker_verdict_is_typed_not_text(self):
        """_error_kind_fields must decide via is_oom and isinstance. Its only
        string constants are the closed vocabulary and dict keys, so a
        'contains out of memory' check cannot hide in it."""
        tree = ast.parse(WORKER.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_error_kind_fields")
        src = ast.unparse(fn)
        assert "is_oom" in src and "isinstance" in src
        assert ".lower()" not in src and "in str(" not in src

    def test_worker_reads_the_typed_interrupt_field(self):
        """_call_parent must raise the interrupt from the typed field; the
        text match survives only inside the progress hook as the old-parent
        fallback."""
        src = WORKER.read_text(encoding="utf-8")
        assert 'response.get("error_kind") == "interrupt"' in src, (
            "_call_parent no longer reads the typed interrupt field")

    def test_parent_interrupt_is_a_typed_exception_with_legacy_text(self):
        """pool.py must raise InterruptRequested (typed, for new workers) and
        keep 'interrupted' in the message (old workers still text-match)."""
        src = POOL.read_text(encoding="utf-8")
        assert 'raise InterruptRequested("Processing interrupted by user")' in src
        assert issubclass(InterruptRequested, RuntimeError), (
            "InterruptRequested must stay a RuntimeError so old workers' "
            "except RuntimeError fallback still catches the callback error")
