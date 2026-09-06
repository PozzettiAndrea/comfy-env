"""Contract: comfy-env can say which ComfyUI memory manager a process resolved to.

A worker never runs ``main.py``, so ``comfy.memory_management.aimdo_enabled``
stays at its module default and the worker resolves to the legacy ledger while
the host is normally on aimdo. Nothing reported that before these tests existed,
and the mismatch is not even stable across installs: whether a pack is isolated
at all is a per-pack decision, so two packs in one run can resolve differently.

These tests deliberately avoid importing ComfyUI. The reporting path has to work
on a machine where ComfyUI is absent, because "which manager am I on" is a
question people ask precisely when the stack is not behaving.
"""

import sys
import textwrap
import types
from pathlib import Path

import pytest

from comfy_env.memory_manager import (
    AIMDO,
    ENABLE_ENV_VAR,
    LEDGER,
    describe,
    maybe_enable_aimdo,
    release_node_boundary,
    summary_line,
)
from comfy_env.packages.toml_generator import read_host_pin


@pytest.fixture
def fake_comfy(monkeypatch):
    """Install a minimal fake ``comfy.memory_management`` and hand back the module."""

    def _install(aimdo_enabled: bool):
        comfy = types.ModuleType("comfy")
        mm = types.ModuleType("comfy.memory_management")
        mm.aimdo_enabled = aimdo_enabled
        comfy.memory_management = mm
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.memory_management", mm)
        return mm

    return _install


def test_describe_reports_ledger_when_comfy_is_absent(monkeypatch):
    """No ComfyUI is an answer, not an exception."""
    monkeypatch.setitem(sys.modules, "comfy.memory_management", None)
    info = describe()
    assert info["manager"] == LEDGER
    assert "reason" in info


def test_describe_reports_ledger_when_aimdo_never_initialised(fake_comfy):
    """The worker's real case: importable or not, nothing ever set the flag."""
    fake_comfy(aimdo_enabled=False)
    info = describe()
    assert info["manager"] == LEDGER
    assert info["reason"]


def test_describe_reports_aimdo_when_the_flag_is_set(fake_comfy):
    fake_comfy(aimdo_enabled=True)
    info = describe()
    assert info["manager"] == AIMDO
    assert info["reason"] == "aimdo_enabled is True"


def test_summary_line_is_greppable(fake_comfy):
    fake_comfy(aimdo_enabled=False)
    line = summary_line("[worker] ")
    assert line.startswith("[worker] memory manager:")
    assert "ledger" in line


def test_no_parent_signal_means_ledger(monkeypatch):
    """The worker FOLLOWS the host. Without the parent's signal it stays on the
    ledger, which is the behaviour before this existed."""
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    messages = []
    assert maybe_enable_aimdo(log=messages.append) is False
    assert any("no parent signal" in m for m in messages)


def test_parent_signal_off_means_ledger(monkeypatch):
    """A host on the ledger chose it; the worker must not second-guess."""
    monkeypatch.setenv(ENABLE_ENV_VAR, "0")
    messages = []
    assert maybe_enable_aimdo(log=messages.append) is False
    assert any("disabled by" in m for m in messages)


def test_parent_signal_on_proceeds_to_device_check(monkeypatch):
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setattr("comfy_env.memory_manager._cuda_devices", lambda: [])
    messages = []
    assert maybe_enable_aimdo(log=messages.append) is False
    assert any("no CUDA device" in m for m in messages)


def test_enable_refuses_without_a_cuda_device(monkeypatch):
    """Gated on real devices, not on COMFY_CPU, which only describes the parent.

    aimdo has no CPU path at all: ``_vbar_get`` returns None for a CPU load
    device and ``partially_unload`` asserts a non-CPU one.
    """
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setattr("comfy_env.memory_manager._cuda_devices", lambda: [])
    messages = []
    assert maybe_enable_aimdo(log=messages.append) is False
    assert any("no CUDA device" in m for m in messages)


def test_cast_epoch_boundary_resets_per_prompt_non_aimdo(
        fake_comfy, monkeypatch):
    """The old form was a full noop without aimdo, which let a non-aimdo
    worker on the lowvram cast path ratchet STREAM_CAST_BUFFERS to
    NUM_STREAMS x its largest-ever layer for the life of the process
    (measured: 2 x 512 MiB held through unloads). Now the cast rung runs on
    prompt-epoch changes (and per node when no token arrives, the safe
    default), while the prefetch/vbar rungs stay aimdo only."""
    import sys
    import types
    from comfy_env import memory_manager
    fake_comfy(aimdo_enabled=False)
    calls = []
    mmm = types.ModuleType("comfy.model_management")
    mmm.reset_cast_buffers = lambda: calls.append("reset")
    monkeypatch.setitem(sys.modules, "comfy.model_management", mmm)
    monkeypatch.setattr(memory_manager, "_CAST_EPOCH", [object()])
    memory_manager.cast_epoch_boundary(1)     # first sight of epoch 1: reset
    memory_manager.cast_epoch_boundary(1)     # same epoch: no churn
    memory_manager.cast_epoch_boundary(2)     # epoch changed: reset
    memory_manager.cast_epoch_boundary(None)  # no token: per-call safe default
    assert calls == ["reset", "reset", "reset"]
    # and the per-node finally hook stays aimdo-gated (no double churn)
    release_node_boundary()
    assert calls == ["reset", "reset", "reset"]


def test_release_node_boundary_never_raises_without_aimdo(fake_comfy):
    """Must never raise: a failed release cannot fail the node that
    succeeded (here comfy.model_management is absent entirely)."""
    fake_comfy(aimdo_enabled=False)
    release_node_boundary()  # no exception is the assertion


def test_release_node_boundary_survives_a_broken_comfy(fake_comfy):
    mm = fake_comfy(aimdo_enabled=True)
    assert mm.aimdo_enabled is True
    release_node_boundary()  # comfy.model_management is absent; must not raise


class TestReadHostPin:
    """comfy-env replicates the host's pin, it never authors one.

    comfy-aimdo ships roughly weekly and ComfyUI pins it exactly, so a literal
    in this repository would be stale within the month.
    """

    def _tree(self, tmp_path: Path, body: str) -> Path:
        (tmp_path / "requirements.txt").write_text(textwrap.dedent(body), encoding="utf-8")
        return tmp_path

    def test_reads_an_exact_pin(self, tmp_path):
        root = self._tree(tmp_path, """\
            torch
            comfy-aimdo==0.4.15
            numpy
        """)
        assert read_host_pin(root, "comfy-aimdo") == "0.4.15"

    def test_accepts_the_underscore_spelling(self, tmp_path):
        root = self._tree(tmp_path, "comfy_aimdo==0.4.13\n")
        assert read_host_pin(root, "comfy-aimdo") == "0.4.13"

    def test_returns_none_when_unpinned(self, tmp_path):
        root = self._tree(tmp_path, "comfy-aimdo>=0.4\n")
        assert read_host_pin(root, "comfy-aimdo") is None

    def test_returns_none_when_absent(self, tmp_path):
        root = self._tree(tmp_path, "torch\n")
        assert read_host_pin(root, "comfy-aimdo") is None

    def test_returns_none_without_a_requirements_file(self, tmp_path):
        assert read_host_pin(tmp_path, "comfy-aimdo") is None

    def test_returns_none_for_no_directory(self):
        assert read_host_pin(None, "comfy-aimdo") is None


class TestHostDerivedSubstitution:
    """A pack's unpinned comfy-aimdo must not be allowed to drift off the host.

    Observed in production before this existed: a worker env resolved
    ``comfy-aimdo = "*"`` to 0.4.14 while its host ComfyUI pinned 0.4.13. Parent
    and worker then hold different builds of a native wheel that both touch the
    same device.
    """

    def _cfg(self, tmp_path: Path, declared, key="comfy-aimdo"):
        from comfy_env.config import load_config

        body = '[pypi-dependencies]\npillow = ">=9.0.0"\n'
        if declared is not None:
            body += f'"{key}" = "{declared}"\n'
        (tmp_path / "comfy-env.toml").write_text(body, encoding="utf-8")
        return load_config(tmp_path / "comfy-env.toml")

    def _host(self, tmp_path: Path, pin="0.4.13"):
        root = tmp_path / "ComfyUI"
        root.mkdir(exist_ok=True)
        (root / "requirements.txt").write_text(f"torch\ncomfy-aimdo=={pin}\n", encoding="utf-8")
        return root

    CUDA_INDEX = "https://download.pytorch.org/whl/cu128"

    def _build(self, cfg, host, torch_index=CUDA_INDEX, **kw):
        from comfy_env.packages.toml_generator import _build_node_feature

        return _build_node_feature(
            cfg, "pack", "3.13", torch_pin="==2.8.*", torch_index=torch_index,
            glibc_version=None, log=lambda m: None, comfyui_dir=host, **kw,
        )["pypi-dependencies"]

    def test_a_wildcard_is_replaced_by_the_host_pin(self, tmp_path):
        cfg = self._cfg(tmp_path, "*")
        pypi = self._build(cfg, self._host(tmp_path))
        assert pypi["comfy-aimdo"] == "==0.4.13"

    def test_a_pack_pin_does_not_stand_when_the_host_has_no_pin(self, tmp_path):
        """These versions are not a pack's to choose, and the rule has to hold
        when the host's pin is unreadable too, which is where nobody would
        notice it was not being applied. Catches the earlier invariant, "never
        remove without a replacement", which was right for a pack's own
        dependencies and wrong for the two comfy-env replicates: a worker
        pages the same card as the host and must page under the same policy.
        With no substitute the solver picks, which is what a pack that never
        declared it already gets."""
        cfg = self._cfg(tmp_path, "*")
        root = tmp_path / "ComfyUI"
        root.mkdir(exist_ok=True)
        (root / "requirements.txt").write_text("torch\n", encoding="utf-8")
        pypi = self._build(cfg, root)
        assert "comfy-aimdo" not in pypi

    def test_a_conflicting_explicit_pin_is_an_error(self, tmp_path):
        """A wildcard is boilerplate. An explicit disagreement is a statement."""
        cfg = self._cfg(tmp_path, "==0.4.16")
        with pytest.raises(ValueError, match="host ComfyUI pins"):
            self._build(cfg, self._host(tmp_path))

    def test_a_cuda_pack_declaring_nothing_still_gets_the_host_pin(self, tmp_path):
        """Absence is the expensive mistake: a worker cannot be made aimdo
        transparent later without the wheel, and it is inert until initialised."""
        cfg = self._cfg(tmp_path, None)
        pypi = self._build(cfg, self._host(tmp_path))
        assert pypi["comfy-aimdo"] == "==0.4.13"

    def test_a_cpu_env_does_not_get_it(self, tmp_path):
        """aimdo has no CPU path at all, so there it would be dead weight."""
        cfg = self._cfg(tmp_path, None)
        pypi = self._build(cfg, self._host(tmp_path), torch_index=None)
        assert "comfy-aimdo" not in pypi

    def test_case_and_separator_variants_are_matched(self, tmp_path):
        """PEP 503: missing one would leave the pack's key beside our substitute
        as two spellings of one distribution with conflicting specs."""
        for spelling in ("Comfy-AIMDO", "comfy_aimdo", "comfy.aimdo"):
            cfg = self._cfg(tmp_path, "*", key=spelling)
            pypi = self._build(cfg, self._host(tmp_path))
            keys = [k for k in pypi if "aimdo" in k.lower()]
            assert keys == ["comfy-aimdo"], f"{spelling} left {keys}"
            assert pypi["comfy-aimdo"] == "==0.4.13"

    def test_a_conflicting_table_form_pin_also_raises(self, tmp_path):
        """The table form is a legal pixi spelling and must not slip past."""
        (tmp_path / "comfy-env.toml").write_text(
            '[pypi-dependencies]\ncomfy-aimdo = {version = "==0.4.16"}\n', encoding="utf-8"
        )
        from comfy_env.config import load_config

        with pytest.raises(ValueError, match="host ComfyUI pins"):
            self._build(load_config(tmp_path / "comfy-env.toml"), self._host(tmp_path))

    def test_disabled_leaves_everything_alone(self, tmp_path):
        """Opting out of replication has to mean opting out. Catches a strip
        that runs regardless of the switch, which would take the pack's
        declaration away and give nothing back."""
        cfg = self._cfg(tmp_path, "*")
        pypi = self._build(cfg, self._host(tmp_path), host_derived=False)
        assert pypi["comfy-aimdo"] == "*"


    def test_target_table_wildcard_is_substituted(self, tmp_path):
        """A target table wins over the feature table on its own platform, so an
        unpinned aimdo left there would silently beat the host pin."""
        from comfy_env.config import load_config
        from comfy_env.detection import get_pixi_platform

        plat = get_pixi_platform()
        (tmp_path / "comfy-env.toml").write_text(
            f'[target.{plat}.pypi-dependencies]\ncomfy-aimdo = "*"\n',
            encoding="utf-8",
        )
        cfg = load_config(tmp_path / "comfy-env.toml")
        pypi = self._build(cfg, self._host(tmp_path))
        assert pypi["comfy-aimdo"] == "==0.4.13"

    def test_target_table_conflicting_pin_raises(self, tmp_path):
        from comfy_env.config import load_config
        from comfy_env.detection import get_pixi_platform

        plat = get_pixi_platform()
        (tmp_path / "comfy-env.toml").write_text(
            f'[target.{plat}.pypi-dependencies]\ncomfy-aimdo = "==0.4.16"\n',
            encoding="utf-8",
        )
        cfg = load_config(tmp_path / "comfy-env.toml")
        with pytest.raises(ValueError, match="host ComfyUI pins"):
            self._build(cfg, self._host(tmp_path))


class TestParentCapturesNewModels:
    """The parent must harvest `_new_models` from EVERY response frame, error
    included: a model loaded during a call that raised is GPU resident, and
    losing the frame means the host can never see or evict it."""

    def test_capture_precedes_any_status_check(self):
        """Source level: in _send_request, the `_new_models` harvest must come
        before the first inspection of response status."""
        import ast as _ast
        from pathlib import Path

        src_path = (
            Path(__file__).resolve().parents[1]
            / "src" / "comfy_env" / "isolation" / "workers" / "subprocess.py"
        )
        tree = _ast.parse(src_path.read_text(encoding="utf-8"))
        fn = next(
            n for n in _ast.walk(tree)
            if isinstance(n, _ast.FunctionDef) and n.name == "_send_request"
        )
        harvest_line = None
        status_line = None
        for node in _ast.walk(fn):
            if (harvest_line is None and isinstance(node, _ast.Constant)
                    and node.value == "_new_models"):
                harvest_line = node.lineno
            if (status_line is None and isinstance(node, _ast.Constant)
                    and node.value == "status"):
                status_line = node.lineno
        assert harvest_line is not None, "_new_models harvest not found"
        if status_line is not None:
            assert harvest_line < status_line, (
                "the _new_models harvest no longer precedes the status check; "
                "models loaded during a raising call would be lost again"
            )


class TestApplyReserveBootstrap:
    """The budget owner's advance payment. Zero is a VALUE here (upstream
    honors --reserve-vram 0 and the reply assigns it verbatim), deliberately
    different from apply_pin_budget's disabled sentinel; a harmonizing
    refactor of the two appliers is the named enemy."""

    def _mm(self, monkeypatch, preset=400 * 1024 * 1024):
        import sys
        import types
        mod = types.ModuleType("comfy.model_management")
        mod.EXTRA_RESERVED_VRAM = preset
        monkeypatch.setitem(sys.modules, "comfy.model_management", mod)
        return mod

    def test_absent_is_a_noop_byte_identical(self, monkeypatch):
        """Catches: a default of 0 silently zeroing every worker's margin
        when the pool does not inject."""
        from comfy_env.memory_manager import apply_reserve_bootstrap
        mod = self._mm(monkeypatch)
        assert apply_reserve_bootstrap(None) is False
        assert mod.EXTRA_RESERVED_VRAM == 400 * 1024 * 1024

    def test_zero_is_a_value_not_absence(self, monkeypatch):
        """Catches: an `if value:` truthiness guard, whose failure mode is a
        --reserve-vram 0 host whose workers keep a 400 MiB phantom margin
        until first load and then drop to 0, the exact window inconsistency
        this fix removes."""
        from comfy_env.memory_manager import apply_reserve_bootstrap
        mod = self._mm(monkeypatch)
        assert apply_reserve_bootstrap("0") is True
        assert mod.EXTRA_RESERVED_VRAM == 0

    def test_sets_exactly_the_given_bytes_never_adds(self, monkeypatch):
        """Catches: += accumulation, under which the worker believes margin
        equal to bootstrap plus reply and ratchets every load."""
        from comfy_env.memory_manager import apply_reserve_bootstrap
        mod = self._mm(monkeypatch)
        two_gib = 2 * 1024 ** 3
        apply_reserve_bootstrap(str(two_gib))
        assert mod.EXTRA_RESERVED_VRAM == two_gib
        assert mod.EXTRA_RESERVED_VRAM != two_gib + 400 * 1024 * 1024

    def test_negative_crosses_verbatim_with_a_warn(self, monkeypatch):
        """The flag is an unbounded float upstream, so negatives are
        reachable from the CLI; both settlement legs forward them verbatim,
        and clamping only the advance would disagree with its own
        settlement."""
        from comfy_env.memory_manager import apply_reserve_bootstrap
        mod = self._mm(monkeypatch)
        msgs = []
        assert apply_reserve_bootstrap("-1024", log=msgs.append) is True
        assert mod.EXTRA_RESERVED_VRAM == -1024
        assert any("negative" in m for m in msgs)

    def test_garbage_warns_and_noops(self, monkeypatch):
        """Catches: an uncaught ValueError crashing worker startup before the
        ready frame, turning a config typo into a dead pool."""
        from comfy_env.memory_manager import apply_reserve_bootstrap
        mod = self._mm(monkeypatch)
        msgs = []
        assert apply_reserve_bootstrap("8GB", log=msgs.append) is False
        assert mod.EXTRA_RESERVED_VRAM == 400 * 1024 * 1024
        assert msgs

    def test_no_comfy_is_a_noop(self, monkeypatch):
        import sys
        from comfy_env.memory_manager import apply_reserve_bootstrap
        monkeypatch.setitem(sys.modules, "comfy.model_management", None)
        assert apply_reserve_bootstrap("1024") is False


class TestAimdoDeviceArgs:
    def test_a_modern_wheel_gets_tuples(self):
        from comfy_env.memory_manager import aimdo_device_args
        args, dropped = aimdo_device_args([0, 1], 512, True)
        assert args == [(0, 512), (1, 512)] and dropped is None

    def test_a_pre_0_4_10_wheel_gets_bare_ints_and_says_what_it_lost(self):
        """Catches the shipped bug: comfy-env passed (index, headroom) tuples
        unconditionally, and comfy-aimdo before 0.4.10 does int(device_id) on
        each, so the call raised TypeError and the worker fell to the legacy
        ledger. ComfyUI's own main.py has this fallback; comfy-env did not."""
        from comfy_env.memory_manager import aimdo_device_args
        args, dropped = aimdo_device_args([0], 512, False)
        assert args == [0]
        assert dropped and "headroom" in dropped

    def test_a_pre_0_4_10_wheel_with_no_headroom_loses_nothing(self):
        """Catches: reporting a dropped policy that was never requested, which
        would make an ordinary old-wheel worker look misconfigured."""
        from comfy_env.memory_manager import aimdo_device_args
        assert aimdo_device_args([0], 0, False) == ([0], None)

    def test_generator_is_not_returned(self):
        """Catches: handing init_devices a generator, which cannot be
        inspected, logged, or retried after a TypeError."""
        from comfy_env.memory_manager import aimdo_device_args
        args, _ = aimdo_device_args([0, 1], 0, True)
        assert isinstance(args, list) and len(args) == 2


class TestAimdoPolicyDrops:
    def test_every_init_ladder_rung_logs_what_it_dropped(self):
        """Catches: a silent TypeError ladder. A worker paging under a
        different policy than its host is the divergence this seam exists to
        prevent, and each fallback rung drops one."""
        import ast
        src = Path("src/comfy_env/memory_manager.py").read_text(encoding="utf-8")
        fn = next(n for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.FunctionDef) and n.name == "maybe_enable_aimdo")
        for node in ast.walk(fn):
            if not isinstance(node, ast.Try):
                continue
            for handler in node.handlers:
                if handler.type is None or "TypeError" not in ast.unparse(handler.type):
                    continue
                body = ast.unparse(handler)
                if "control.init(" not in body:
                    continue
                assert "_log(" in body, (
                    "a TypeError rung around control.init drops a policy "
                    "without saying so")


class TestTupleDeviceDetection:
    """The one thing about an installed comfy-aimdo that changes an argument
    comfy-env builds. A wrong answer here hands bare ints to a wheel that
    wants pairs, or the reverse, and either way the worker lands on the
    legacy ledger with the per device headroom lost."""

    def test_an_unreadable_shape_assumes_the_modern_form(self):
        """Catches: falling back to the old form. Unreadable is not evidence
        of absence, and the pair form has been the only one since 0.4.10, so
        guessing old drops the headroom silently for the process's life."""
        from comfy_env.memory_manager import aimdo_takes_tuple_devices
        control = type("C", (), {"init_devices": len})   # builtin: no source
        assert aimdo_takes_tuple_devices(control) is True

    def test_a_genuine_old_wheel_is_still_detected(self):
        """Catches: the fallback becoming a rubber stamp. A pre 0.4.10 wheel
        takes bare ints and must be read as such, or init_devices raises
        TypeError inside a comprehension."""
        from comfy_env.memory_manager import aimdo_takes_tuple_devices

        def init_devices(device_ids):
            return True
        control = type("C", (), {"init_devices": staticmethod(init_devices)})
        assert aimdo_takes_tuple_devices(control) is False

    def test_the_annotation_is_read_before_the_source(self):
        """Catches: reading source text as the only evidence. An annotated
        signature is the durable answer and needs no source at all."""
        from typing import List, Tuple
        from comfy_env.memory_manager import aimdo_takes_tuple_devices

        def init_devices(device_ids: List[Tuple[int, int]]):
            return True
        control = type("C", (), {"init_devices": staticmethod(init_devices)})
        assert aimdo_takes_tuple_devices(control) is True


class TestNoLevelRefusal:
    """The ladder that used to live here refused to page across a protocol
    difference. It went because refusing drops the worker to the legacy
    ledger, which on a card shared with the host is worse than the mismatch
    it was avoiding, and because the parent's level shaped no argument."""

    def test_a_version_difference_never_refuses_aimdo(self):
        """Catches: any re-introduction of a skew refusal. A worker on a
        neighbouring patch version must still page."""
        import ast
        import inspect
        from comfy_env import memory_manager
        src = inspect.getsource(memory_manager.maybe_enable_aimdo)
        tree = ast.parse(src.lstrip())
        raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
        for node in raises:
            text = ast.unparse(node)
            assert "skew" not in text.lower(), text

    def test_the_parent_no_longer_exports_a_level(self):
        """Catches: the export coming back. Nothing reads it, and a stale one
        would refuse a worker for a difference that does not matter."""
        import pathlib
        src = pathlib.Path(
            "src/comfy_env/isolation/workers/subprocess.py"
        ).read_text(encoding="utf-8")
        assert "COMFY_ENV_AIMDO_LEVEL" not in src
        assert "aimdo_installed_level" not in src


class TestKitchenVersionReport:
    """comfy-kitchen gets a version line and nothing else, and the reason is
    not modesty: it fails loudly by itself. ComfyUI calls
    int8_attention_is_available() at module scope, so drift is an
    AttributeError at worker start with a file, a line and a name. aimdo
    fails silently instead, which is what its machinery is for."""

    def test_describe_carries_the_kitchen_version(self):
        """Catches: reporting aimdo and leaving the other replicated pin
        invisible, so a reader of a crash cannot tell which two versions were
        in play."""
        from comfy_env.memory_manager import describe
        assert "kitchen_version" in describe()

    def test_the_summary_line_names_it_when_present(self, monkeypatch):
        from comfy_env import memory_manager as mm
        monkeypatch.setattr(mm, "_KITCHEN_VERSION_READ", True)
        monkeypatch.setattr(mm, "_KITCHEN_VERSION", "0.2.31")
        assert "kitchen 0.2.31" in mm.summary_line()

    def test_an_absent_kitchen_adds_nothing_to_the_line(self, monkeypatch):
        """Catches: printing 'kitchen none' on every CPU worker, where its
        absence is correct and expected."""
        from comfy_env import memory_manager as mm
        monkeypatch.setattr(mm, "_KITCHEN_VERSION_READ", True)
        monkeypatch.setattr(mm, "_KITCHEN_VERSION", None)
        assert "kitchen" not in mm.summary_line()

    def test_the_lookup_is_cached_like_aimdos(self, monkeypatch):
        """Catches: an importlib.metadata walk on a path describe() runs per
        request."""
        from comfy_env import memory_manager as mm
        monkeypatch.setattr(mm, "_KITCHEN_VERSION_READ", False)
        monkeypatch.setattr(mm, "_KITCHEN_VERSION", None)
        calls = []
        first = mm.kitchen_version()
        monkeypatch.setattr(mm, "_KITCHEN_VERSION", "sentinel")
        assert mm.kitchen_version() == "sentinel"
        assert calls == []
        assert first is None or isinstance(first, str)
