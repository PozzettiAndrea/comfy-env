"""Contract: generated per-env manifests are self-contained and torch-pinned."""

from comfy_env.config import parse_config
from comfy_env.packages.toml_generator import (
    build_env_toml,
)


def test_build_env_toml_shape_and_torch_pin():
    cfg = parse_config({"dependencies": {"ffmpeg": "*"}})
    manifest = build_env_toml(
        "sam3-nodes", cfg,
        torch_index="https://download.pytorch.org/whl/cpu",
        bootstrap_python="3.12",
        torch_pin="==2.8.0",
        log=lambda m: None,
    )
    # Feature must be "node" (pixi reserves "default" as a feature name);
    # the single environment must be "default" with no-default-feature.
    assert set(manifest["feature"].keys()) == {"node"}
    envs = manifest["environments"]
    assert envs["default"]["features"] == ["node"]
    assert envs["default"]["no-default-feature"] is True
    # The workspace-wide torch pin is replicated into the env's feature.
    assert "==2.8.0" in str(manifest["feature"]["node"])
    # Conda passthrough deps landed in the feature.
    assert "ffmpeg" in str(manifest["feature"]["node"])


class TestHostRequiredPackages:
    """The host's ComfyUI imports these unguarded, so they are not the pack's
    to declare. A worker missing one cannot import comfy at all."""

    @staticmethod
    def _pypi(monkeypatch, torch_index):
        from comfy_env.config import parse_config
        from comfy_env.packages import toml_generator as tg

        monkeypatch.setattr(
            tg, "read_host_pin",
            lambda d, pkg: {"comfy-aimdo": "0.4.13",
                            "comfy-kitchen": "0.2.31"}.get(pkg))
        manifest = tg.build_env_toml(
            "somepack-nodes", parse_config({}),
            torch_index=torch_index,
            bootstrap_python="3.12",
            torch_pin="==2.8.0",
            log=lambda m: None,
        )
        return manifest["feature"]["node"].get("pypi-dependencies", {})

    def test_kitchen_is_injected_even_though_no_pack_declares_it(self, monkeypatch):
        """Catches the shipped behaviour: host-derived packages were only
        substituted where a pack already declared them, so a pack that never
        mentioned comfy-kitchen got an env that raises ModuleNotFoundError on
        `import comfy.model_patcher`. Four of nineteen envs on the development
        machine were in exactly that state."""
        pypi = self._pypi(monkeypatch, "https://download.pytorch.org/whl/cu128")
        assert pypi.get("comfy-kitchen") == "==0.2.31"

    def test_aimdo_is_injected_on_a_cuda_stack(self, monkeypatch):
        pypi = self._pypi(monkeypatch, "https://download.pytorch.org/whl/cu128")
        assert pypi.get("comfy-aimdo") == "==0.4.13"

    def test_kitchen_reaches_cpu_envs_but_aimdo_does_not(self, monkeypatch):
        """Catches: applying one platform rule to both. comfy-aimdo has no CPU
        path and would be dead weight, but comfy-kitchen is imported by
        comfy/ldm/modules/attention.py on any stack, so a CPU worker needs it
        just as much."""
        pypi = self._pypi(monkeypatch, "https://download.pytorch.org/whl/cpu")
        assert pypi.get("comfy-kitchen") == "==0.2.31"
        assert "comfy-aimdo" not in pypi

    def test_nothing_is_injected_without_torch(self, monkeypatch):
        """Catches: injecting into a torch-less env, which cannot import comfy
        anyway and would gain two native wheels for nothing."""
        from comfy_env.config import parse_config
        from comfy_env.packages import toml_generator as tg
        monkeypatch.setattr(tg, "read_host_pin", lambda d, pkg: "1.0")
        manifest = tg.build_env_toml(
            "somepack-nodes", parse_config({}), torch_index=None,
            bootstrap_python="3.12", torch_pin=None, log=lambda m: None)
        pypi = manifest["feature"]["node"].get("pypi-dependencies", {})
        assert "comfy-aimdo" not in pypi and "comfy-kitchen" not in pypi


class TestPacksDoNotPinHostDerived:
    """The versions comfy-env replicates are not a pack's to choose: the
    worker pages the same card as the host and must page under the same
    policy. The rule has to hold whether or not we can read the host's pin,
    because the case where we cannot is where nobody would notice."""

    def test_a_pack_pin_is_dropped_when_there_is_no_host_pin(self):
        """Catches the hole this closes: with no substitute the pack's own
        declaration used to stand, so a pack could pin the pager exactly
        when the host's requirements were unreadable or absent."""
        from comfy_env.packages.toml_generator import _replace_host_derived
        from comfy_env.packages.toml_generator import _HOST_DERIVED_PKGS
        node = {"comfy-aimdo": "==0.4.9", "numpy": "*"}
        out = _replace_host_derived(node, {}, "pack", log=lambda *_: None,
                                    derived=_HOST_DERIVED_PKGS)
        assert "comfy-aimdo" not in node
        assert out == {}
        assert node == {"numpy": "*"}

    def test_the_host_pin_still_substitutes_when_there_is_one(self):
        from comfy_env.packages.toml_generator import _replace_host_derived
        from comfy_env.packages.toml_generator import _HOST_DERIVED_PKGS
        node = {"comfy-aimdo": "*"}
        out = _replace_host_derived(node, {"comfy-aimdo": "==0.4.13"}, "pack",
                                    log=lambda *_: None,
                                    derived=_HOST_DERIVED_PKGS)
        assert out == {"comfy-aimdo": "==0.4.13"}

    def test_the_opt_out_leaves_a_pack_declaration_standing(self):
        """Catches a strip that ignores the switch. host_derived=False passes
        no derived names, and then this must touch nothing."""
        from comfy_env.packages.toml_generator import _replace_host_derived
        node = {"comfy-aimdo": "==0.4.9"}
        _replace_host_derived(node, {}, "pack", log=lambda *_: None, derived=())
        assert node == {"comfy-aimdo": "==0.4.9"}

    def test_a_disagreeing_exact_pin_still_raises(self):
        """Catches: softening the rule into a silent override. An exact pin
        is a deliberate statement and the author should hear that the seam
        cannot honour it."""
        import pytest
        from comfy_env.packages.toml_generator import _replace_host_derived
        with pytest.raises(ValueError):
            _replace_host_derived({"comfy-aimdo": "==0.4.9"},
                                  {"comfy-aimdo": "==0.4.13"}, "pack",
                                  log=lambda *_: None)

    def test_packages_comfy_env_does_not_own_are_untouched(self):
        """Catches: dropping a pack's own dependencies. Only the host-derived
        names are ours to overrule."""
        from comfy_env.packages.toml_generator import _replace_host_derived
        from comfy_env.packages.toml_generator import _HOST_DERIVED_PKGS
        node = {"trimesh": "==4.0.0"}
        _replace_host_derived(node, {}, "pack", log=lambda *_: None,
                              derived=_HOST_DERIVED_PKGS)
        assert node == {"trimesh": "==4.0.0"}


class TestFastKeyTracksTheHostPin:
    """The level-1 gate decides "nothing changed, skip". If it cannot see the
    host's pins, a ComfyUI update never reaches an existing env."""

    def test_the_key_moves_when_the_host_pin_moves(self, monkeypatch, tmp_path):
        """Catches THE reason "replicate the host's pin" was only true on the
        first install: the gate short-circuits before the manifest is built,
        so an env kept its old comfy-aimdo for the life of the install and
        the only sign was a version-skew note at runtime."""
        from comfy_env.install import workspace
        from comfy_env.packages import toml_generator

        cf = tmp_path / "comfy-env-root.toml"
        cf.write_text("")
        pins = {"comfy-aimdo": "0.4.13"}
        monkeypatch.setattr(toml_generator, "read_host_pin",
                            lambda d, pkg: pins.get(pkg))
        monkeypatch.setattr(workspace, "_fast_key", workspace._fast_key)
        first = workspace._fast_key(cf, [], tmp_path)
        pins["comfy-aimdo"] = "0.4.14"
        second = workspace._fast_key(cf, [], tmp_path)
        assert first != second

    def test_an_unchanged_pin_keeps_the_key_stable(self):
        """Catches: hashing something that moves on its own, which would
        re-derive every env on every run."""
        from pathlib import Path
        from comfy_env.install import workspace
        cf = Path("/nonexistent/comfy-env-root.toml")
        assert workspace._fast_key(cf, [], None) == workspace._fast_key(cf, [], None)
