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
