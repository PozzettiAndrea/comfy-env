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
