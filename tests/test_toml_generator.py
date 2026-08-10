"""Contract: generated per-env manifests are self-contained and torch-pinned."""

from comfy_env.config import parse_config
from comfy_env.packages.toml_generator import (
    build_env_toml,
    deep_merge,
    parse_requirement_line,
)


def test_parse_requirement_line_docstring_cases():
    assert parse_requirement_line("torch==2.8.0") == ("torch", "==2.8.0")
    assert parse_requirement_line("numpy>=1.25.0") == ("numpy", ">=1.25.0")
    assert parse_requirement_line("Pillow") == ("Pillow", "*")
    assert parse_requirement_line("trimesh[easy]>=4.0.0") == (
        "trimesh", {"version": ">=4.0.0", "extras": ["easy"]})
    assert parse_requirement_line("# comment") is None
    assert parse_requirement_line("") is None
    assert parse_requirement_line("-r other.txt") is None
    assert parse_requirement_line("git+https://github.com/x/y.git") is None


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


def test_deep_merge_override_wins_and_nests():
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"a": 2, "nested": {"y": 3, "z": 4}}
    merged = deep_merge(base, override)
    assert merged["a"] == 2
    assert merged["nested"] == {"x": 1, "y": 3, "z": 4}
