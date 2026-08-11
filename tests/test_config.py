"""Contract: what comfy-env.toml / comfy-env-root.toml mean to users."""

from comfy_env.config import parse_config


def test_unknown_keys_pass_through_to_pixi():
    cfg = parse_config({
        "dependencies": {"ffmpeg": "*"},
        "pypi-dependencies": {"trimesh": "*"},
        "some-future-pixi-table": {"x": 1},
    })
    assert cfg.pixi_passthrough["dependencies"] == {"ffmpeg": "*"}
    assert cfg.pixi_passthrough["pypi-dependencies"] == {"trimesh": "*"}
    assert cfg.pixi_passthrough["some-future-pixi-table"] == {"x": 1}


def test_known_sections_are_extracted_not_passed_through():
    cfg = parse_config({
        "python": "3.11",
        "cuda": {"packages": ["nvdiffrast"]},
        "env_vars": {"FOO": 1},
        "settings": {"isolate": False},
    })
    assert cfg.python == "3.11"
    assert cfg.cuda_packages == ["nvdiffrast"]
    assert cfg.env_vars == {"FOO": "1"}
    assert cfg.settings == {"isolate": False}
    assert cfg.pixi_passthrough == {}


def test_unknown_tables_are_forwarded_pixi_validates():
    # ADR-0013 honest passthrough: unknown tables (including legacy [apt])
    # are FORWARDED into the generated manifest at feature level, where the
    # pinned pixi rejects invalid ones loudly at install time -- pixi is the
    # validator for its own language, comfy-env keeps no allowlist.
    from comfy_env.packages.toml_generator import build_env_toml
    cfg = parse_config({"apt": {"packages": ["libgl1"]}, "dependencies": {"ffmpeg": "*"}})
    manifest = build_env_toml("t", cfg, torch_index=None, log=lambda m: None)
    assert manifest["feature"]["node"]["apt"] == {"packages": ["libgl1"]}


def test_cuda_scalar_normalized_to_list():
    cfg = parse_config({"cuda": {"packages": "flash-attn"}})
    assert cfg.cuda_packages == ["flash-attn"]
    assert cfg.has_cuda


def test_node_reqs_string_and_table_forms():
    cfg = parse_config({"node_reqs": {
        "ComfyUI-GeometryPack": "https://github.com/PozzettiAndrea/ComfyUI-GeometryPack",
        "OtherPack": {"registry": "other-pack", "version": "1.2.3"},
    }})
    by_name = {r["name"]: r for r in cfg.node_reqs}
    assert by_name["ComfyUI-GeometryPack"]["github"].endswith("GeometryPack")
    assert by_name["OtherPack"]["registry"] == "other-pack"
    assert by_name["OtherPack"]["version"] == "1.2.3"
    assert cfg.has_dependencies
