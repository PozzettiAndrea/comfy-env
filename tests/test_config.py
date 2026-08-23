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
        "settings": {"pool_ipc": False},
    })
    assert cfg.python == "3.11"
    assert cfg.cuda_packages == ["nvdiffrast"]
    assert cfg.env_vars == {"FOO": "1"}
    assert cfg.settings == {"pool_ipc": False}
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


def test_types_table_parses_and_validates():
    # ADR-0015: [types] is a closed vocabulary -- builtin | custom.
    cfg = parse_config({"types": {"TRIMESH": "custom", "SKELETON": "builtin"}})
    assert cfg.types == {"TRIMESH": "custom", "SKELETON": "builtin"}

    import pytest
    with pytest.raises(ValueError, match="TRIMESH.*not valid"):
        parse_config({"types": {"TRIMESH": "cusotm"}})


def test_serializers_section_rejected_with_migration_message(tmp_path):
    # [serializers] was replaced by [types] (ADR-0015, no backcompat) --
    # the env-file parser must say so instead of silently forwarding it.
    import pytest
    from comfy_env.config import load_config
    cf = tmp_path / "comfy-env.toml"
    cf.write_text('[serializers]\nmodules = ["my_wire_types"]\n')
    with pytest.raises(ValueError, match=r"\[types\]"):
        load_config(cf)


def _run_register_nodes_in(pack_dir):
    """Call register_nodes() with pack_dir as the caller's package dir."""
    shim = pack_dir / "shim.py"
    shim.write_text(
        "from comfy_env import register_nodes\n"
        "MAPPINGS = register_nodes()\n")
    import runpy
    return runpy.run_path(str(shim))


def test_types_custom_without_serialization_py_fails_loudly(tmp_path):
    # ADR-0015 teeth: declaring a custom socket without shipping the code
    # is a broken contract -- register_nodes must refuse, loudly.
    import pytest
    (tmp_path / "comfy-env-root.toml").write_text('[types]\nTRIMESH = "custom"\n')
    (tmp_path / "nodes").mkdir()
    with pytest.raises(ValueError, match="serialization.py"):
        _run_register_nodes_in(tmp_path)


def test_types_custom_loads_serialization_py(tmp_path):
    # The happy path: [types] custom + serialization.py that registers.
    (tmp_path / "comfy-env-root.toml").write_text('[types]\nWIDGET = "custom"\n')
    (tmp_path / "nodes").mkdir()
    (tmp_path / "serialization.py").write_text(
        "from comfy_env.isolation.workers._ipc_shared import register_serializer\n"
        "register_serializer('TmpWidget', lambda o, r: {'v': o.v})\n")
    result = _run_register_nodes_in(tmp_path)
    assert result["MAPPINGS"] == ({}, {})  # no nodes, but no refusal
    from comfy_env.isolation.workers._ipc_shared import REGISTRY
    assert REGISTRY.lookup_deserializer("TmpWidget") is None  # serialize-only
    assert "TmpWidget" in REGISTRY._by_type


def test_types_custom_registering_nothing_fails_loudly(tmp_path):
    import pytest
    (tmp_path / "comfy-env-root.toml").write_text('[types]\nWIDGET = "custom"\n')
    (tmp_path / "nodes").mkdir()
    (tmp_path / "serialization.py").write_text("# forgot to register\n")
    with pytest.raises(ValueError, match="registered no serializers"):
        _run_register_nodes_in(tmp_path)
