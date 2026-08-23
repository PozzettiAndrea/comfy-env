"""Contract: the in-process import fallback registers nodes.

With the isolate flag removed, this path is the ONLY degraded mode left:
a pack whose comfy-env.toml exists but whose env is not materialized (or
whose stamp is refused) must still boot via a plain in-process import
(ADR-0008). It has been load-bearing since 0.4.x and never had a
behavioural test -- a regression here is "every node in the pack silently
vanishes", the worst supported failure mode.

The test builds a real mini-pack and calls the real register_nodes() from
its __init__.py, exactly as ComfyUI would import it.
"""

import sys
import textwrap
import types


def _build_pack(tmp_path, with_config):
    pack = tmp_path / "ComfyUI-FallbackPack"
    nodes = pack / "nodes"
    nodes.mkdir(parents=True)
    (pack / "__init__.py").write_text(textwrap.dedent("""\
        from comfy_env import register_nodes
        NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()
    """), encoding="utf-8")
    (nodes / "__init__.py").write_text(textwrap.dedent("""\
        class FallbackNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {}}
            RETURN_TYPES = ()
            FUNCTION = "run"
            CATEGORY = "test"
            def run(self):
                return ()

        NODE_CLASS_MAPPINGS = {"FallbackNode": FallbackNode}
        NODE_DISPLAY_NAME_MAPPINGS = {"FallbackNode": "Fallback Node"}
    """), encoding="utf-8")
    if with_config:
        # Config present, env NEVER materialized: the fallback case.
        (nodes / "comfy-env.toml").write_text('python = "3.12"\n', encoding="utf-8")
    return pack


def _import_pack(tmp_path, monkeypatch, with_config):
    # A comfyui-shaped base so the plugin-root/comfyui walks resolve.
    comfyui = tmp_path / "comfyui"
    custom_nodes = comfyui / "custom_nodes"
    custom_nodes.mkdir(parents=True)
    (comfyui / "comfy").mkdir()
    (comfyui / "main.py").write_text("", encoding="utf-8")
    (comfyui / "folder_paths.py").write_text("", encoding="utf-8")
    pack = _build_pack(custom_nodes, with_config)

    monkeypatch.setitem(
        sys.modules, "folder_paths", types.SimpleNamespace(
            base_path=str(comfyui), __file__=str(comfyui / "folder_paths.py"))
    )
    monkeypatch.syspath_prepend(str(custom_nodes))
    sys.modules.pop("ComfyUI-FallbackPack", None)
    # Import the pack the way ComfyUI does: as a top-level package by dir name.
    import importlib
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_FallbackPack", pack / "__init__.py",
        submodule_search_locations=[str(pack)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_FallbackPack"] = mod
    try:
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.modules.pop("ComfyUI_FallbackPack", None)
        sys.modules.pop("ComfyUI_FallbackPack.nodes", None)


def test_missing_env_falls_back_to_inprocess_import(tmp_path, monkeypatch):
    """comfy-env.toml present, env never materialized -> nodes still register."""
    mod = _import_pack(tmp_path, monkeypatch, with_config=True)
    assert "FallbackNode" in mod.NODE_CLASS_MAPPINGS
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["FallbackNode"] == "Fallback Node"


def test_plain_pack_imports_inprocess(tmp_path, monkeypatch):
    """No comfy-env.toml anywhere -> ordinary in-process import."""
    mod = _import_pack(tmp_path, monkeypatch, with_config=False)
    assert "FallbackNode" in mod.NODE_CLASS_MAPPINGS


def test_all_sources_failing_raises_import_error(tmp_path, monkeypatch):
    """A pack whose only source fails to import must go IMPORT FAILED in
    ComfyUI's startup summary, not load green with zero nodes. The raise is
    the mechanism: ComfyUI's load_custom_node catches it and marks the pack."""
    import pytest
    comfyui = tmp_path / "comfyui"
    custom_nodes = comfyui / "custom_nodes"
    custom_nodes.mkdir(parents=True)
    (comfyui / "comfy").mkdir()
    (comfyui / "main.py").write_text("", encoding="utf-8")
    (comfyui / "folder_paths.py").write_text("", encoding="utf-8")
    pack = custom_nodes / "ComfyUI-BrokenPack"
    nodes = pack / "nodes"
    nodes.mkdir(parents=True)
    (pack / "__init__.py").write_text(
        "from comfy_env import register_nodes\n"
        "NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()\n",
        encoding="utf-8")
    (nodes / "__init__.py").write_text(
        "import a_module_that_does_not_exist_anywhere\n", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules, "folder_paths", types.SimpleNamespace(
            base_path=str(comfyui), __file__=str(comfyui / "folder_paths.py")))
    import importlib
    spec = importlib.util.spec_from_file_location(
        "ComfyUI_BrokenPack", pack / "__init__.py",
        submodule_search_locations=[str(pack)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ComfyUI_BrokenPack"] = mod
    try:
        with pytest.raises(ImportError, match="all 1 node source"):
            spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("ComfyUI_BrokenPack", None)
        sys.modules.pop("ComfyUI_BrokenPack.nodes", None)
