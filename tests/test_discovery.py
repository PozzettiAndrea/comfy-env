"""Contract: install discovery matches the runtime binder exactly, and
duplicate env names are a hard error (not silent shared-dir thrash)."""

import pytest

from comfy_env.install.workspace import _discover_node_configs


def _pack(custom_nodes, name):
    d = custom_nodes / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _env_cfg(dirpath, body="[dependencies]\n"):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "comfy-env.toml").write_text(body, encoding="utf-8")


def test_discovers_nodes_root_and_subdirs(tmp_path):
    cn = tmp_path / "custom_nodes"
    a = _pack(cn, "ComfyUI-A")
    _env_cfg(a / "nodes")
    b = _pack(cn, "ComfyUI-B")
    _env_cfg(b / "nodes" / "x")
    _env_cfg(b / "nodes" / "y")

    found = _discover_node_configs(tmp_path, log=lambda m: None)
    names = sorted(name for name, *_ in found)
    assert names == ["a-nodes", "b-x", "b-y"]


def test_ignores_unbindable_shapes(tmp_path):
    """The old rglob discovered all of these; the binder can bind none of
    them, so discovery must not either."""
    cn = tmp_path / "custom_nodes"
    p = _pack(cn, "ComfyUI-P")
    (p / "comfy-env.toml").write_text("[dependencies]\n", encoding="utf-8")  # pack root
    _env_cfg(p / "nodes" / "a" / "deep")            # depth 3
    _env_cfg(p / "vendored" / "third_party")         # outside nodes/

    assert _discover_node_configs(tmp_path, log=lambda m: None) == []


def test_duplicate_env_name_is_hard_error(tmp_path):
    """ComfyUI-Foo and comfyui_foo both derive foo-nodes: sharing one env
    dir means permanently rebuilding over each other. Refuse loudly."""
    cn = tmp_path / "custom_nodes"
    _env_cfg(_pack(cn, "ComfyUI-Foo") / "nodes")
    _env_cfg(_pack(cn, "comfyui_foo") / "nodes")

    with pytest.raises(ValueError, match="foo-nodes.*BOTH"):
        _discover_node_configs(tmp_path, log=lambda m: None)


def test_disabled_and_hidden_packs_skipped(tmp_path):
    cn = tmp_path / "custom_nodes"
    _env_cfg(_pack(cn, "ComfyUI-Live") / "nodes")
    _env_cfg(_pack(cn, "ComfyUI-Old.disabled") / "nodes")
    _env_cfg(_pack(cn, ".hidden") / "nodes")

    found = _discover_node_configs(tmp_path, log=lambda m: None)
    assert [name for name, *_ in found] == ["live-nodes"]
