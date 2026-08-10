"""Contract: env names are user-visible on-disk identity; changes orphan envs."""

from pathlib import Path

import comfy_env.environment.cache as cache


def test_root_config_uses_plugin_name_only(tmp_path):
    plugin = tmp_path / "ComfyUI-SAM3"
    plugin.mkdir()
    assert cache.get_env_name(plugin, plugin / "comfy-env.toml") == "sam3"


def test_subdir_config_appends_subdir(tmp_path):
    plugin = tmp_path / "comfyui-motioncapture"
    (plugin / "nodes").mkdir(parents=True)
    name = cache.get_env_name(plugin, plugin / "nodes" / "comfy-env.toml")
    assert name == "motioncapture-nodes"


def test_prefix_strip_variants(tmp_path):
    # One parent per case: Windows filesystems are case-insensitive, so
    # ComfyUI-Foo and comfyui-foo would collide in a shared tmp_path.
    for i, (raw, expected) in enumerate([
        ("ComfyUI-Foo", "foo"),
        ("ComfyUI_Foo", "foo"),
        ("comfyui-foo", "foo"),
        ("comfyui_foo", "foo"),
        ("NotComfy", "notcomfy"),
    ]):
        plugin = tmp_path / f"case{i}" / raw
        plugin.mkdir(parents=True)
        assert cache.get_env_name(plugin, plugin / "comfy-env.toml") == expected


def test_sanitization_produces_pixi_safe_names(tmp_path):
    plugin = tmp_path / "ComfyUI-Foo._disabled (copy)"
    plugin.mkdir()
    name = cache.get_env_name(plugin, plugin / "comfy-env.toml")
    assert name == "foo-disabled-copy"
    # pixi env names must match [a-z0-9-]+
    assert all(c.isdigit() or c.islower() or c == "-" for c in name)


def test_env_dir_name_is_abi_qualified(monkeypatch):
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2-10-cu128")
    assert cache._env_dir_name("sam3-nodes") == "sam3-nodes-py313-torch2-10-cu128"
