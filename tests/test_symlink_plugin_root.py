"""Contract: a pack reached through a junction/symlink keeps its
custom_nodes-side identity (#8).

The env NAME is the contract between install time and runtime: install
enumerates `custom_nodes/` unresolved (install/workspace.py), so it
materializes `<pack>-<subdir>`. The runtime binder must derive the same name
from the caller's path. `Path.resolve()` follows the link, the plugin-root
walk in `_find_env_dir` then never sees `custom_nodes`, and the name degrades
to the bare subdir ("nodes") -- an env install never creates, so every
symlinked pack silently falls back to in-process import. Worse, every
symlinked pack degrades to the SAME name.

Symlinks stand in for Windows junctions here: same code path, and
`os.path.abspath` (GetFullPathName) does not follow junctions either.
"""

import os
import sys
import types

import pytest

import comfy_env.environment.cache as cache
import comfy_env.isolation.wrap as wrap


pytestmark = pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="platform has no symlink support"
)


def _fake_comfyui(tmp_path):
    """A tree find_comfyui_dir_from_node accepts: main.py + comfy/ + custom_nodes/."""
    comfyui = tmp_path / "comfyui"
    (comfyui / "comfy").mkdir(parents=True)
    (comfyui / "custom_nodes").mkdir()
    (comfyui / "main.py").write_text("", encoding="utf-8")
    return comfyui


def _fake_pack(root, name):
    pack = root / name
    nodes = pack / "nodes"
    nodes.mkdir(parents=True)
    (nodes / "comfy-env.toml").write_text('python = "3.12"\n', encoding="utf-8")
    return pack


def _captured_env_name(monkeypatch, comfyui, node_dir):
    """Run _find_env_dir and capture the env name it looks up.

    folder_paths is stubbed the way ComfyUI runtime provides it -- that is the
    short-circuit real startups take, and without it the comfyui-root walk has
    its own (separate, latent) resolve issue outside this test's scope.
    """
    monkeypatch.setitem(
        sys.modules, "folder_paths", types.SimpleNamespace(base_path=str(comfyui))
    )
    seen = {}

    def fake_workspace_env_dir(comfyui_dir, env_name):
        seen["env_name"] = env_name
        return comfyui / "ws" / "envs" / env_name  # never exists -> returns None

    monkeypatch.setattr(cache, "get_workspace_env_dir", fake_workspace_env_dir)
    result = wrap._find_env_dir(node_dir, node_dir / "comfy-env.toml")
    assert result is None  # env not materialized; only the NAME is under test
    return seen.get("env_name")


def test_symlinked_pack_keeps_custom_nodes_identity(tmp_path, monkeypatch):
    comfyui = _fake_comfyui(tmp_path)
    real_pack = _fake_pack(tmp_path / "repo", "ComfyUI-FakePack")
    link = comfyui / "custom_nodes" / "ComfyUI-FakePack"
    os.symlink(real_pack, link, target_is_directory=True)

    env = _captured_env_name(monkeypatch, comfyui, link / "nodes")
    assert env == "fakepack-nodes", (
        f"symlinked pack derived env name {env!r} -- resolve() followed the "
        f"link and the plugin-root walk missed custom_nodes (#8)"
    )


def test_plain_pack_unchanged(tmp_path, monkeypatch):
    """The non-symlink layout must keep producing the same name (no regression)."""
    comfyui = _fake_comfyui(tmp_path)
    pack = _fake_pack(comfyui / "custom_nodes", "ComfyUI-PlainPack")

    env = _captured_env_name(monkeypatch, comfyui, pack / "nodes")
    assert env == "plainpack-nodes"


def test_walk_miss_logs_loudly(tmp_path, monkeypatch, capsys):
    """A pack with no custom_nodes ancestor must say so, not silently degrade.

    Every pack that misses the walk degrades to the same subdir-derived name,
    so a quiet miss is a cross-pack identity collision in waiting.
    """
    comfyui = _fake_comfyui(tmp_path)
    stray = _fake_pack(tmp_path / "elsewhere", "ComfyUI-Stray")

    env = _captured_env_name(monkeypatch, comfyui, stray / "nodes")
    assert env == "nodes"  # the degraded name, still used for the lookup
    err = capsys.readouterr().err
    assert "plugin root not found" in err
