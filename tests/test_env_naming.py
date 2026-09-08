"""Contract: env names are user-visible on-disk identity; changes orphan envs."""


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
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2.10-cu128")
    assert cache._env_dir_name("sam3-nodes") == "sam3-nodes_py313-torch2.10-cu128"


def test_the_seam_cannot_occur_in_either_half(monkeypatch):
    """ADR-0039. Catches a seam that is also a legal name character.

    The old separator was `-`, which `get_env_name` ALSO uses to join a pack
    to its config subdirectory and `_abi_tag` uses to join its own fields. So
    `foo-nodes-py313-...` could not be split, by eye or by code, and
    `ComfyUI-Foo-Bar` (root config) collided with `ComfyUI-Foo` (config in
    `bar/`). `_` is safe precisely because the sanitizer collapses it out of
    every name component.
    """
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2.10-cu128")
    dir_name = cache._env_dir_name("foo-bar")
    assert dir_name.count(cache._DIR_SEP) == 1, dir_name
    logical, tag = dir_name.split(cache._DIR_SEP)
    assert logical == "foo-bar" and tag == "py313-torch2.10-cu128"


def test_abi_tag_keeps_version_dots(monkeypatch):
    """Catches re-running the tag through `_sanitize_pixi_name`.

    That is what rendered torch 2.10 as `torch2-10`, which reads as a
    component boundary rather than a version and is indistinguishable from
    torch 2 build 10. The tag is a directory name only; the pixi environment
    is always called `default`, so the [a-z0-9-] rule never applied to it.
    """
    tag = cache._abi_tag()          # the live interpreter, no mocking
    assert "_" not in tag, f"the seam must not occur inside the tag: {tag}"
    if "torch" in tag and "notorch" not in tag:
        assert "." in tag, f"version dots were collapsed: {tag}"


def test_legacy_names_recover_the_pre_adr_spelling(monkeypatch):
    """Adoption, not orphaning. Catches shipping the rename without the alias.

    Without this, every env built before ADR-0039 becomes unreferenced, so
    `comfy-env gc --delete` offers to delete envs that are in active use, and
    `install()` re-materializes what is already on disk. That is exactly what
    the previous rename did: four pre-tag directories, 26 GB, still on the
    maintainer's machine.
    """
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2.10-cu128")
    legacy = cache.legacy_dir_names("sam3-nodes")
    assert "sam3-nodes-py313-torch2.10-cu128" in legacy   # seam was a dash
    assert "sam3-nodes-py313-torch2-10-cu128" in legacy   # and dots were dashed
    assert cache._env_dir_name("sam3-nodes") not in legacy


def test_colliding_packs_get_distinct_source_ids(tmp_path):
    """ADR-0039. Catches shipping the seam fix and calling the collision solved.

    `ComfyUI-Foo-Bar` with a root config and `ComfyUI-Foo` with a config in
    `bar/` both derive the env name `foo-bar`. A separator makes a name
    readable; it cannot stop two sources reducing to one name. Only the
    recorded source tells them apart.
    """
    a = tmp_path / "ComfyUI-Foo-Bar"
    (a).mkdir()
    (a / "comfy-env.toml").touch()
    b = tmp_path / "ComfyUI-Foo" / "bar"
    b.mkdir(parents=True)
    (b / "comfy-env.toml").touch()

    assert cache.get_env_name(a, a / "comfy-env.toml") == \
           cache.get_env_name(a.parent / "ComfyUI-Foo", b / "comfy-env.toml")
    assert cache.env_source_id(a, a / "comfy-env.toml") != \
           cache.env_source_id(a.parent / "ComfyUI-Foo", b / "comfy-env.toml")


def test_stamp_refuses_a_bind_from_a_different_source(tmp_path, monkeypatch):
    """The collision must be LOUD. Catches recording source and never reading it.

    Both packs build for the same stack, so the abi_tag check passes and the
    two silently share one directory, rebuilding over each other forever with
    no diagnostic. That is the failure this field exists for.
    """
    import json
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2.10-cu128")
    d = tmp_path / "env"
    d.mkdir()
    (d / "env.stamp.json").write_text(json.dumps({
        "abi_tag": "py313-torch2.10-cu128", "source": "ComfyUI-Foo-Bar/comfy-env.toml",
    }), encoding="utf-8")

    ok, _ = cache.validate_env_stamp(d, expected_source="ComfyUI-Foo-Bar/comfy-env.toml")
    assert ok
    ok, reason = cache.validate_env_stamp(d, expected_source="ComfyUI-Foo/bar/comfy-env.toml")
    assert not ok and "ComfyUI-Foo-Bar" in reason and "ComfyUI-Foo/bar" in reason


def test_stamp_without_a_source_still_binds(tmp_path, monkeypatch):
    """Stamps written before ADR-0039 have no source and must not be refused."""
    import json
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2.10-cu128")
    d = tmp_path / "env"
    d.mkdir()
    (d / "env.stamp.json").write_text(
        json.dumps({"abi_tag": "py313-torch2.10-cu128"}), encoding="utf-8")
    ok, _ = cache.validate_env_stamp(d, expected_source="anything/at/all.toml")
    assert ok


def test_env_label_names_the_env_not_default(tmp_path):
    """Catches `Path(env_dir).name`, which printed "default" for every env.

    `get_workspace_env_dir` returns `<ws>/envs/<dir>/.pixi/envs/default`, so
    the two loudest log lines in the system (worker teardown and worker OOM)
    named nothing at all.
    """
    materialized = tmp_path / "envs" / "sam3-nodes_py313-torch2.8-cu128" / ".pixi" / "envs" / "default"
    materialized.mkdir(parents=True)
    assert cache.env_label(materialized) == "sam3-nodes_py313-torch2.8-cu128"
    assert cache.env_label(tmp_path / "plain") == "plain"


def test_pixi_detection_matches_a_component_not_a_substring():
    """Catches `".pixi" in str(python)`, which is true for any path containing
    the characters anywhere.

    A host interpreter installed by `pixi global` lives under `~/.pixi/`, so
    that test fired for an interpreter that is not one of our envs at all, and
    the code then resolved a per-env manifest three directories up from
    somewhere it does not exist. Same reading-semantics-out-of-a-path mistake
    as the libomp dedupe, third instance.
    """
    from pathlib import PurePosixPath as P
    ours = P("/home/u/.ce/envs/sam3-nodes_py313-torch2.8-cu128/.pixi/envs/default/bin/python")
    theirs = P("/home/u/.pixi/envs/somethingelse/bin/python")
    assert ".pixi" in ours.parts
    assert ".pixi" in theirs.parts          # both are pixi envs, correctly
    # the substring form ALSO matched a pack that merely contains the text:
    innocent = P("/home/u/.ce/envs/my.pixiart-nodes_py313-torch2.8-cu128/bin/python")
    assert ".pixi" in str(innocent)         # old test: wrongly true
    assert ".pixi" not in innocent.parts    # new test: correctly false
