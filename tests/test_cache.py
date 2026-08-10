"""Contract: workspace root override and env stamp validation."""

import json

import comfy_env.environment.cache as cache


def test_workspace_root_override(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    monkeypatch.setenv("COMFY_ENV_ROOT", str(root))
    monkeypatch.setattr(cache, "_ANNOUNCED_WS", True)  # silence banner
    assert cache.get_workspace_dir() == root
    assert root.is_dir()  # created on resolution


def test_stamp_roundtrip_and_abi_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_ABI_TAG", "py313-torch2-10-cu128")

    # Missing stamp: passes, but explicitly unverified.
    ok, reason = cache.validate_env_stamp(tmp_path)
    assert ok and "not verified" in reason

    # Write + validate on the same stack: ok.
    cache.write_env_stamp(tmp_path, torch_pin="==2.10.0", provenance="test")
    ok, reason = cache.validate_env_stamp(tmp_path)
    assert ok and "verified" in reason

    # A stamp from a different stack must be rejected loudly.
    stamp_file = tmp_path / cache._STAMP_FILE
    stamp = json.loads(stamp_file.read_text())
    stamp["abi_tag"] = "py310-torch2-4-cpu"
    stamp_file.write_text(json.dumps(stamp))
    ok, reason = cache.validate_env_stamp(tmp_path)
    assert not ok
    assert "py310-torch2-4-cpu" in reason
