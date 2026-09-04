"""Contract: env rebuild decisions key on the DERIVATION OUTPUT (generated
manifest + wheel URLs), not on config bytes -- comment edits never rebuild,
real changes always do, and both builders agree on the pin rule."""

from comfy_env.config import parse_config
from comfy_env.install.workspace import (
    _bootstrap_torch_pin,
    _env_identity,
    _read_hash_file,
    _write_hash_file,
)
from comfy_env.packages.toml_generator import build_env_toml


def _manifest(config_dict, **kw):
    cfg = parse_config(config_dict)
    kw.setdefault("torch_index", None)
    kw.setdefault("bootstrap_python", "3.12")
    kw.setdefault("log", lambda m: None)
    return build_env_toml("t-env", cfg, **kw)


def test_identity_ignores_env_vars_and_comment_only_changes():
    # [env_vars] never reaches the generated manifest; comments never reach
    # parse_config. Neither may change the identity.
    a = _manifest({"dependencies": {"ffmpeg": "*"}})
    b = _manifest({"dependencies": {"ffmpeg": "*"},
                   "env_vars": {"FOO": "1"}})
    assert _env_identity(a, []) == _env_identity(b, [])


def test_identity_changes_on_real_derivation_changes():
    base = _manifest({"dependencies": {"ffmpeg": "*"}})
    diff_python = _manifest({"dependencies": {"ffmpeg": "*"}, "python": "3.11"})
    diff_pin = _manifest({"dependencies": {"ffmpeg": "*"}},
                         torch_pin="==2.10.*")
    assert _env_identity(base, []) != _env_identity(diff_python, [])
    assert _env_identity(base, []) != _env_identity(diff_pin, [])
    # Wheel URLs are part of the derivation output (post-pixi uv step).
    assert _env_identity(base, []) != _env_identity(base, ["https://x/a.whl"])


def test_identity_is_order_independent():
    m1 = {"workspace": {"name": "x", "platforms": ["win-64"]},
          "feature": {"node": {"dependencies": {"a": "*", "b": "*"}}}}
    m2 = {"feature": {"node": {"dependencies": {"b": "*", "a": "*"}}},
          "workspace": {"platforms": ["win-64"], "name": "x"}}
    urls = ["https://x/b.whl", "https://x/a.whl"]
    assert _env_identity(m1, urls) == _env_identity(m2, list(reversed(urls)))


def test_hash_file_roundtrip_and_v1_detection(tmp_path):
    hp = tmp_path / "install.hash"

    # v2 round-trip
    _write_hash_file(hp, "v2:abc", "deadbeef", log=lambda m: None)
    identity, fastkey, legacy = _read_hash_file(hp)
    assert (identity, fastkey, legacy) == ("v2:abc", "deadbeef", False)

    # legacy single-line v1 format is detected for grandfathering
    hp.write_text("0123456789abcdef\n", encoding="utf-8")
    identity, fastkey, legacy = _read_hash_file(hp)
    assert identity is None and fastkey is None and legacy is True

    # missing file
    identity, fastkey, legacy = _read_hash_file(tmp_path / "nope")
    assert identity is None and fastkey is None and legacy is False


def test_pin_rule_is_shared_and_wildcarded():
    # THE pin rule for both builders: major.minor wildcard, matching the
    # ABI-tag granularity. An exact pin here caused manifest thrash.
    assert _bootstrap_torch_pin("2.10.3") == "==2.10.*"
    assert _bootstrap_torch_pin("2.8.0") == "==2.8.*"
    assert _bootstrap_torch_pin(None) is None


class TestStampRecordsWhatItWasBuiltAgainst:
    """Nothing recorded the ComfyUI or the host-derived pins an env was built
    against, so no env could say what it expected and nothing could report
    drift. Two envs on the development machine ran a different memory manager
    than their host for exactly this reason."""

    def test_reads_comfyuis_own_generated_version_file(self, tmp_path):
        from comfy_env.environment.cache import read_comfyui_version
        (tmp_path / "comfyui_version.py").write_text(
            '# generated\n__version__ = "0.33.0"\n', encoding="utf-8")
        assert read_comfyui_version(tmp_path) == "0.33.0"

    def test_absent_or_unreadable_version_is_none_not_a_raise(self, tmp_path):
        """Catches: assuming every ComfyUI is a git checkout or ships the
        file. A missing identity must degrade, never take out install."""
        from comfy_env.environment.cache import read_comfyui_version
        assert read_comfyui_version(tmp_path) is None
        assert read_comfyui_version(None) is None
        (tmp_path / "comfyui_version.py").write_text("nonsense", encoding="utf-8")
        assert read_comfyui_version(tmp_path) is None

    def test_stamp_carries_the_new_fields(self, tmp_path):
        import json
        from comfy_env.environment.cache import write_env_stamp
        write_env_stamp(tmp_path, torch_pin="==2.8.*",
                        comfyui_version="0.33.0",
                        host_derived={"comfy-aimdo": "0.4.13"})
        stamp = json.loads((tmp_path / "env.stamp.json").read_text())
        assert stamp["comfyui_version"] == "0.33.0"
        assert stamp["host_derived"]["comfy-aimdo"] == "0.4.13"

    def test_drift_is_reported_but_never_fatal(self, tmp_path):
        """Catches: promoting drift to a hard failure. An env built against a
        different ComfyUI usually still works, and whether it does is decided
        at runtime by the protocol level, not by a string in a file."""
        from comfy_env.environment.cache import (
            describe_env_stamp_drift, validate_env_stamp, write_env_stamp,
        )
        write_env_stamp(tmp_path, comfyui_version="0.33.0",
                        host_derived={"comfy-aimdo": "0.4.13"})
        (tmp_path / "comfyui_version.py").write_text(
            '__version__ = "0.34.1"\n', encoding="utf-8")
        drift = describe_env_stamp_drift(
            tmp_path, tmp_path, {"comfy-aimdo": "0.4.15"})
        assert "0.33.0" in drift and "0.34.1" in drift
        assert "0.4.13" in drift and "0.4.15" in drift
        assert validate_env_stamp(tmp_path)[0] is True

    def test_no_drift_reports_nothing(self, tmp_path):
        """Catches: a reporter that always has something to say, which trains
        the operator to ignore it."""
        from comfy_env.environment.cache import (
            describe_env_stamp_drift, write_env_stamp,
        )
        write_env_stamp(tmp_path, comfyui_version="0.33.0",
                        host_derived={"comfy-aimdo": "0.4.13"})
        (tmp_path / "comfyui_version.py").write_text(
            '__version__ = "0.33.0"\n', encoding="utf-8")
        assert describe_env_stamp_drift(
            tmp_path, tmp_path, {"comfy-aimdo": "0.4.13"}) is None

    def test_old_stamps_without_the_fields_report_no_drift(self, tmp_path):
        """Catches: treating a missing field as a mismatch, which would make
        every pre-existing env look drifted on the first run after upgrade."""
        import json
        from comfy_env.environment.cache import describe_env_stamp_drift
        (tmp_path / "env.stamp.json").write_text(
            json.dumps({"abi_tag": "py313-torch2.8-cu128"}), encoding="utf-8")
        (tmp_path / "comfyui_version.py").write_text(
            '__version__ = "0.34.1"\n', encoding="utf-8")
        assert describe_env_stamp_drift(
            tmp_path, tmp_path, {"comfy-aimdo": "0.4.15"}) is None
