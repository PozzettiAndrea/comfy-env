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
