"""Contract: settings resolve from env vars, then the default.

(The per-pack [settings] tier was removed in 0.4.25, and resolve_bool -- the
wrapper that still took a `node_settings` argument it ignored -- went with the
rest of the dead settings surface.)
"""

from comfy_env.settings import get_numeric


def test_numeric_resolution(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_TEST_NUMERIC", "8")
    assert get_numeric("COMFY_ENV_TEST_NUMERIC", 0) == 8.0
    monkeypatch.setenv("COMFY_ENV_TEST_NUMERIC", "garbage")
    assert get_numeric("COMFY_ENV_TEST_NUMERIC", 0) == 0
