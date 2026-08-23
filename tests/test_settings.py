"""Contract: setting resolution order is env var > per-node [settings] > default."""

from comfy_env.settings import get_numeric, resolve_bool


def test_env_var_states(monkeypatch):
    monkeypatch.delenv("COMFY_ENV_POOL_IPC", raising=False)
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, True) is True
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, False) is False
    for truthy in ("1", "true", "YES"):
        monkeypatch.setenv("COMFY_ENV_POOL_IPC", truthy)
        assert resolve_bool("COMFY_ENV_POOL_IPC", None, False) is True
    monkeypatch.setenv("COMFY_ENV_POOL_IPC", "0")
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, True) is False


def test_numeric_resolution(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_TEST_NUMERIC", "8")
    assert get_numeric("COMFY_ENV_TEST_NUMERIC", 0) == 8.0
    monkeypatch.setenv("COMFY_ENV_TEST_NUMERIC", "garbage")
    assert get_numeric("COMFY_ENV_TEST_NUMERIC", 0) == 0
