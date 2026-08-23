"""Contract: setting resolution order is env var > per-node [settings] > default."""

from comfy_env.settings import get_numeric, resolve_bool, resolve_numeric


def test_env_var_states(monkeypatch):
    monkeypatch.delenv("COMFY_ENV_POOL_IPC", raising=False)
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, True) is True
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, False) is False
    for truthy in ("1", "true", "YES"):
        monkeypatch.setenv("COMFY_ENV_POOL_IPC", truthy)
        assert resolve_bool("COMFY_ENV_POOL_IPC", None, False) is True
    monkeypatch.setenv("COMFY_ENV_POOL_IPC", "0")
    assert resolve_bool("COMFY_ENV_POOL_IPC", None, True) is False


def test_per_node_settings_take_priority(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_POOL_IPC", "1")
    assert resolve_bool("COMFY_ENV_POOL_IPC", {"pool_ipc": False}, True) is False
    monkeypatch.delenv("COMFY_ENV_POOL_IPC", raising=False)
    assert resolve_bool("COMFY_ENV_POOL_IPC", {"pool_ipc": True}, False) is True


def test_numeric_resolution(monkeypatch):
    monkeypatch.setenv("COMFY_ENV_WORKER_VRAM_BUDGET", "8")
    assert get_numeric("COMFY_ENV_WORKER_VRAM_BUDGET", 0) == 8.0
    monkeypatch.setenv("COMFY_ENV_WORKER_VRAM_BUDGET", "garbage")
    assert get_numeric("COMFY_ENV_WORKER_VRAM_BUDGET", 0) == 0
    assert resolve_numeric(
        "COMFY_ENV_WORKER_VRAM_BUDGET", {"worker_vram_budget": "4"}, 0) == 4.0
