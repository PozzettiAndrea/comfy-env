"""Contract: the removed isolate/install_isolated settings tombstone correctly.

Removed in 0.4.25 (pre-1.0, ADR-0017). The tombstones are value-sensitive by
design: a FALSY value is a semantic inversion -- the machine was told to run
un-isolated and no longer will -- and must fail loudly rather than silently
flip. A TRUTHY value matches the only behavior that exists now (and our own
docs shipped `isolate = true` in the sample [settings] block), so it warns.
Keys in ~/.comfy-env/settings.env are residue: the TUI wrote EVERY settings
key on save, so those never error and self-heal on the next save.

These tests are scheduled for deletion together with the tombstones.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from comfy_env.config import load_config


# --- [settings] keys in comfy-env-root.toml --------------------------------


def _root(tmp_path, body):
    p = tmp_path / "comfy-env-root.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_toml_falsy_isolate_is_a_hard_error(tmp_path):
    with pytest.raises(ValueError, match="removed in comfy-env 0.4.25"):
        load_config(_root(tmp_path, "[settings]\nisolate = false\n"))


def test_toml_falsy_install_isolated_is_a_hard_error(tmp_path):
    with pytest.raises(ValueError, match="removed in comfy-env 0.4.25"):
        load_config(_root(tmp_path, "[settings]\ninstall_isolated = false\n"))


def test_toml_truthy_isolate_warns_and_is_dropped(tmp_path, capsys):
    cfg = load_config(_root(tmp_path, "[settings]\nisolate = true\n"))
    err = capsys.readouterr().err
    assert "removed in 0.4.25" in err
    # Dropped, not forwarded: nothing downstream may see the removed key.
    assert "isolate" not in (cfg.settings or {})
    # And NOT the misleading typo warning -- it was a recognized key for years.
    assert "unrecognized key 'isolate'" not in err


# --- env vars: checked in a subprocess, since settings.py acts at import ----


def _run_with_env(var, value):
    """Import comfy_env.settings under a controlled environment."""
    code = "import comfy_env.settings; print('IMPORTED-OK')"
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent-so-settings-env-is-absent",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        var: value,
    }
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
    )


@pytest.mark.parametrize("var", ["COMFY_ENV_ISOLATE", "COMFY_ENV_INSTALL_ISOLATED"])
def test_falsy_env_var_fails_the_import(var):
    r = _run_with_env(var, "0")
    assert r.returncode != 0
    assert "removed in 0.4.25" in r.stderr
    assert "Unset this variable" in r.stderr


@pytest.mark.parametrize("var", ["COMFY_ENV_ISOLATE", "COMFY_ENV_INSTALL_ISOLATED"])
def test_truthy_env_var_warns_and_continues(var):
    r = _run_with_env(var, "1")
    assert r.returncode == 0
    assert "IMPORTED-OK" in r.stdout
    assert "removed in 0.4.25" in r.stderr


def test_settings_env_file_keys_are_skipped_not_errored(tmp_path):
    """TUI residue: a falsy key in settings.env must NOT brick the boot --
    it is skipped before it can reach os.environ, with a cleanup hint."""
    home = tmp_path / "home"
    (home / ".comfy-env").mkdir(parents=True)
    (home / ".comfy-env" / "settings.env").write_text(
        "COMFY_ENV_ISOLATE=0\nCOMFY_ENV_AUTO_INSTALL=0\n", encoding="utf-8"
    )
    code = (
        "import os, comfy_env.settings as s; "
        "print('ISOLATE-IN-ENV' if 'COMFY_ENV_ISOLATE' in os.environ else 'SKIPPED'); "
        "print('OTHERS-LOADED' if os.environ.get('COMFY_ENV_AUTO_INSTALL') == '0' else 'OTHERS-MISSING')"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SKIPPED" in r.stdout          # removed key never enters os.environ
    assert "OTHERS-LOADED" in r.stdout    # surviving keys still load
    assert "rerun `comfy-env settings`" in r.stderr


def test_toml_worker_vram_budget_warns_any_value(tmp_path, capsys):
    """Numeric key: no inversion value exists, so every value warns-and-drops
    (0 already meant auto; a stale cap returns to negotiated behavior)."""
    cfg = load_config(_root(tmp_path, "[settings]\nworker_vram_budget = 8\n"))
    err = capsys.readouterr().err
    assert "worker_vram_budget was removed in 0.4.25" in err
    assert "worker_vram_budget" not in (cfg.settings or {})


def test_env_worker_vram_budget_warns_and_continues():
    r = _run_with_env("COMFY_ENV_WORKER_VRAM_BUDGET", "8")
    assert r.returncode == 0
    assert "IMPORTED-OK" in r.stdout
    assert "removed in 0.4.25" in r.stderr
