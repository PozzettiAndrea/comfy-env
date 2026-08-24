"""Contract: leftovers of the removed settings fail or vanish correctly.

The [settings] section itself was removed in 0.4.25 (pre-1.0, ADR-0017), so
ANY [settings] table in a root config now hits the closed-schema error --
which subsumes the earlier per-key tombstones. Env-var side: only a FALSY
isolate/install_isolated fails loudly (semantic inversion -- the machine was
told to run un-isolated and no longer will); truthy values and settings.env
residue (the old TUI wrote every key on save) are ignored silently.

These tests are scheduled for deletion together with the tombstones.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from comfy_env.config import load_config
from conftest import subprocess_env


def _root(tmp_path, body):
    p = tmp_path / "comfy-env-root.toml"
    p.write_text(body, encoding="utf-8")
    return p


@pytest.mark.parametrize("body", [
    "[settings]\nisolate = false\n",
    "[settings]\ninstall_isolated = false\n",
    "[settings]\nisolate = true\n",
    "[settings]\npool_ipc = true\n",
])
def test_any_settings_table_hits_the_closed_schema(tmp_path, body):
    """[settings] no longer exists; the closed root schema rejects the whole
    table loudly, naming what IS allowed and where settings went."""
    with pytest.raises(ValueError, match=r"unsupported section\(s\) \[settings\]"):
        load_config(_root(tmp_path, body))


def _run_with_env(var, value):
    """Import comfy_env.settings under a controlled environment."""
    code = "import comfy_env.settings; print('IMPORTED-OK')"
    env = subprocess_env(**{var: value})
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
def test_truthy_env_var_is_silently_ignored(var):
    r = _run_with_env(var, "1")
    assert r.returncode == 0
    assert "IMPORTED-OK" in r.stdout
    assert "removed" not in r.stderr


def test_settings_env_file_keys_are_skipped_not_errored(tmp_path):
    """TUI residue: a falsy key in settings.env must NOT brick the boot --
    it is silently skipped before it can reach os.environ."""
    home = tmp_path / "home"
    (home / ".comfy-env").mkdir(parents=True)
    (home / ".comfy-env" / "settings.env").write_text(
        "COMFY_ENV_ISOLATE=0\nCOMFY_ENV_POOL_IPC=0\n", encoding="utf-8"
    )
    code = (
        "import os, comfy_env.settings as s; "
        "print('ISOLATE-IN-ENV' if 'COMFY_ENV_ISOLATE' in os.environ else 'SKIPPED'); "
        "print('OTHERS-LOADED' if os.environ.get('COMFY_ENV_POOL_IPC') == '0' else 'OTHERS-MISSING')"
    )
    env = subprocess_env(HOME=str(home), USERPROFILE=str(home))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "SKIPPED" in r.stdout          # removed key never enters os.environ
    assert "OTHERS-LOADED" in r.stdout    # surviving keys still load


# --- COMFY_ENV_AUTO_INSTALL (removed 0.4.25) --------------------------------
# Opposite polarity to the two above: it did nothing when falsy (already the
# default) and DID something when truthy, so TRUTHY is the semantic inversion
# here -- the machine was told to self-heal missing envs and no longer will.

def test_truthy_auto_install_fails_the_import():
    r = _run_with_env("COMFY_ENV_AUTO_INSTALL", "1")
    assert r.returncode != 0
    assert "removed in 0.4.25" in r.stderr
    assert "install() is the only" in r.stderr


def test_falsy_auto_install_is_silently_ignored():
    """It was already the default, so a falsy value never meant anything."""
    r = _run_with_env("COMFY_ENV_AUTO_INSTALL", "0")
    assert r.returncode == 0
    assert "IMPORTED-OK" in r.stdout
    assert "removed" not in r.stderr


def test_auto_install_module_is_gone():
    """One builder: install/workspace.py. No lazy second implementation."""
    import importlib
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("comfy_env.isolation.auto_install")
