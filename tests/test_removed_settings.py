"""Contract: leftovers of the removed settings fail or vanish correctly.

The [settings] section itself was removed in 0.4.25 (pre-1.0, ADR-0017), so
ANY [settings] table in a root config now hits the closed-schema error --
which subsumes the earlier per-key tombstones. Env-var side: only a FALSY
isolate/install_isolated fails loudly (semantic inversion -- the machine was
told to run un-isolated and no longer will); truthy values are ignored
silently.

The ~/.comfy-env/settings.env tier is gone too, and one test below holds it
gone: the tombstones now reach the runtime through `import comfy_env`, which
is the reason they can fire at all.

These tests are scheduled for deletion together with the tombstones.
"""

import subprocess
import sys

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


def test_settings_env_file_is_not_read(tmp_path):
    """The file tier is gone, not quietly still wired.

    Catches a reintroduced loader: settings.env used to be read into os.environ
    with setdefault, which looked like a settings system and was one only for
    the two processes that imported comfy_env.settings -- never a worker. If a
    key in this file reaches os.environ, the dead tier is back.
    """
    home = tmp_path / "home"
    (home / ".comfy-env").mkdir(parents=True)
    (home / ".comfy-env" / "settings.env").write_text(
        "COMFY_ENV_POOL_IPC=1\n", encoding="utf-8"
    )
    code = (
        "import os, comfy_env.settings; "
        "print('FILE-READ' if 'COMFY_ENV_POOL_IPC' in os.environ else 'FILE-IGNORED')"
    )
    env = subprocess_env(HOME=str(home), USERPROFILE=str(home))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "FILE-IGNORED" in r.stdout, r.stdout


def test_importing_the_facade_arms_the_tombstones():
    """The tombstones are only worth having where ComfyUI reaches them.

    Catches the state this replaced: comfy_env.settings was imported by the CLI
    and the installer's wheel lookup and by nothing on the runtime path, so a
    machine exporting COMFY_ENV_ISOLATE=0 got no error from ComfyUI at all.
    Asserting on `import comfy_env` (not on comfy_env.settings) is the point.
    """
    r = subprocess.run(
        [sys.executable, "-c", "import comfy_env"],
        env=subprocess_env(COMFY_ENV_ISOLATE="0"),
        capture_output=True, text=True,
    )
    assert r.returncode != 0, r.stdout
    assert "removed in 0.4.25" in r.stderr


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
