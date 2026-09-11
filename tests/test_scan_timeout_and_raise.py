"""Contract: the metadata scan cannot hang ComfyUI's startup, and a node whose
INPUT_TYPES raises is a missing node, not a hollow one.

Hang: the scan runs a pack's import and every INPUT_TYPES() in a subprocess
before ComfyUI finishes starting. It had no timeout, so a pack that hung held
startup forever with nothing printed. A plain timeout would not have fixed it:
under `pixi run` the scanning Python is a grandchild, and killing the wrapper
leaves it running (and on Windows, blocks the parent on its pipe).

Raise: a node whose INPUT_TYPES() raised in the scan registered with zero
inputs. It looked healthy in the menu, rendered with no widgets, and the
frontend serialises a registered node from its live widgets, so a workflow
saved with it lost every widget value. A missing node keeps its saved data
verbatim (litegraph's `last_serialization`) and upstream reports the real
cause when the workflow is queued. So the proxy re-raises.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md
from comfy_env.isolation.procgroup import run_with_tree_timeout


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_tree_timeout_kills_the_grandchild():
    """The shape of `pixi run python scan.py`: a wrapper that spawns the real
    process. The grandchild must be dead when the call returns."""
    child = textwrap.dedent(f"""
        import subprocess, sys, time
        g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(g.pid, flush=True)
        time.sleep(60)
    """)
    t0 = time.perf_counter()
    result, timed_out = run_with_tree_timeout([sys.executable, "-c", child], timeout=2.0)
    elapsed = time.perf_counter() - t0
    assert timed_out
    assert elapsed < 15, f"communicate() blocked after the kill ({elapsed:.1f}s)"
    grandchild = int(result.stdout.decode().strip())
    # give the OS a moment to reap
    for _ in range(50):
        if not _pid_alive(grandchild):
            break
        time.sleep(0.1)
    assert not _pid_alive(grandchild), "the grandchild survived the timeout"


def test_tree_timeout_returns_the_partial_stderr():
    """What the child wrote before the kill is what names the culprit."""
    child = "import sys, time; print('progress', file=sys.stderr, flush=True); time.sleep(60)"
    result, timed_out = run_with_tree_timeout([sys.executable, "-c", child], timeout=1.5)
    assert timed_out and b"progress" in result.stderr


# --------------------------------------------------------------------------
# the scan, end to end through fetch_metadata
# --------------------------------------------------------------------------

def _fake_env(tmp_path: Path) -> Path:
    """An env dir whose python is this interpreter."""
    env_dir = tmp_path / "env"
    if sys.platform == "win32":
        env_dir.mkdir()
        (env_dir / "python.exe").symlink_to(sys.executable)
    else:
        (env_dir / "bin").mkdir(parents=True)
        (env_dir / "bin" / "python").symlink_to(sys.executable)
    return env_dir


def _pack(tmp_path: Path, name: str, body: str, init_extra: str = "") -> Path:
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "node.py").write_text(textwrap.dedent(body), encoding="utf-8")
    (pkg / "__init__.py").write_text(init_extra + "from .node import *\n", encoding="utf-8")
    return pkg


HANGING_PACK = """
    import time
    class Fine:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("INT", {"default": 0})}}
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
    class Hanger:
        @classmethod
        def INPUT_TYPES(cls):
            time.sleep(120)
            return {"required": {}}
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
    NODE_CLASS_MAPPINGS = {"Fine": Fine, "Hanger": Hanger}
    NODE_DISPLAY_NAME_MAPPINGS = {}
"""


def test_scan_timeout_names_the_node_that_hung(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(md.SCAN_TIMEOUT_ENV_VAR, "3")
    _pack(tmp_path, "hang_pkg", HANGING_PACK)
    env_dir = _fake_env(tmp_path)
    t0 = time.perf_counter()
    payload = md.fetch_metadata(env_dir, "hang_pkg", tmp_path)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20, f"the timeout did not end the scan ({elapsed:.1f}s)"
    assert payload == {"nodes": {}, "display": {}}
    err = capsys.readouterr().err
    assert "exceeded 3s and was killed while scanning node 'Hanger'" in err, err
    assert md.SCAN_TIMEOUT_ENV_VAR in err
    assert not (env_dir / ".metadata_cache.json").exists(), "a killed scan was cached"


def test_scan_timeout_during_import_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(md.SCAN_TIMEOUT_ENV_VAR, "3")
    _pack(tmp_path, "hang_import_pkg", "NODE_CLASS_MAPPINGS = {}\n",
          init_extra="import time; time.sleep(120)\n")
    env_dir = _fake_env(tmp_path)
    payload = md.fetch_metadata(env_dir, "hang_import_pkg", tmp_path)
    assert payload == {"nodes": {}, "display": {}}
    assert "during import (before any node was scanned)" in capsys.readouterr().err


# --------------------------------------------------------------------------
# a raising INPUT_TYPES
# --------------------------------------------------------------------------

RAISING_PACK = """
    class Fine:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("INT", {"default": 0})}}
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
    class Broken:
        @classmethod
        def INPUT_TYPES(cls):
            raise FileNotFoundError("models/list.json is missing")
        RETURN_TYPES = ("INT",)
        FUNCTION = "run"
    NODE_CLASS_MAPPINGS = {"Fine": Fine, "Broken": Broken}
    NODE_DISPLAY_NAME_MAPPINGS = {}
"""


def test_raising_input_types_reraises_through_the_proxy(tmp_path, capsys):
    _pack(tmp_path, "raise_pkg", RAISING_PACK)
    env_dir = _fake_env(tmp_path)
    payload = md.fetch_metadata(env_dir, "raise_pkg", tmp_path)
    assert set(payload["nodes"]) == {"Fine", "Broken"}, "the healthy node must survive"

    Broken = md.build_proxy_class(
        node_name="Broken", meta=payload["nodes"]["Broken"], env_dir=env_dir,
        package_root=tmp_path, sys_path=[], env_vars={})
    with pytest.raises(RuntimeError) as ei:
        Broken.INPUT_TYPES()
    # the real cause, verbatim, so validate_prompt's exception_during_validation
    # and /object_info's logged traceback both show it
    assert "models/list.json is missing" in str(ei.value)
    assert "Broken.INPUT_TYPES() raised in its isolated environment" in str(ei.value)

    Fine = md.build_proxy_class(
        node_name="Fine", meta=payload["nodes"]["Fine"], env_dir=env_dir,
        package_root=tmp_path, sys_path=[], env_vars={})
    assert Fine.INPUT_TYPES()["required"]["x"][0] == "INT"

    err = capsys.readouterr().err
    assert "node 'Broken' INPUT_TYPES() raised during the scan" in err
    assert "ZERO inputs" not in err, "the warning still describes the old hollow registration"


def test_raising_input_types_is_not_cached_but_a_healthy_scan_is(tmp_path):
    env_dir = _fake_env(tmp_path)
    _pack(tmp_path, "raise_pkg", RAISING_PACK)
    md.fetch_metadata(env_dir, "raise_pkg", tmp_path)
    assert not (env_dir / ".metadata_cache.json").exists(), \
        "a scan-time failure was cached and would outlive its cause"

    env_ok = _fake_env(tmp_path / "ok")
    _pack(tmp_path, "fine_pkg", RAISING_PACK.replace(
        'raise FileNotFoundError("models/list.json is missing")',
        'return {"required": {}}'))
    md.fetch_metadata(env_ok, "fine_pkg", tmp_path)
    cache = json.loads((env_ok / ".metadata_cache.json").read_text(encoding="utf-8"))
    assert "Broken" in json.dumps(cache), "a healthy scan must still be cached"
