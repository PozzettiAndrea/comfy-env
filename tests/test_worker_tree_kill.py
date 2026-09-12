"""Contract: killing a worker kills its whole process tree.

Workers run under `pixi run`, which does not exec: the handle comfy-env
holds is the wrapper, and the Python doing the work is its child. A plain
kill() on the handle killed the wrapper and left the Python alive with its
VRAM, invisible to the pool (its parent became pid 1) and to the host's own
Ctrl-C. Every kill path now goes through one helper that kills the group.

These tests use a stand-in wrapper (a Python that spawns a Python) so they
need no pixi env; the shape is the same.
"""

import os
import subprocess
import sys
import textwrap
import time

import pytest

from comfy_env.isolation.procgroup import kill_process_tree, popen_in_own_group

WRAPPER = textwrap.dedent("""
    import subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    print(child.pid, flush=True)
    sys.stdout.flush()
    time.sleep(120)
""")


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_dead(pid, seconds=5.0):
    for _ in range(int(seconds * 20)):
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_group_kill_reaches_the_grandchild():
    proc = popen_in_own_group([sys.executable, "-c", WRAPPER], stdout=subprocess.PIPE)
    grandchild = int(proc.stdout.readline())
    assert _alive(grandchild)
    kill_process_tree(proc)
    proc.wait(timeout=5)
    assert _wait_dead(grandchild), "the grandchild survived the tree kill"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_plain_kill_is_the_bug():
    """The behaviour being fixed, kept as a negative so it stays fixed."""
    proc = popen_in_own_group([sys.executable, "-c", WRAPPER], stdout=subprocess.PIPE)
    grandchild = int(proc.stdout.readline())
    proc.kill(); proc.wait(timeout=5)
    time.sleep(0.2)
    assert _alive(grandchild), "if this fails, plain kill() now propagates and the helper is moot"
    os.kill(grandchild, 9)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_non_leader_is_still_killed():
    """Review trap: killpg on a process that is not a group leader raises
    ESRCH; swallowing it would have killed nothing. The helper must fall
    back to the process itself."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    kill_process_tree(proc)
    proc.wait(timeout=5)
    assert proc.returncode is not None and proc.returncode < 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_dead_leader_does_not_shield_its_group():
    """Review trap two: the leader has already exited but its child is alive
    in the group. An early return on poll() would leave the child."""
    leader_src = textwrap.dedent("""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(child.pid, flush=True)
    """)   # leader exits immediately; child stays in the group
    proc = popen_in_own_group([sys.executable, "-c", leader_src], stdout=subprocess.PIPE)
    child = int(proc.stdout.readline())
    proc.wait(timeout=5)
    assert proc.poll() is not None and _alive(child)
    kill_process_tree(proc)
    assert _wait_dead(child), "the child outlived a tree kill whose leader was already gone"


def test_every_kill_site_goes_through_the_helper():
    """No bare _process.kill() may reappear in subprocess.py."""
    from pathlib import Path
    import comfy_env.isolation.workers.subprocess as m
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "self._process.kill()" not in src
    assert src.count("self._kill_tree()") >= 4
    assert "popen_in_own_group(" in src
