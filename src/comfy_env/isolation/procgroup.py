"""Run a child in its own process group, so that killing it kills its tree.

``subprocess.run(timeout=...)`` kills the direct child only. Both places
comfy-env spawns a child under a wrapper (the metadata scan under ``pixi
run``, the worker likewise) that leaves the real Python orphaned and still
running, and on Windows it is worse: ``run()`` then blocks in
``communicate()`` for as long as the orphan holds the inherited pipe, so the
timeout that was meant to end the hang extends it.

POSIX: ``start_new_session`` makes the child a session and group leader, so
``killpg(child.pid)`` reaches everything it spawned. Windows: a new process
group plus ``taskkill /T``, which walks the tree by parent id.

Stdlib only; imports nothing from comfy_env.
"""

import os
import signal
import subprocess
import sys
from typing import List, Tuple


def popen_in_own_group(cmd: List[str], **kw) -> subprocess.Popen:
    if sys.platform == "win32":
        kw["creationflags"] = kw.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and every descendant. Safe on an already-dead process."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_with_tree_timeout(cmd: List[str], timeout: float, **kw
                          ) -> Tuple[subprocess.CompletedProcess, bool]:
    """``subprocess.run(capture_output=True)`` whose timeout kills the tree.

    Returns ``(completed, timed_out)``. On timeout the tree is dead before
    this returns, and stdout/stderr hold whatever the child wrote up to then,
    which is how a caller names the thing that hung. A KeyboardInterrupt (or
    any other exception) while waiting also kills the tree: the child is in
    its own session and would not see the terminal's SIGINT itself.
    """
    proc = popen_in_own_group(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        out, err = proc.communicate()
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err), True
    except BaseException:
        kill_process_tree(proc)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err), False
