"""Shared harness for the memory-floor experiments.

Nothing here is imported by ``comfy_env``. These are standalone scripts run
against a real ComfyUI tree and a real pack environment, answering questions
the design asserts answers to.

This file exists because the equivalent harness has now been lost twice to
temp-directory cleanup. Keep experiments here, in the repo.

Configuration, all overridable by environment variable:

    COMFY_DIR        a ComfyUI source tree              (default: the
                     sam3dbody checkout on this machine)
    WORKER_PY        the interpreter a worker runs      (default: the
                     sam3dbody pack env's pixi python)
    COMFY_ENV_SRC    which comfy_env to import          (default: this repo)

Units: every number crossing a function boundary is BYTES. Format only at
the edge, with ``gib``/``mib``. A previous round mixed GiB and 1e9 and the
figures could not be compared.
"""

import os
import subprocess
import sys

GIB = 1024 ** 3
MIB = 1024 ** 2

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(_HERE, "..", "..", "src"))

COMFY_DIR = os.environ.get("COMFY_DIR", "/home/work/sam3dbody/ComfyUI")
WORKER_PY = os.environ.get(
    "WORKER_PY",
    "/home/andrej/.ce/envs/sam3dbody-nodes-py313-torch2-8-cu128"
    "/.pixi/envs/default/bin/python",
)
COMFY_ENV_SRC = os.environ.get("COMFY_ENV_SRC", REPO_SRC)
PROBE_DIR = _HERE


def bootstrap():
    """Put comfy_env and ComfyUI on this process's path. Call first."""
    try:
        import tomllib
        sys.modules.setdefault("tomli", tomllib)
    except ImportError:
        pass
    for path in (COMFY_ENV_SRC, COMFY_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


def gib(n):
    return "{:.2f}GiB".format((n or 0) / GIB)


def mib(n):
    return "{:.0f}MiB".format((n or 0) / MIB)


class Report:
    """Pass/fail accumulator. Every check states what it proves."""

    def __init__(self, title):
        self.title = title
        self.passed = []
        self.failed = []
        print("== {} ==".format(title))

    def check(self, name, ok, detail=""):
        (self.passed if ok else self.failed).append(name)
        print("  {}  {}  {}".format("PASS" if ok else "FAIL", name, detail))
        return ok

    def note(self, msg):
        print("  ..  {}".format(msg))

    def finish(self):
        print("\nRESULT: {} passed, {} failed".format(
            len(self.passed), len(self.failed)))
        if self.failed:
            print("FAILED: {}".format(", ".join(self.failed)))
            return 1
        return 0


def spawn(name, extra_env=None, worker_py=None, comfy_dir=None):
    """Start a real comfy-env worker. Caller must shut it down."""
    from comfy_env.isolation.workers.subprocess import SubprocessWorker

    worker = SubprocessWorker(
        python=worker_py or WORKER_PY,
        working_dir=comfy_dir or COMFY_DIR,
        sys_path=[PROBE_DIR, comfy_dir or COMFY_DIR],
        name=name,
        env=extra_env or {},
    )
    worker._ensure_started()
    return worker


def budget_callback(reply=None):
    """A stand-in for the pool's VRAM budget callback.

    A worker blocks on its first comfy load waiting for this, so every
    experiment that loads a model must register one.
    """
    seen = []

    def _cb(request):
        seen.append(request)
        out = {
            "device": "cuda:0",
            "extra_reserved_vram": 700 * MIB,
            "vram_state": "NORMAL_VRAM",
            "device_free_bytes": int(18 * GIB),
        }
        if reply:
            out.update(reply)
        return out

    return _cb, seen


def smi_free_bytes():
    """Device-wide free VRAM per the driver, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0]) * MIB
    except Exception:
        pass
    return None


def gpu_is_idle():
    """True when no compute process holds the card. Guards against a
    neighbouring experiment's worker polluting a measurement."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return not out.stdout.strip()
    except Exception:
        return True
