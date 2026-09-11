"""Contract: a V3 node's execute() receives the same locked class clone in a
worker that it receives in plain ComfyUI.

execution.py:278-285 always validates, clones and LOCKS the class before
calling execute, so `cls.foo = x` raises AttributeError in every native run.
The worker built the clone but never locked it, and only when hidden inputs
were present. That was the one place isolation was looser than upstream: a
node could work isolated and break for everyone else. Every other conformance
gap runs the other way.

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers import WorkerError
from comfy_env.isolation.workers.subprocess import SubprocessWorker

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")
if not COMFYUI_DIR:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    # The PARENT reconstructs io.NodeOutput from the wire, which in real use
    # runs inside ComfyUI. Give this process the checkout for the test only.
    sys.path.insert(0, COMFYUI_DIR)
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         sys_path=[COMFYUI_DIR], name="v3-lock-worker")
    try:
        yield w
    finally:
        w.shutdown()
        sys.path.remove(COMFYUI_DIR)


def _run(worker, cls, x):
    return worker.call_method(module_name="v3_lock_node", class_name=cls,
                              method_name="EXECUTE_NORMALIZED",
                              kwargs={"x": x}, timeout=120.0)


def test_class_write_in_execute_raises_as_upstream_does(worker):
    with pytest.raises(WorkerError) as ei:
        _run(worker, "V3Mutator", 3)
    # upstream's own message, from comfy_api.internal.lock_class
    assert "Cannot modify class attribute 'leak'" in str(ei.value)


def test_clone_is_built_even_with_no_hidden_inputs(worker):
    """cls.hidden is a holder of Nones, never None: PREPARE_CLASS_CLONE runs
    on every call, hidden or not, as it does upstream."""
    out = _run(worker, "V3HiddenReader", 1)
    assert tuple(out.args) == ("None",), out.args


def test_borrow_premise_holds():
    """The worker reaches make_locked_method_func through sys.modules, never
    an import (ADR-0006). That works only while (a) importing a V3 node's
    base module loads comfy_api.internal and (b) the function keeps its
    three-argument shape. Trip here before the worker degrades to a WARN."""
    saved = list(sys.path)
    sys.path.insert(0, COMFYUI_DIR)
    try:
        for m in [m for m in sys.modules if m.startswith("comfy_api")]:
            del sys.modules[m]
        from comfy_api.latest import io  # noqa: F401  -- what any V3 pack does
        internal = sys.modules.get("comfy_api.internal")
        assert internal is not None, "comfy_api.internal no longer loads with io"
        fn = internal.make_locked_method_func
        assert fn.__code__.co_varnames[:3] == ("type_obj", "func", "class_clone")
    finally:
        sys.path[:] = saved
