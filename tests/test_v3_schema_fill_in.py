"""Contract: a V3 node that returns an expand graph works in a worker.

Upstream sets cls.SCHEMA once at registration (the comfy_entrypoint loop in
nodes.py calls GET_SCHEMA); NodeOutput's expand check then dereferences
cls.SCHEMA.enable_expand. A worker imports the pack directly and never runs
registration, so SCHEMA sat at its class default of None and an expand node
died on NoneType while a plain node was fine (the check short-circuits). The
worker now calls GET_SCHEMA() when SCHEMA is None.

Two halves, as review asked: first pin UPSTREAM's contract with a negative
(the fill-in is load-bearing only while EXECUTE_NORMALIZED dereferences an
unset SCHEMA), then show expand survives the wire.

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""

import os
import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")
if not COMFYUI_DIR:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures"

# comfy.model_management parses ComfyUI's args on first import and, on a
# CPU-only torch, asserts unless --cpu was given. That happens on the FIRST
# comfy import in the pytest process, whichever test module triggers it, so
# every comfyui-marked module that imports comfy does this before anything.
def _parse_cpu_args():
    sys.path.insert(0, COMFYUI_DIR)
    import comfy.options
    comfy.options.enable_args_parsing()
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]
    import comfy.cli_args  # noqa: F401  -- parses now, under --cpu
    sys.path.remove(COMFYUI_DIR)


_parse_cpu_args()


@pytest.fixture()
def comfy_path():
    sys.path.insert(0, COMFYUI_DIR)
    try:
        yield
    finally:
        sys.path.remove(COMFYUI_DIR)


def test_upstream_dereferences_schema_before_registration(comfy_path):
    """The negative. If this ever passes without the fill-in, upstream made
    EXECUTE_NORMALIZED self-sufficient and the worker's GET_SCHEMA() call is
    dead code."""
    from comfy_api.latest import io

    class Fresh(io.ComfyNode):
        @classmethod
        def define_schema(cls):
            return io.Schema(node_id="Fresh", inputs=[io.Int.Input("x")],
                             outputs=[io.Int.Output()], enable_expand=True)

        @classmethod
        def execute(cls, x):
            return io.NodeOutput(expand={"n1": {}})

    assert Fresh.SCHEMA is None, "unregistered V3 classes no longer start with SCHEMA=None"
    with pytest.raises(AttributeError):
        Fresh.EXECUTE_NORMALIZED(x=1)
    Fresh.GET_SCHEMA()
    out = Fresh.EXECUTE_NORMALIZED(x=1)
    assert out.expand == {"n1": {}}


def test_expand_graph_survives_the_worker(comfy_path):
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         sys_path=[COMFYUI_DIR], name="expand-worker")
    try:
        out = w.call_method(module_name="v3_expand_node", class_name="Expander",
                            method_name="EXECUTE_NORMALIZED", kwargs={"x": 4}, timeout=120.0)
    finally:
        w.shutdown()
    assert out.expand == {"n1": {"class_type": "Nothing", "inputs": {"v": 4}}}
