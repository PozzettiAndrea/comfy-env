"""Contract: hidden inputs reach the node they were declared for.

ComfyUI hands a node engine/client state it never wired up -- the prompt graph,
the workflow, the node id, API credentials -- by declaring a sentinel in
INPUT_TYPES["hidden"]. Those values are what put the `prompt` and `workflow`
chunks in a saved PNG, which is what makes drag-and-drop workflow restore work.

comfy-env used to drop all of them except a parameter spelled exactly
`unique_id`, so an isolated save node wrote images with no metadata at all --
silently, because upstream guards every write with `if prompt is not None`.

These run the REAL worker over the REAL transport. That matters: the whole
class of bug lived because no test in the suite drove `call_method`, the path
every node proxy actually takes -- the other real-worker tests all use
`call_module` against module-level functions.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"

PROMPT = {"7": {"class_type": "KSampler", "inputs": {"seed": 42}}}
PNGINFO = {"workflow": {"nodes": [{"id": 7}], "links": []}}


@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="hidden-test-worker")
    yield w
    w.shutdown()


def _call(worker, cls, method, hidden):
    return worker.call_method(
        module_name="hidden_node", class_name=cls, method_name=method,
        kwargs={"images": "IMG"}, hidden=hidden, timeout=60.0,
    )


def test_v1_hidden_arrive_as_kwargs_under_the_authors_spelling(worker):
    # `node_id`, NOT `unique_id` -- upstream matches on the sentinel and does
    # not care what the parameter is called. The old code kept the value only
    # when the author happened to spell it `unique_id`.
    out = _call(worker, "V1SaveNode", "save", [
        ["PROMPT", "prompt", PROMPT],
        ["EXTRA_PNGINFO", "extra_pnginfo", PNGINFO],
        ["UNIQUE_ID", "node_id", "7"],
    ])
    assert out["prompt"] == PROMPT
    assert out["extra_pnginfo"] == PNGINFO
    assert out["node_id"] == "7"


def test_v3_hidden_arrive_on_the_class_not_as_kwargs(worker):
    # A V3 execute() takes no prompt= kwarg; the values must land on a class
    # clone instead. Sent with no parameter name, as the V3 parent proxy does.
    out = _call(worker, "V3SaveNode", "execute", [
        ["PROMPT", None, PROMPT],
        ["EXTRA_PNGINFO", None, PNGINFO],
        ["UNIQUE_ID", None, "7"],
    ])
    assert out["prompt"] == PROMPT
    assert out["extra_pnginfo"] == PNGINFO
    assert out["unique_id"] == "7"


def test_v3_clone_does_not_leak_onto_the_real_class(worker):
    """The clone is per-call. If hidden were set on the real class instead, a
    credential would outlive the call that needed it and be readable by the
    next one -- in a worker that lives for hours."""
    _call(worker, "V3SaveNode", "execute", [["PROMPT", None, PROMPT]])
    leaked = worker.call_module(module="hidden_node_probe", func="real_class_hidden")
    assert leaked is None


def test_no_hidden_is_not_an_error(worker):
    """Every node that declares nothing still has to work."""
    out = _call(worker, "V1SaveNode", "save", None)
    assert out["prompt"] is None and out["node_id"] is None


def test_isolated_save_node_writes_a_png_with_its_workflow(worker, tmp_path):
    """The acceptance test for the whole thing.

    Everything above proves the transport. This proves the outcome: a real
    PNG, written by a node inside a worker, carries the chunks that let you
    drag it back onto the canvas and get the graph that made it.

    It is the assertion that would have caught the original bug. The failure
    was invisible one layer up -- the image saved fine, the right size, the
    right pixels, the right filename. Only the chunks were missing.
    """
    pytest.importorskip("PIL", reason="needs pillow to read the chunks back")
    from PIL import Image

    out = tmp_path / "out.png"
    worker.call_method(
        module_name="png_save_node", class_name="PackSaveImage",
        method_name="save", kwargs={"path": str(out)},
        hidden=[["PROMPT", "prompt", PROMPT],
                ["EXTRA_PNGINFO", "extra_pnginfo", PNGINFO]],
        timeout=60.0,
    )

    import json
    chunks = Image.open(out).text
    assert sorted(chunks) == ["prompt", "workflow"]
    assert json.loads(chunks["prompt"]) == PROMPT
    assert json.loads(chunks["workflow"]) == PNGINFO["workflow"]


def test_isolated_save_node_writes_a_png_with_its_workflow(worker, tmp_path):
    """The acceptance test for the whole thing.

    Everything above proves the transport. This proves the outcome: a real
    PNG, written by a node inside a worker, carries the chunks that let you
    drag it back onto the canvas and get the graph that made it.

    It is the assertion that would have caught the original bug, and the
    reason that bug was invisible one layer up -- the image saved fine, right
    size, right pixels, right filename. Only the chunks were missing.
    """
    import json

    from PIL import Image

    out = tmp_path / "out.png"
    worker.call_method(
        module_name="png_save_node", class_name="PackSaveImage",
        method_name="save", kwargs={"path": str(out)},
        hidden=[["PROMPT", "prompt", PROMPT],
                ["EXTRA_PNGINFO", "extra_pnginfo", PNGINFO]],
        timeout=60.0,
    )

    chunks = Image.open(out).text
    assert sorted(chunks) == ["prompt", "workflow"]
    assert json.loads(chunks["prompt"]) == PROMPT
    assert json.loads(chunks["workflow"]) == PNGINFO["workflow"]
