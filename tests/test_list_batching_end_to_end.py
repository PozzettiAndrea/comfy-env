"""Contract: ComfyUI's implicit list batching behaves through a worker as it
does natively, for every shape a node can declare.

The executor (execution.py, _async_map_node_over_list / merge_result_data)
calls a node once per element when it gets a list where it expected one
value, hands the whole list in one call when the class says INPUT_IS_LIST,
and splices a list-valued output into the downstream stream when the
class says OUTPUT_IS_LIST. All three read flags off the class the executor
holds, which under isolation is the proxy. This drives the real executor
against proxies built from a real scan, backed by a real worker, in both
the V1 and the V3 shape (io.Schema is_input_list / Output is_output_list).

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""
import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md
from comfy_env.isolation import pool
from comfy_env.isolation.workers.subprocess import SubprocessWorker

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")
if not COMFYUI_DIR:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures"


def _parse_cpu_args():
    sys.path.insert(0, COMFYUI_DIR)
    import comfy.options
    comfy.options.enable_args_parsing()
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]
    import comfy.cli_args  # noqa: F401
    sys.path.remove(COMFYUI_DIR)


_parse_cpu_args()


@pytest.fixture()
def proxies(tmp_path):
    """Every node in list_nodes.py, scanned for real and proxied, with one
    real warm worker seeded in the pool for the env."""
    sys.path.insert(0, COMFYUI_DIR)
    pkg = tmp_path / "lpack"
    pkg.mkdir()
    shutil.copy(FIXTURES / "list_nodes.py", pkg / "node.py")
    (pkg / "__init__.py").write_text("from .node import *\n", encoding="utf-8")
    env_dir = tmp_path / "env"; (env_dir / "bin").mkdir(parents=True)
    # A wrapper, not a symlink: a venv python located through a symlink
    # loses its pyvenv.cfg and runs as the bare base interpreter.
    launcher = env_dir / "bin" / "python"
    launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    payload = md.fetch_metadata(env_dir, "lpack", tmp_path, env_vars={"COMFYUI_BASE": COMFYUI_DIR})
    built = {}
    for name, meta in payload["nodes"].items():
        built[name] = md.build_proxy_class(node_name=name, meta=dict(meta), env_dir=env_dir,
                                           package_root=tmp_path, sys_path=[COMFYUI_DIR], env_vars={})
    w = SubprocessWorker(python=sys.executable, working_dir=tmp_path,
                         sys_path=[COMFYUI_DIR], name="list-worker")
    w._ensure_started()
    saved = dict(pool._WORKER_POOL)
    pool._WORKER_POOL[str(env_dir)] = pool.WorkerRecord(w, 1)
    try:
        yield built
    finally:
        w.shutdown()
        pool._WORKER_POOL.clear(); pool._WORKER_POOL.update(saved)
        sys.path.remove(COMFYUI_DIR)


def _run(obj, **inputs):
    """What the executor does for one node: map over the inputs, merge."""
    import execution
    obj = obj if isinstance(obj, type) and _is_v3(obj) else obj()
    output, ui, has_subgraph, pending = asyncio.run(
        execution.get_output_data("p1", "n1", obj, inputs))
    assert not pending and not has_subgraph
    return output


def _is_v3(cls):
    from comfy_api.internal import _ComfyNodeInternal
    return issubclass(cls, _ComfyNodeInternal)


def test_scan_carries_both_flags_for_both_shapes(proxies):
    assert proxies["Gather"].INPUT_IS_LIST is True
    assert proxies["Fan"].OUTPUT_IS_LIST == (True,)
    assert getattr(proxies["PerItem"], "INPUT_IS_LIST", False) is False
    assert "GatherV3" in proxies, "the scan did not see the V3 nodes (comfy_api missing in the scan)"
    assert proxies["GatherV3"].INPUT_IS_LIST is True
    assert proxies["FanV3"].OUTPUT_IS_LIST == (True,)


def test_per_item_node_is_called_once_per_element(proxies):
    assert _run(proxies["PerItem"], x=[1, 2, 3]) == [[2, 4, 6]]


@pytest.mark.parametrize("name", ["Gather", "GatherV3"])
def test_input_is_list_node_gets_the_whole_list_in_one_call(proxies, name):
    assert _run(proxies[name], x=[1, 2, 3]) == [[3], ["1,2,3"]]


@pytest.mark.parametrize("name", ["Fan", "FanV3"])
def test_output_is_list_node_fans_out_downstream(proxies, name):
    out = _run(proxies[name], n=[3])
    assert out == [[0, 1, 2]], "a list-valued output must be spliced, not nested"
    # and the stream feeds a per-item node and an INPUT_IS_LIST node natively
    assert _run(proxies["PerItem"], x=out[0]) == [[0, 2, 4]]
    assert _run(proxies["Gather"], x=out[0]) == [[3], ["0,1,2"]]
