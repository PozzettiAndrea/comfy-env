"""Contract: what upstream reads off a V1 node class reaches the proxy.

The V1 proxy carried seven attributes; the V3 proxy carried seventeen. For
~155 corpus packs that meant no DESCRIPTION in the help panel, deprecated
nodes back in the menu, search aliases that matched nothing; for one node
(impact-pack's BlackPatchRetryHookProvider) a dropped NOT_IDEMPOTENT meant
two copies in one graph shared a cache entry. The scan now sweeps every
UPPERCASE JSON-shaped class attribute instead of maintaining a list, which
had already drifted twice.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

import comfy_env.isolation.metadata as md


def _scan(tmp_path, body, pkg="attrs_pkg"):
    p = tmp_path / pkg
    p.mkdir()
    (p / "node.py").write_text(textwrap.dedent(body), encoding="utf-8")
    (p / "__init__.py").write_text("from .node import *\n", encoding="utf-8")
    script = tmp_path / "scan.py"
    script.write_text(md._METADATA_SCRIPT, encoding="utf-8")
    out = tmp_path / "payload.json"
    proc = subprocess.run([sys.executable, str(script), str(tmp_path), pkg, str(out)],
                          env=dict(os.environ), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))


NODE = """
    class _RaisingProp:
        def __get__(self, obj, cls):
            raise RuntimeError("a V3 classproperty calling GET_SCHEMA")

    class Flags:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"x": ("INT", {"default": 0})}}
        RETURN_TYPES = ("INT",)
        RETURN_NAMES = ("n",)
        OUTPUT_TOOLTIPS = ("the number",)
        FUNCTION = "run"
        CATEGORY = "test"
        DESCRIPTION = "does things"
        DEPRECATED = True
        EXPERIMENTAL = False
        SEARCH_ALIASES = ["thing", "doer"]
        NOT_IDEMPOTENT = True
        FOO_FLAG_UPSTREAM_ADDS_TOMORROW = "carried"
        BIG_TABLE = list(range(10_000))       # data, not a flag
        HELPER = staticmethod(lambda: None)   # callable, not a flag
        BROKEN = _RaisingProp()               # raising classproperty
        lowercase = "not swept"
        def run(self, x=0):
            return (x,)
    NODE_CLASS_MAPPINGS = {"Flags": Flags}
    NODE_DISPLAY_NAME_MAPPINGS = {}
"""


def test_upstream_read_attributes_reach_the_proxy(tmp_path):
    payload = _scan(tmp_path, NODE)
    meta = payload["nodes"]["Flags"]
    Proxy = md.build_proxy_class(node_name="Flags", meta=meta, env_dir=tmp_path,
                                 package_root=tmp_path, sys_path=[], env_vars={})
    assert Proxy.DESCRIPTION == "does things"
    assert list(Proxy.OUTPUT_TOOLTIPS) == ["the number"]
    assert Proxy.DEPRECATED is True and Proxy.EXPERIMENTAL is False
    assert Proxy.SEARCH_ALIASES == ["thing", "doer"]
    assert Proxy.NOT_IDEMPOTENT is True
    assert Proxy.FOO_FLAG_UPSTREAM_ADDS_TOMORROW == "carried", "the sweep, not a list"


def test_the_sweep_is_bounded_and_guarded(tmp_path):
    payload = _scan(tmp_path, NODE)
    swept = payload["nodes"]["Flags"]["class_attrs"]
    assert "BIG_TABLE" not in swept, "a 10k-element list is data, not a flag"
    assert "HELPER" not in swept
    assert "BROKEN" not in swept, "a raising attribute must not kill the scan"
    assert "lowercase" not in swept
    assert "INPUT_TYPES" not in swept


def test_the_builder_wins_over_the_sweep(tmp_path):
    """FUNCTION, CATEGORY and friends are set by the builder from their own
    meta fields; the sweep must never overwrite them."""
    payload = _scan(tmp_path, NODE)
    meta = payload["nodes"]["Flags"]
    meta["class_attrs"]["CATEGORY"] = "poisoned"
    Proxy = md.build_proxy_class(node_name="Flags", meta=meta, env_dir=tmp_path,
                                 package_root=tmp_path, sys_path=[], env_vars={})
    assert Proxy.CATEGORY == "test"


COMFYUI_DIR = os.environ.get("COMFYUI_DIR")


@pytest.mark.comfyui
@pytest.mark.skipif(not COMFYUI_DIR, reason="COMFYUI_DIR not set")
def test_canary_every_attribute_node_info_reads_is_sweepable():
    """If upstream's /object_info starts reading a new class attribute, it
    must be one the sweep would carry. The only names allowed to be missing
    from the sweep are the ones the builder sets itself."""
    import ast
    from pathlib import Path
    src = (Path(COMFYUI_DIR) / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "node_info")
    names = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "obj_class":
            names.add(n.attr)
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "hasattr")
                and len(n.args) >= 2 and isinstance(n.args[0], ast.Name) and n.args[0].id == "obj_class"
                and isinstance(n.args[1], ast.Constant)):
            names.add(n.args[1].value)
    upper = {x for x in names if x.isupper()}
    builder_sets = {"RETURN_TYPES", "RETURN_NAMES", "FUNCTION", "CATEGORY", "OUTPUT_NODE",
                    "OUTPUT_IS_LIST", "INPUT_IS_LIST", "INPUT_TYPES"}
    # ComfyUI assigns this one onto every node class in load_custom_node,
    # proxies included, so the scan must not carry the worker's value.
    comfyui_sets = {"RELATIVE_PYTHON_MODULE"}
    nodes_src = (Path(COMFYUI_DIR) / "nodes.py").read_text(encoding="utf-8")
    assert "node_cls.RELATIVE_PYTHON_MODULE =" in nodes_src, "load_custom_node no longer sets it; the sweep exclusion is stale"
    skipped_by_sweep = {"INPUT_TYPES", "RELATIVE_PYTHON_MODULE"}
    not_carried = (upper & skipped_by_sweep) - builder_sets - comfyui_sets
    assert not not_carried, f"node_info reads {sorted(not_carried)}, which neither the builder nor the sweep carries"
    assert {"DESCRIPTION", "OUTPUT_TOOLTIPS", "SEARCH_ALIASES"} <= upper, "the AST walk lost its bearings"
