"""Contract: the metadata scan payload is strict JSON, failing loudly and
locally (2026-08 review, phase 2).

Runs the REAL _METADATA_SCRIPT as a subprocess on fixture packs, exactly as
fetch_metadata does -- the converter under test is the one that ships.
"""

import json
import os
import subprocess
import sys
import textwrap

import comfy_env.isolation.metadata as md


def _run_scan(tmp_path, module_body, pkg_name="jfix_pkg"):
    pkg = tmp_path / pkg_name
    pkg.mkdir()
    (pkg / "node.py").write_text(textwrap.dedent(module_body), encoding="utf-8")
    (pkg / "__init__.py").write_text(
        "from .node import *\n", encoding="utf-8")
    script = tmp_path / "scan.py"
    script.write_text(md._METADATA_SCRIPT, encoding="utf-8")
    out = tmp_path / "payload.json"
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path), pkg_name, str(out)],
        env=dict(os.environ), capture_output=True, text=True, timeout=120,
    )
    return proc, out


def test_enum_default_drops_the_key_not_the_node(tmp_path):
    """The motivating bug: an Enum default used to ride the pickle and kill
    the WHOLE PACK in the parent (ModuleNotFoundError escaped the caught
    tuple). Now: the key is dropped, the node survives, the warning names
    the exact path."""
    proc, out = _run_scan(tmp_path, """
        import enum
        class Mode(enum.Enum):
            A = "a"
        class GoodNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"x": ("INT", {"default": 0})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
        class EnumNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"m": (["a"], {"default": Mode.A})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"GoodNode": GoodNode, "EnumNode": EnumNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["nodes"]) == {"GoodNode", "EnumNode"}, "no node vanishes"
    m_opts = payload["nodes"]["EnumNode"]["input_types"]["required"]["m"]
    assert "default" not in m_opts[1], "the offending KEY is dropped"
    warns = "\n".join(payload.get("sanitize_warnings", []))
    assert "EnumNode" in warns and "Mode" in warns and "default" in warns


def test_nonfinite_float_dropped_with_path(tmp_path):
    proc, out = _run_scan(tmp_path, """
        class InfNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"v": ("FLOAT", {"default": 0.0,
                                                     "max": float("inf")})}}
            RETURN_TYPES = ("FLOAT",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"InfNode": InfNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    opts = payload["nodes"]["InfNode"]["input_types"]["required"]["v"][1]
    assert "max" not in opts and opts["default"] == 0.0
    assert any("non-finite" in w for w in payload["sanitize_warnings"])


def test_tuples_become_lists_and_str_subclass_coerces(tmp_path):
    proc, out = _run_scan(tmp_path, """
        class Any(str):
            def __ne__(self, other):
                return False
        class TupNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"c": (("a", "b"), {"default": "a"}),
                                     "w": (Any("*"),)}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"TupNode": TupNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    req = payload["nodes"]["TupNode"]["input_types"]["required"]
    assert req["c"][0] == ["a", "b"]          # tuple -> list
    assert req["w"][0] == "*"                 # str subclass -> plain str
    assert type(req["w"][0]) is str


def test_non_string_dict_key_dropped_not_stringified(tmp_path):
    """stdlib json silently coerces {1: x} to {"1": x} -- the one place JSON
    is lossy without telling you. We drop the key loudly instead."""
    proc, out = _run_scan(tmp_path, """
        class KeyNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"x": ("INT", {"default": 0,
                                                   "weird": {1: "one"}})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"KeyNode": KeyNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    weird = payload["nodes"]["KeyNode"]["input_types"]["required"]["x"][1]["weird"]
    assert weird == {} and "1" not in weird
    assert any("non-string dict key" in w for w in payload["sanitize_warnings"])


def test_non_ascii_roundtrips_utf8(tmp_path):
    proc, out = _run_scan(tmp_path, """
        class UniNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"t": ("STRING",
                        {"default": "±2° — ≈ok"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"UniNode": UniNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    raw = out.read_bytes()
    assert "±2°".encode("utf-8") in raw, "ensure_ascii=False + utf-8 on disk"
    payload = json.loads(out.read_text(encoding="utf-8"))
    d = payload["nodes"]["UniNode"]["input_types"]["required"]["t"][1]["default"]
    assert d == "±2° — ≈ok"


def test_numpy_free_int_subclass_coerces(tmp_path):
    proc, out = _run_scan(tmp_path, """
        import enum
        class Level(enum.IntEnum):
            HIGH = 3
        class IntEnumNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"l": ("INT", {"default": Level.HIGH})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
        NODE_CLASS_MAPPINGS = {"IntEnumNode": IntEnumNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    d = payload["nodes"]["IntEnumNode"]["input_types"]["required"]["l"][1]["default"]
    assert d == 3 and type(d) is int
