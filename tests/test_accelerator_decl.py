"""Contract: ACCELERATOR node declarations -- capture, degrade, import rule.

The rule (v1): a node declares at most ONE accelerator (or none), meaning
"requires this backend at execution". Accelerator packages must be imported
lazily inside declaring nodes; a top-level import is flagged by the scan.
On machines lacking the backend the node registers VISIBLY and raises a
named-reason error when executed.
"""

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import comfy_env.isolation.metadata as md


def _write_fixture_pkg(root: Path) -> None:
    pkg = root / "fixture_pkg"
    pkg.mkdir(parents=True)
    # A fake accelerator package, importable from working_dir
    (root / "fake_cumesh.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "cpu_node.py").write_text(textwrap.dedent("""
        class CpuNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"x": ("INT", {"default": 0})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
            CATEGORY = "test"
            def run(self, x):
                return (x,)
    """), encoding="utf-8")
    (pkg / "cuda_node.py").write_text(textwrap.dedent("""
        import fake_cumesh  # DELIBERATE top-level accelerator import (violation)

        class CudaNode:
            ACCELERATOR = "cuda"
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"x": ("INT", {"default": 0})}}
            RETURN_TYPES = ("INT",)
            FUNCTION = "run"
            CATEGORY = "test"
            def run(self, x):
                return (x,)
    """), encoding="utf-8")
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        from .cpu_node import CpuNode
        from .cuda_node import CudaNode
        NODE_CLASS_MAPPINGS = {"CpuNode": CpuNode, "CudaNode": CudaNode}
        NODE_DISPLAY_NAME_MAPPINGS = {}
    """), encoding="utf-8")


def test_scan_captures_accelerator_and_flags_toplevel_import(tmp_path):
    _write_fixture_pkg(tmp_path)
    script = tmp_path / "scan.py"
    script.write_text(md._METADATA_SCRIPT, encoding="utf-8")
    out = tmp_path / "payload.pkl"

    env = dict(os.environ)
    env["COMFY_ENV_ACCEL_PKGS"] = "fake-cumesh"  # dist-style name, dash variant
    proc = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "fixture_pkg", str(out)],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = pickle.loads(out.read_bytes())

    nodes = payload["nodes"]
    assert nodes["CpuNode"]["accelerator"] is None
    assert nodes["CudaNode"]["accelerator"] == "cuda"
    # The top-level `import fake_cumesh` must be observed as a violation.
    assert payload["accel_import_violations"] == ["fake_cumesh"]


def test_accelerator_availability_logic(monkeypatch):
    monkeypatch.setattr(md, "_MACHINE_BACKEND", "cpu")
    assert md._accelerator_available(None)
    assert not md._accelerator_available("cuda")
    assert not md._accelerator_available("gpu")

    monkeypatch.setattr(md, "_MACHINE_BACKEND", "cuda")
    assert md._accelerator_available("cuda")
    assert md._accelerator_available("gpu")
    assert not md._accelerator_available("mps")


def test_unavailable_node_is_visible_and_raises_named_reason(monkeypatch):
    monkeypatch.setattr(md, "_MACHINE_BACKEND", "cpu")
    meta = {
        "accelerator": "cuda",
        "function": "run",
        "category": "test",
        "class_name": "CudaNode",
        "module_name": "fixture_pkg.cuda_node",
        "return_types": ("INT",),
        "return_names": ("x",),
        "output_node": False,
        "input_types": {"required": {"x": ("INT", {"default": 0})}},
    }
    cls = md.build_proxy_class(
        node_name="CudaNode", meta=meta, env_dir=Path("unused"),
        package_root=Path("unused"), sys_path=[], lib_path=None, env_vars={},
    )
    # Registered with real inputs/outputs so workflows still load...
    assert cls.INPUT_TYPES() == meta["input_types"]
    assert cls.RETURN_TYPES == ("INT",)
    assert cls._comfy_env_accelerator == "cuda"
    # ...but hidden from the node picker (ADR-0012: DEPRECATED hides from
    # menu/search without unregistering the type).
    assert cls.DEPRECATED is True
    assert "requires CUDA" in cls.DESCRIPTION or "unavailable" in cls.DESCRIPTION
    # Executing raises the named reason, before any worker is spawned.
    with pytest.raises(RuntimeError, match="requires CUDA.*backend 'cpu'"):
        cls().run(x=1)


def test_available_accelerator_builds_normal_proxy(monkeypatch):
    # On a cuda machine a cuda-tagged node must NOT get the stub.
    monkeypatch.setattr(md, "_MACHINE_BACKEND", "cuda")
    meta = {
        "accelerator": "cuda",
        "function": "run",
        "category": "test",
        "class_name": "CudaNode",
        "module_name": "fixture_pkg.cuda_node",
        "return_types": ("INT",),
        "input_types": {"required": {"x": ("INT", {"default": 0})}},
    }
    cls = md.build_proxy_class(
        node_name="CudaNode", meta=meta, env_dir=Path("unused"),
        package_root=Path("unused"), sys_path=[], lib_path=None, env_vars={},
    )
    assert not hasattr(cls, "_comfy_env_unavailable")
    assert cls._comfy_env_accelerator == "cuda"
