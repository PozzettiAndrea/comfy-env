"""Contract: the subprocess model proxy matches what ComfyUI actually touches.

`SubprocessModelPatcher` is deliberately NOT a `ModelPatcher` subclass -- it is
a duck-type implementing only the members ComfyUI's memory manager reads off
`LoadedModel.model`. That buys a loud failure instead of silent corruption when
upstream drifts, but only if something watches the boundary.

This module is that watch. The canary greps the installed ComfyUI for every
`.model.<name>` access in `model_management.py` and asserts the proxy covers
it. When ComfyUI starts touching something new, CI fails here with the name --
which is the tripwire that stands in for an upstream interface contract we do
not have.

Skips cleanly when ComfyUI is not importable/locatable.
"""

import re
from pathlib import Path

import pytest


def _comfyui_model_management() -> "Path | None":
    """Locate ComfyUI's model_management.py without importing torch."""
    try:
        import comfy.model_management as mm  # noqa: F401
        return Path(mm.__file__)
    except Exception:
        pass
    for env in ("COMFYUI_BASE", "COMFYUI_PATH"):
        import os
        base = os.environ.get(env)
        if base and (Path(base) / "comfy" / "model_management.py").is_file():
            return Path(base) / "comfy" / "model_management.py"
    for guess in (Path("D:/geometrypack/ComfyUI"), Path.home() / "ComfyUI"):
        p = guess / "comfy" / "model_management.py"
        if p.is_file():
            return p
    return None


def _proxy_surface():
    """COMFY_SURFACE without importing comfy (the module imports comfy at top)."""
    src = (Path(__file__).parent.parent / "src" / "comfy_env" / "isolation"
           / "model_patcher.py").read_text(encoding="utf-8")
    block = re.search(r"COMFY_SURFACE\s*=\s*frozenset\(\{(.*?)\}\)", src, re.S)
    assert block, "COMFY_SURFACE not found in model_patcher.py"
    return set(re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', block.group(1)))


def _implemented_names():
    """Attributes/methods the proxy actually defines (source-level, no import)."""
    src = (Path(__file__).parent.parent / "src" / "comfy_env" / "isolation"
           / "model_patcher.py").read_text(encoding="utf-8")
    cls = src.split("class SubprocessModelPatcher", 1)[1]
    names = set(re.findall(r"^    def ([a-zA-Z_][a-zA-Z0-9_]*)", cls, re.M))
    names |= set(re.findall(r"^        self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*=", cls, re.M))
    names |= set(re.findall(r"^    ([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*[a-zA-Z_]", cls, re.M))
    return names


def test_declared_surface_is_implemented():
    """Every name in COMFY_SURFACE must actually exist on the proxy."""
    missing = sorted(_proxy_surface() - _implemented_names())
    assert not missing, (
        f"COMFY_SURFACE declares names the proxy does not implement: {missing}")


def test_canary_comfyui_touches_nothing_new():
    """CI tripwire: ComfyUI must not read a patcher member we do not provide.

    If this fails, upstream changed what it expects of a loaded model. Add the
    member to SubprocessModelPatcher and to COMFY_SURFACE -- do NOT go back to
    inheriting ModelPatcher to make it pass.
    """
    mm_path = _comfyui_model_management()
    if mm_path is None:
        pytest.skip("ComfyUI not available")
    src = mm_path.read_text(encoding="utf-8", errors="replace")

    touched = set(re.findall(r"\.model\.([a-zA-Z_][a-zA-Z0-9_]*)", src))

    # Names that are NOT patcher members:
    #   model            -> the inner nn.Module stand-in (SubprocessModel)
    #   dynamic_pins     -> reached only via is_dynamic(), which we return False for
    #   __class__ etc.   -> python builtins
    not_patcher_members = {"model", "dynamic_pins", "__class__"}
    # Guarded by `is_dynamic()`; our False excludes these paths entirely.
    dynamic_only = {"loaded_ram_size", "pinned_memory_size", "partially_unload_ram"}

    expected = touched - not_patcher_members - dynamic_only
    missing = sorted(expected - _proxy_surface())
    assert not missing, (
        f"ComfyUI ({mm_path}) reads patcher member(s) the proxy does not "
        f"declare: {missing}. Implement them and extend COMFY_SURFACE.")


def test_proxy_is_not_a_modelpatcher_subclass():
    """The whole point: no inheritance from comfy internals."""
    src = (Path(__file__).parent.parent / "src" / "comfy_env" / "isolation"
           / "model_patcher.py").read_text(encoding="utf-8")
    assert "class SubprocessModelPatcher:" in src, (
        "SubprocessModelPatcher must stay a standalone duck-type -- inheriting "
        "ModelPatcher silently re-imports ~120 members that are wrong for an "
        "object holding no weights.")
    assert "import comfy.model_patcher" not in src


def test_is_dynamic_is_false():
    """is_dynamic() False is load-bearing: it excludes every dynamic-pin path,
    which is where most upstream churn lives."""
    src = (Path(__file__).parent.parent / "src" / "comfy_env" / "isolation"
           / "model_patcher.py").read_text(encoding="utf-8")
    body = src.split("def is_dynamic", 1)[1].split("def ", 1)[0]
    assert "return False" in body
