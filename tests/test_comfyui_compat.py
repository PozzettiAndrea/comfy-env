"""Canary: the exact surface of ComfyUI that comfy-env touches.

comfy-env has no control over upstream ComfyUI. This module asserts only the
attributes/signatures comfy-env actually relies on, so an upstream rename
shows up as a red canary run instead of a user bug report.

Contact surface (keep this list in sync with reality):
  - comfy.cli_args.args.base_directory        (environment/setup.py)
  - folder_paths.base_path                    (setup.py, isolation/wrap.py)
  - comfy.model_patcher.ModelPatcher          (isolation/model_patcher.py)
  - comfy.model_management: get_free_memory, get_total_memory,
    get_torch_device, LoadedModel, current_loaded_models, cleanup_models
                                              (model_patcher.py, wrap.py,
                                               environment/setup.py pool patch)
  - folder_paths.get_input_directory          (isolation/metadata.py dynamic
                                               combos + mtime fingerprint)
  - execution.py validate contract: inputs named in the VALIDATE_INPUTS /
    validate_inputs argspec are exempted     (metadata.py synthesized
                                              named-arg validate)
  - execution.py caching contract: IS_CHANGED / fingerprint_inputs consulted
    once per node per prompt                 (metadata.py mtime fingerprint)

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""

from pathlib import Path
import ast
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")

if COMFYUI_DIR:
    sys.path.insert(0, COMFYUI_DIR)
else:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)


def test_cli_args_base_directory():
    from comfy.cli_args import args
    assert hasattr(args, "base_directory")


def test_folder_paths_base_path():
    import folder_paths
    assert isinstance(folder_paths.base_path, str)


def _params_of(path, cls, func):
    """Parameter names of a method, read from source.

    Source level on purpose. Importing comfy.model_patcher drags in torch,
    tqdm and comfy_aimdo, and this lane runs on a CPU only hosted runner
    where at least one of those is always missing. Three consecutive weekly
    canary runs failed on exactly that and nobody read them, which makes a
    red canary worth nothing. What is being checked here is a SIGNATURE, and
    a signature is in the text.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == func:
                    a = sub.args
                    return [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    return None


def test_model_patcher_surface():
    """The three methods the stand-in mirrors, and their argument names.

    Deliberately NOT ModelPatcher.__init__: comfy-env stopped subclassing in
    0.4.22, and test_model_patcher_surface.py now asserts that it must never
    start again, so pinning the constructor here pinned a coupling the rest
    of the suite forbids.
    """
    mp = Path(COMFYUI_DIR) / "comfy" / "model_patcher.py"
    for name, expected in (
        ("partially_load", ("device_to", "extra_memory")),
        ("partially_unload", ("device_to", "memory_to_free")),
        ("detach", ("unpatch_all",)),
    ):
        params = _params_of(mp, "ModelPatcher", name)
        assert params is not None, f"ModelPatcher.{name} is gone"
        for arg in expected:
            assert arg in params, (
                f"ModelPatcher.{name} no longer takes {arg!r}; the stand-in "
                f"mirrors this signature in isolation/model_patcher.py")


def test_folder_paths_input_directory():
    import folder_paths
    assert callable(folder_paths.get_input_directory)


def test_execution_validate_exemption_contract():
    """Synthesized named-arg validate relies on execution.py exempting inputs
    named in the validate argspec. Source-level canary (read the file, not
    import it -- importing execution drags in torch/model_management)."""
    src = (Path(COMFYUI_DIR) / "execution.py").read_text(encoding="utf-8")
    assert "validate_function_inputs" in src
    assert "validate_has_kwargs" in src


def test_model_management_surface():
    """Same reasoning as above: read it, do not import it."""
    src = (Path(COMFYUI_DIR) / "comfy" / "model_management.py").read_text(
        encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    for name in ("get_free_memory", "get_total_memory", "get_torch_device",
                 "cleanup_models", "free_memory", "load_models_gpu",
                 "unload_all_models"):
        assert name in funcs, f"comfy.model_management.{name} gone"
    assert "LoadedModel" in classes
    assert "current_loaded_models" in src
    # comfy-env reproduces these two signatures when it reads free memory; a
    # new leading parameter upstream would silently shift the arguments.
    for name, expected in (("get_free_memory", ["dev", "torch_free_too"]),
                           ("get_total_memory", ["dev", "torch_total_too"])):
        a = funcs[name].args
        got = [p.arg for p in a.posonlyargs + a.args][:2]
        assert got == expected, f"{name}{tuple(got)} moved its first two args"
