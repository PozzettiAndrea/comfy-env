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
                                               combos)
  - execution.py validate contract: inputs named in the VALIDATE_INPUTS /
    validate_inputs argspec are exempted     (metadata.py synthesized
                                              named-arg validate)
  - execution.py caching contract: IS_CHANGED / fingerprint_inputs consulted
    once per node per prompt, a raise is NaN, and NaN never equals a stored
    key                                       (metadata.py _forward_fingerprint)

Needs a ComfyUI checkout: set COMFYUI_DIR. Skipped otherwise.
"""

import ast
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.comfyui

COMFYUI_DIR = os.environ.get("COMFYUI_DIR")

if not COMFYUI_DIR:
    pytest.skip("COMFYUI_DIR not set", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _comfyui_importable():
    """Put COMFYUI_DIR on sys.path for this module only, and take it back off.

    Catches the wrong implementation this replaced: a module-level
    ``sys.path.insert(0, COMFYUI_DIR)`` with no teardown. It made ComfyUI
    importable for the WHOLE pytest session, and the two tests below that
    really import leave `comfy` and `folder_paths` in ``sys.modules``. Every
    later test then ran against a process that could import folder_paths --
    which is exactly how comfy_env decides it is running inside a ComfyUI
    server (``environment/cache.py:find_comfyui_source_dir``, first branch).
    Four tests in test_worker_comfyui_base.py, which assert on a walk that
    must find NOTHING, got this checkout's root instead. Restoring sys.path
    alone is not enough: an already-imported module stays importable from
    sys.modules, so the residue has to go too.
    """
    sys.path.insert(0, COMFYUI_DIR)
    before = set(sys.modules)
    try:
        yield
    finally:
        try:
            sys.path.remove(COMFYUI_DIR)
        except ValueError:
            pass
        # Purge by FILE LOCATION, not by name prefix: `comfy` and `comfy_env`
        # share a prefix and only one of them is upstream's.
        root = Path(COMFYUI_DIR).resolve()
        for name in sorted(set(sys.modules) - before):
            mod = sys.modules.get(name)
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            try:
                if Path(f).resolve().is_relative_to(root):
                    del sys.modules[name]
            except (OSError, ValueError):
                pass
        assert "folder_paths" not in sys.modules, (
            "this module left ComfyUI importable for the rest of the session")


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


def test_execution_is_changed_contract():
    """The forwarded fingerprint rests on three upstream facts: the V3 name
    is found by first_real_override, the V1 name by hasattr on the class,
    and a failure (or our miss) is float("NaN"), which caching.py folds into
    a fresh Unhashable so it never equals a stored key. Source-level, for
    the same reason as above."""
    src = (Path(COMFYUI_DIR) / "execution.py").read_text(encoding="utf-8")
    assert 'first_real_override(class_def, "fingerprint_inputs")' in src
    assert 'hasattr(class_def, "IS_CHANGED")' in src
    assert 'node["is_changed"] = float("NaN")' in src
    caching = (Path(COMFYUI_DIR) / "comfy_execution" / "caching.py").read_text(
        encoding="utf-8")
    assert "class Unhashable" in caching


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
