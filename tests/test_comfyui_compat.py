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

import inspect
from pathlib import Path
import os
import sys

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


def test_model_patcher_surface():
    from comfy.model_patcher import ModelPatcher
    assert callable(getattr(ModelPatcher, "unpatch_model", None))
    # SubprocessModelPatcher passes (model, load_device, offload_device)
    params = list(inspect.signature(ModelPatcher.__init__).parameters)
    for expected in ("model", "load_device", "offload_device"):
        assert expected in params


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
    import comfy.model_management as mm
    for name in ("get_free_memory", "get_total_memory", "get_torch_device",
                 "cleanup_models"):
        assert callable(getattr(mm, name, None)), f"comfy.model_management.{name} gone"
    assert hasattr(mm, "LoadedModel")
    assert isinstance(mm.current_loaded_models, list)
    # setup.py's pool patch replicates these signatures; a new required
    # parameter upstream would break the patched versions.
    assert list(inspect.signature(mm.get_free_memory).parameters)[:2] == ["dev", "torch_free_too"]
    assert list(inspect.signature(mm.get_total_memory).parameters)[:2] == ["dev", "torch_total_too"]
