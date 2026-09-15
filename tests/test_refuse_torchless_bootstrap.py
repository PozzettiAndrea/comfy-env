"""install() refuses an interpreter with no torch.

A Python without torch is never a ComfyUI's -- ComfyUI imports torch
unconditionally, and a CPU-only ComfyUI still has a CPU torch. So a
torchless bootstrap can only mean install.py was run with the wrong
interpreter. The env that used to come out of it was keyed py3XX-notorch and
stamped for a host with no torch, which no real host is, so nothing could
ever bind it: a gigabyte of the wrong torch in a directory nothing uses.
"""
import sys

import pytest

from comfy_env.install import workspace
import comfy_env.detection as det           # the package re-exports; workspace imports from here
from comfy_env.detection import cuda as det_cuda


def test_a_bootstrap_without_torch_is_refused_before_touching_disk(monkeypatch, tmp_path):
    """Catches: the old fallback that logged 'will rely on cuda-wheels
    resolver to pick a combo' and went on to build an unbindable env."""
    monkeypatch.setattr(det, "get_bootstrap_torch_version", lambda: None)
    monkeypatch.setattr(det, "get_bootstrap_python_version", lambda: "3.12")
    monkeypatch.setattr(det, "get_bootstrap_torch_cuda", lambda: None)
    seen = []
    with pytest.raises(RuntimeError) as e:
        workspace._resolve_workspace_torch(seen.append)
    msg = str(e.value)
    assert "has no torch" in msg and "not a ComfyUI's" in msg
    assert "main.py" in msg, "the message must say what to run instead"
    assert sys.executable in msg, "and name the interpreter that was used"
    assert not any("cuda-wheels resolver" in m for m in seen), "the old fallback line must be gone"
    assert not list(tmp_path.iterdir()), "nothing may be written before the refusal"


def test_a_bootstrap_with_torch_is_not_refused(monkeypatch):
    """The guard must key on torch's ABSENCE only. Catches: a guard that
    fires on a CPU torch, or on a torch whose CUDA tag is unset."""
    monkeypatch.setattr(det, "get_bootstrap_torch_version", lambda: "2.8.0")
    monkeypatch.setattr(det, "get_bootstrap_python_version", lambda: "3.13")
    monkeypatch.setattr(det, "get_bootstrap_torch_cuda", lambda: None)   # a CPU torch
    monkeypatch.setattr(det_cuda, "has_nvidia_gpu", lambda: False)   # this one is imported from .cuda
    seen = []
    result = workspace._resolve_workspace_torch(seen.append)
    assert result[3] == "3.13" and result[4] == "2.8.0"
    assert any("torch 2.8.0" in m for m in seen)
