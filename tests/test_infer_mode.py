"""Contract: the worker wraps node calls in no_grad, never inference_mode.

Only the node call is wrapped (`with _infer_mode(): result = method(**inputs)`
in _persistent_worker.py), so a model that lazily creates an nn.Parameter on its
first forward creates it INSIDE that context. inference_mode stamps such a
tensor `is_inference` for good, and every later touch from outside the context
raises -- which for us is routine, because SubprocessModelPatcher moves models
between devices and packs apply LoRA patches outside the call.

These pin the semantics; tests/test_infer_mode_source.py pins the default the
worker actually selects, and runs without torch. If a future torch makes
inference_mode safe here, the first test below starts failing and the choice can
be revisited on evidence instead of folklore.
"""

import pytest

torch = pytest.importorskip("torch", reason="these assert torch semantics")
import torch.nn as nn  # noqa: E402


def _param_made_inside(ctx):
    """An nn.Parameter created inside `ctx`, as a lazily-built model would."""
    with ctx():
        return nn.Parameter(torch.zeros(1, 3))


def test_inference_mode_poisons_lazily_created_parameters():
    """The failure this default exists to avoid -- documented, not hypothetical."""
    p = _param_made_inside(torch.inference_mode)
    assert p.is_inference()

    # An autograd-tracked op on it, from outside the context.
    with pytest.raises(RuntimeError, match="[Ii]nference tensor"):
        (p * torch.randn(1, 3, requires_grad=True)).sum().backward()

    # An in-place update from outside the context -- what a device move or a
    # weight patch does.
    with pytest.raises(RuntimeError, match="[Ii]nplace update to inference tensor"):
        with torch.no_grad():
            p.add_(1.0)


def test_no_grad_leaves_lazily_created_parameters_usable():
    p = _param_made_inside(torch.no_grad)
    assert not p.is_inference()

    (p * torch.randn(1, 3, requires_grad=True)).sum().backward()  # must not raise
    with torch.no_grad():
        p.add_(1.0)                                                # must not raise
