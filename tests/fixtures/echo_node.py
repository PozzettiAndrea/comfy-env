"""Fixture module imported INSIDE the worker subprocess (via sys_path)."""

import os
import time


def echo(value=None):
    return value


def make_tensor(rows=4, cols=8):
    import torch
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols)


def crash():
    os._exit(17)


def slow(seconds=30):
    time.sleep(seconds)
    return "done"


def make_custom():
    import numpy as np
    from custom_type_mod import ColoredPoint
    return ColoredPoint(np.arange(4, dtype=np.float32), "teal")


def make_worker_only():
    from worker_only_type import WorkerOnly
    return WorkerOnly(secret=41)


def bump_worker_only(value=None):
    """Assert the opaque payload came back as a real WorkerOnly here."""
    from worker_only_type import WorkerOnly
    assert isinstance(value, WorkerOnly), f"got {type(value).__name__}"
    return value.secret + 1
