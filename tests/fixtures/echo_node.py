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
