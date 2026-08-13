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
    from worker_only_class import WorkerOnly
    return WorkerOnly(secret=41)


def bump_worker_only(value=None):
    """Assert the opaque payload came back as a real WorkerOnly here."""
    from worker_only_class import WorkerOnly
    assert isinstance(value, WorkerOnly), f"got {type(value).__name__}"
    return value.secret + 1


def make_worker_only_arr(n=64):
    import numpy as np
    from worker_only_class import WorkerOnlyArr
    return WorkerOnlyArr(np.arange(n, dtype=np.float32))


def sum_worker_only_arr(value=None):
    """Assert the array-bearing opaque payload reconstructs here."""
    import numpy as np
    from worker_only_class import WorkerOnlyArr
    assert isinstance(value, WorkerOnlyArr), f"got {type(value).__name__}"
    return float(np.asarray(value.data).sum())


def make_unpicklable():
    return lambda: 1  # lambdas don't pickle -> exercises the loud error


def make_pickle_only(n=7):
    from pickle_only_type import PickleOnly
    return PickleOnly(n)


def bump_pickle_only(value=None):
    """Assert the held pickle bytes unpickled back into the real class."""
    from pickle_only_type import PickleOnly
    assert isinstance(value, PickleOnly), f"got {type(value).__name__}"
    return value.n + 1
