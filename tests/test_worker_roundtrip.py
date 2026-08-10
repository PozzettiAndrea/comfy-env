"""Contract: data sent through a worker comes back equal; failures are loud.

These tests run the REAL worker (_persistent_worker.py) in a subprocess using
the current interpreter -- no pixi env needed. This is the don't-break-
userspace test of the IPC layer: it survives any serialization rewrite and
fails only when the wire contract actually breaks.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker, _PERSISTENT_WORKER_SCRIPT
from comfy_env.isolation.workers import _ipc_shared

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    w = SubprocessWorker(
        python=sys.executable,
        working_dir=FIXTURES,
        name="test-worker",
    )
    yield w
    w.shutdown()


def test_roundtrip_primitives_and_nesting(worker):
    payload = {"a": [1, 2.5, "x", True, None], "b": {"c": "d"}, "empty": []}
    assert worker.call_module(module="echo_node", func="echo", value=payload) == payload


def test_roundtrip_torch_tensor(worker):
    import torch
    result = worker.call_module(module="echo_node", func="make_tensor", rows=4, cols=8)
    expected = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    assert torch.equal(result, expected)


def test_roundtrip_numpy_values(worker):
    import numpy as np
    sent = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = worker.call_module(module="echo_node", func="echo", value=sent)
    got = np.asarray(result)
    assert got.shape == sent.shape
    assert np.array_equal(got, sent)


def test_worker_error_propagates(worker):
    from comfy_env.isolation.workers import WorkerError
    with pytest.raises(WorkerError):
        worker.call_module(module="echo_node", func="no_such_function")


def test_crash_is_loud(worker):
    with pytest.raises(RuntimeError, match="died|closed"):
        worker.call_module(module="echo_node", func="crash")
    assert not worker.is_alive()


def test_timeout_kills_worker(worker):
    with pytest.raises(TimeoutError):
        worker.call_module(module="echo_node", func="slow", seconds=60, timeout=3)


def test_worker_imports_shared_constants():
    # The worker takes the faulthandler filename and retention windows FROM
    # _ipc_shared (single source of truth) -- the hand-synced literals that
    # once drifted (.log vs .txt; 30s vs 60s) are structurally gone.
    assert "import _ipc_shared" in _PERSISTENT_WORKER_SCRIPT
    assert "_ipc_shared.WORKER_FAULTHANDLER_BASENAME" in _PERSISTENT_WORKER_SCRIPT
    assert "_ipc_shared.TENSOR_KEEPER_TTL" in _PERSISTENT_WORKER_SCRIPT
