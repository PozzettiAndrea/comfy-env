"""Contract: the canary handshake verifies transport by probing reality.

The zero-copy tiers ride torch's private multiprocessing reduction protocol,
which has no cross-version guarantee. Instead of predicting compatibility
from version numbers (a hand-maintained matrix would rot), each worker's
transport is verified at startup by round-tripping a tensor through the
PRODUCTION serialization path. Failing GPU tiers demote to CPU transport,
loudly; a failing CPU tier is a hard error.
"""

import sys
from pathlib import Path

import pytest

import comfy_env.isolation.workers._ipc_parent as ipc_parent
from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="probe-worker")
    yield w
    w.shutdown()


def test_echo_travels_production_path(worker):
    import torch
    canary = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    result, worker_torch = worker.echo(canary=canary, tag="x")
    assert torch.equal(result["canary"].cpu(), canary)
    assert result["tag"] == "x"
    # The worker reports its torch version so family mismatches surface.
    assert worker_torch == torch.__version__  # same interpreter, same torch


def test_verify_transport_passes_on_same_stack(worker, capsys):
    assert worker.verify_transport() is True
    assert worker.gpu_zero_copy_ok is True
    # Same interpreter both sides: no family-mismatch warning.
    assert "torch family mismatch" not in capsys.readouterr().err


def test_torch_family_helper():
    fam = SubprocessWorker._torch_family
    assert fam("2.8.0+cu128") == "2.8"
    assert fam("2.10.1") == "2.10"
    assert fam(None) is None


def test_demotion_flag_routes_cuda_tensors_to_cpu_path():
    """With the demotion flag set, a CUDA tensor must take the CPU
    shared-memory path instead of the zero-copy GPU tiers."""
    import torch

    class FakeCudaTensor:
        is_cuda = True

        def detach(self):
            return torch.arange(4, dtype=torch.float32)

    old = ipc_parent._gpu_zero_copy_demoted
    ipc_parent._gpu_zero_copy_demoted = True
    registry = []
    try:
        meta = ipc_parent._parent_tensor_serializer(FakeCudaTensor(), registry, set())
        # CPU path produces a TensorRef-style payload, never CudaIPC/PoolIPC.
        assert meta.get("__type__") not in ("CudaIPC", "PoolIPC")
    finally:
        ipc_parent._gpu_zero_copy_demoted = old
        ipc_parent._cleanup_shm(registry) if hasattr(ipc_parent, "_cleanup_shm") else None
