"""Contract: held values outlive the worker; frames are freed on ack.

The two lifetime guarantees added 2026-08:

1. MATERIALIZE-ON-RECEIPT -- a value this process cannot reconstruct
   (OpaquePayload from the registry, OpaquePickle from the pickle rung) is
   copied into receiver-owned memory the moment it arrives. Holding it for
   minutes, forwarding it after the producing worker RESTARTED, all fine.
   This is the bare-host principle as a test: the parent env has none of
   the pack's libs and never needs them.

2. CONSUMED-ACK -- after the parent reads a reply it tells the worker
   {"type": "consumed", "call_id": N}; the worker frees that call's keeper
   entries immediately. The TTL sweep survives only as a crash fallback,
   so no correctness depends on a timer.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers import WorkerError
from comfy_env.isolation.workers._ipc_shared import OpaquePayload, OpaquePickle
from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"

_ENV = {"COMFY_ENV_SERIALIZER_FILES": ",".join([
    str(FIXTURES / "worker_only_type.py"),
    str(FIXTURES / "worker_only_arr.py"),
])}


def _spawn(name):
    return SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                            env=_ENV, name=name)


@pytest.fixture()
def worker():
    w = _spawn("lifetime-worker")
    yield w
    w.shutdown()


def _crash(worker):
    """Kill the worker process dead (models wrap.py's restart trigger --
    production then builds a NEW SubprocessWorker for the next generation)."""
    with pytest.raises((WorkerError, RuntimeError, TimeoutError)):
        worker.call_module(module="echo_node", func="crash")
    assert not worker.is_alive()


def test_opaque_payload_survives_worker_restart(worker):
    # Parent never imports worker_only_type -> value held as OpaquePayload.
    opaque = worker.call_module(module="echo_node", func="make_worker_only")
    assert isinstance(opaque, OpaquePayload)

    _crash(worker)

    # Forward the receipt to the NEXT-GENERATION worker: with raw frames
    # this was FileNotFoundError on a dead shm name; materialized payloads
    # survive the producer's death.
    replacement = _spawn("lifetime-worker-gen2")
    try:
        assert replacement.call_module(module="echo_node",
                                       func="bump_worker_only",
                                       value=opaque) == 42
    finally:
        replacement.shutdown()


def test_array_bearing_opaque_survives_worker_restart(worker):
    # The production crash shape: the payload contains real shared-memory
    # array frames (mesh vertices/faces in GeometryPack's case).
    opaque = worker.call_module(module="echo_node", func="make_worker_only_arr",
                                n=64)
    assert isinstance(opaque, OpaquePayload)

    _crash(worker)

    replacement = _spawn("lifetime-worker-gen2")
    try:
        expected = float(sum(range(64)))
        assert replacement.call_module(module="echo_node",
                                       func="sum_worker_only_arr",
                                       value=opaque) == expected
    finally:
        replacement.shutdown()


def test_unregistered_pickle_type_held_and_forwarded(worker):
    # No registered serializer -> pickle rung. The parent cannot import
    # pickle_only_type, so loads must degrade to held bytes, not raise.
    held = worker.call_module(module="echo_node", func="make_pickle_only", n=7)
    assert isinstance(held, OpaquePickle)

    _crash(worker)

    # Owned bytes re-emit as a fresh pickle block for the new worker.
    replacement = _spawn("lifetime-worker-gen2")
    try:
        assert replacement.call_module(module="echo_node",
                                       func="bump_pickle_only",
                                       value=held) == 8
    finally:
        replacement.shutdown()


def test_consumed_ack_frees_keepers_immediately(worker):
    # A tensor-bearing reply pins worker memory in the keepers. The
    # consumed-ack must free it as soon as the parent has read the reply --
    # long before the TTL sweep would.
    result = worker.call_module(module="echo_node", func="make_tensor",
                                rows=4, cols=8)
    assert result is not None

    # The ping travels the same socket AFTER the consumed message (FIFO),
    # so the counts it reports are post-release. No sleeps, no races.
    worker._transport.send({"method": "ping"})
    pong = worker._transport.recv(timeout=10)
    assert pong.get("status") == "pong"
    assert pong.get("keepers") == {"tensors": 0, "shm": 0}, (
        f"keeper entries not freed on consumed-ack: {pong.get('keepers')}")
