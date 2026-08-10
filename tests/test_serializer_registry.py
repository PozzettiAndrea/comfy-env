"""Contract: custom node-pack types cross the boundary via the registry.

- A serializer module imported on BOTH sides round-trips real objects.
- A type only the WORKER knows arrives in the parent as an OpaquePayload,
  and survives the pass BACK to the worker intact (opaque round-trip).
- Stale frames with the wrong call_id are dropped, not returned.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker
from comfy_env.isolation.workers._ipc_shared import OpaquePayload

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    w = SubprocessWorker(
        python=sys.executable,
        working_dir=FIXTURES,
        env={"COMFY_ENV_SERIALIZER_MODULES": "custom_type_mod,worker_only_type"},
        name="registry-worker",
    )
    yield w
    w.shutdown()


def test_custom_type_roundtrips_when_both_sides_register(worker):
    import numpy as np
    sys.path.insert(0, str(FIXTURES))
    try:
        import custom_type_mod  # registers ColoredPoint in THIS (parent) process
    finally:
        sys.path.remove(str(FIXTURES))

    result = worker.call_module(module="echo_node", func="make_custom")
    assert type(result).__name__ == "ColoredPoint"
    assert result.color == "teal"
    assert np.allclose(np.asarray(result.xy), np.arange(4, dtype=np.float32))
    # And the parent can send one BACK through the worker unchanged.
    echoed = worker.call_module(module="echo_node", func="echo",
                                value=custom_type_mod.ColoredPoint(
                                    np.ones(3, dtype=np.float32), "red"))
    assert echoed.color == "red"


def test_unknown_type_is_opaque_and_survives_roundtrip(worker):
    # The parent NEVER imports worker_only_type -> the value arrives opaque.
    opaque = worker.call_module(module="echo_node", func="make_worker_only")
    assert isinstance(opaque, OpaquePayload)
    assert opaque.tag == "WorkerOnly"
    # Passing the opaque value back: the worker reconstructs the real type.
    assert worker.call_module(module="echo_node", func="bump_worker_only",
                              value=opaque) == 42


def test_stale_call_id_frames_are_dropped():
    class FakeTransport:
        def __init__(self, frames):
            self.frames = list(frames)
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

        def recv(self, timeout=None):
            return self.frames.pop(0)

        def close(self):
            pass

    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="stale-test")
    try:
        w._transport = FakeTransport([
            {"status": "ok", "call_id": 1, "result": "STALE"},
            {"status": "ok", "call_id": 2, "result": "FRESH"},
        ])
        response = w._send_request({"type": "x", "call_id": 2}, timeout=5)
        assert response["result"] == "FRESH"
    finally:
        w._transport = None
        w.shutdown()
