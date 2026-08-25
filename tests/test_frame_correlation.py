"""Contract: only a correlated reply may be returned as a call's result.

The worker writes three kinds of frame to one socket: replies, async `log`
frames (it replaces builtins.print and installs a handler on the root logger),
and `callback` frames. Every loop that reads a reply must skip the async ones
and must verify the reply is answering THIS call.

Frames the worker emits with no call_id at all -- `ready`, `pool_fd_sent`, and
the reply to a call_id-less ping -- made the old correlation check a no-op,
because it only compared when both ids were present.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import SubprocessWorker

FIXTURES = Path(__file__).parent / "fixtures"


class _ScriptedTransport:
    """Replays queued frames; records what the parent sent back."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.closed = False

    def send(self, obj):
        self.sent.append(obj)

    def recv(self, timeout=None):
        if not self._frames:
            return None
        return self._frames.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="correlation-worker")
    yield w
    w.shutdown()


def test_a_frame_without_a_call_id_is_not_this_calls_reply(worker):
    """The bug: `ready` was returned as the node's result, and read as None."""
    worker._transport = _ScriptedTransport([
        {"status": "ready"},                       # no call_id -- not a reply
        {"type": "pool_fd_sent", "device": 0},     # no call_id -- not a reply
        {"status": "ok", "call_id": 41, "result": "the real answer"},
    ])

    response = worker._send_request({"type": "call_module", "call_id": 41}, timeout=5)

    assert response["result"] == "the real answer", (
        "an uncorrelated frame was returned as the call's reply"
    )


def test_a_reply_for_a_different_call_is_dropped(worker):
    """A late reply from a predecessor must not answer the current call."""
    worker._transport = _ScriptedTransport([
        {"status": "ok", "call_id": 7, "result": "stale"},
        {"status": "ok", "call_id": 8, "result": "current"},
    ])

    response = worker._send_request({"type": "call_module", "call_id": 8}, timeout=5)

    assert response["result"] == "current"


def test_log_frames_do_not_fail_the_health_check(worker):
    """The bug: one log frame ahead of the pong killed a healthy worker.

    The worker forwards every print() and every logging record over this
    socket, from whatever thread emits it. A background thread in pack code
    logging at the wrong moment restarted the worker and dropped every model
    it had loaded.
    """
    worker._transport = _ScriptedTransport([])

    def recv(timeout=None):
        # A log frame arrives first, then the pong for whatever id was pinged.
        ping = next(f for f in worker._transport.sent if f.get("method") == "ping")
        worker._transport.recv = lambda timeout=None: {
            "status": "pong", "call_id": ping["call_id"],
        }
        return {"type": "log", "message": "downloading weights..."}

    worker._transport.recv = recv

    assert worker._check_socket_health() is True, (
        "a healthy worker failed its health check because a log frame "
        "arrived ahead of the pong"
    )


def test_health_check_rejects_an_uncorrelated_pong(worker):
    """A pong left over from an earlier probe is not proof of liveness now."""
    worker._transport = _ScriptedTransport([
        {"status": "pong", "call_id": 999},
    ])

    assert worker._check_socket_health() is False


def test_shutdown_releases_a_worker_that_already_failed(tmp_path):
    """The bug: shutdown() early-returned on exactly the workers it must clean.

    _send_request sets _shutdown on a dead socket or a timeout; the pool then
    calls shutdown() to release the worker. Gating teardown on that same flag
    made the call a no-op, leaking the process, the fd pair and the temp dir
    for the life of the session.
    """
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="crashed-worker")
    temp_dir = Path(w._temp_dir)
    transport = _ScriptedTransport([])
    w._transport = transport

    w._shutdown = True          # what a ConnectionError/timeout leaves behind
    w.shutdown()

    assert transport.closed, "transport was left open on a crashed worker"
    assert w._transport is None
    assert not temp_dir.exists(), f"temp dir leaked: {temp_dir}"


def test_shutdown_is_idempotent(worker):
    """Teardown must still run at most once."""
    worker.shutdown()
    worker.shutdown()   # must not raise
    assert worker._torn_down is True
