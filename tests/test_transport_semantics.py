"""Contract: recv() returns None for a timeout and ONLY for a timeout.

Parent and worker each had their own SocketTransport, and the two had drifted
into opposite contracts wearing one class name:

    parent   None = timed out;    peer closed -> ConnectionError
    worker   None = peer closed;  timed out   -> propagates

Nothing caught it because the usage is disjoint -- the parent always passes a
timeout, the worker never does. A desynced length-prefixed stream hangs rather
than raising, so drift here is expensive to find.
"""

import socket
import struct
import threading

import pytest

from comfy_env.isolation.workers._ipc_shared import (
    MAX_MESSAGE_SIZE,
    SocketTransport,
)


@pytest.fixture()
def pair():
    a, b = socket.socketpair()
    yield a, b
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def test_roundtrip(pair):
    a, b = pair
    SocketTransport(a).send({"hello": "world", "n": 7})
    assert SocketTransport(b).recv(timeout=5) == {"hello": "world", "n": 7}


def test_timeout_returns_none(pair):
    """The ONLY case that may return None."""
    _, b = pair
    assert SocketTransport(b).recv(timeout=0.05) is None


def test_closed_peer_raises_rather_than_returning_none(pair):
    """The drift: the worker returned None here, so 'peer died' and 'nothing
    arrived yet' were indistinguishable to any shared caller."""
    a, b = pair
    a.close()
    with pytest.raises(ConnectionError, match="Socket closed"):
        SocketTransport(b).recv(timeout=5)


def test_truncated_payload_raises_rather_than_reaching_json(pair):
    """The drift: without this check a short read reached json.loads and
    surfaced as JSONDecodeError with no length context."""
    a, b = pair
    a.sendall(struct.pack(">I", 4096) + b'{"partial":')   # header promises more
    a.close()
    with pytest.raises(ConnectionError, match="Incomplete message"):
        SocketTransport(b).recv(timeout=5)


def test_oversize_header_is_refused_before_allocating(pair):
    a, b = pair
    a.sendall(struct.pack(">I", MAX_MESSAGE_SIZE + 1))
    with pytest.raises(ValueError, match="Message too large"):
        SocketTransport(b).recv(timeout=5)


def test_concurrent_sends_do_not_interleave(pair):
    """send() must hold the lock.

    The worker routes every print() and logging record through transport.send
    from whatever thread emits it. Two unlocked sends interleave partial
    writes and desync the stream permanently -- which presents as a hang.
    """
    a, b = pair
    sender, receiver = SocketTransport(a), SocketTransport(b)
    payloads = [{"who": i, "pad": "x" * 3000} for i in range(12)]

    threads = [threading.Thread(target=sender.send, args=(p,)) for p in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    seen = sorted(receiver.recv(timeout=5)["who"] for _ in payloads)
    assert seen == list(range(12)), "frames interleaved -- the stream desynced"


def test_both_sides_use_the_same_implementation():
    """The worker cannot import comfy_env, so this is a source-text check.

    It is the assertion that would have caught the original drift: the worker
    once carried its own copy, and a comment in it claimed parity with the
    parent's while three behaviours differed.
    """
    from pathlib import Path

    import comfy_env.isolation.workers as pkg

    worker_src = (Path(pkg.__file__).parent / "_persistent_worker.py").read_text(
        encoding="utf-8")

    assert "class SocketTransport" not in worker_src, (
        "the worker has its own SocketTransport again -- it drifted three times "
        "the last time that was true"
    )
    assert "_ipc_shared.SocketTransport(" in worker_src
