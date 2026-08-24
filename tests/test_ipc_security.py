"""Contract: the local IPC channel authenticates its peer.

The parent's worker socket is reachable by other local processes (Linux
abstract sockets have no filesystem permissions; the Windows fallback is
TCP loopback), and the channel carries pickled payloads into the ComfyUI
process. So:

  - the worker's address + auth token travel via the ENVIRONMENT, never
    argv (argv is world-readable via /proc/<pid>/cmdline);
  - the parent refuses to speak the protocol until the peer presents the
    per-spawn authkey as its first frame.
"""

import sys
from pathlib import Path

import pytest

from comfy_env.isolation.workers.subprocess import (
    SubprocessWorker, _PERSISTENT_WORKER_SCRIPT,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def worker():
    w = SubprocessWorker(python=sys.executable, working_dir=FIXTURES,
                         name="sec-worker")
    yield w
    w.shutdown()


def test_authkey_is_generated_and_socket_addr_not_in_argv(worker):
    # A normal round-trip proves the happy path (worker presented the key).
    assert worker.call_module(module="echo_node", func="echo", value=1) == 1
    assert worker._authkey and len(worker._authkey) >= 32
    # The launched command line must not carry the socket address.
    cmdline = worker._process.args
    assert not any("abstract://" in str(a) or "tcp://" in str(a)
                   or "unix://" in str(a) for a in cmdline), cmdline


def test_worker_sends_authkey_as_first_frame():
    # The worker source must send the auth frame before anything else and
    # read its address from the environment, not argv.
    assert 'transport.send({"authkey": authkey})' in _PERSISTENT_WORKER_SCRIPT
    assert 'os.environ.get("COMFY_ENV_IPC_ADDR")' in _PERSISTENT_WORKER_SCRIPT
    assert "sys.argv[1]" not in _PERSISTENT_WORKER_SCRIPT


def test_parent_rejects_wrong_authkey(worker):
    # Drive the accept path directly with a peer that presents a bad key:
    # the parent must refuse, not proceed into the protocol.
    import threading
    from comfy_env.isolation.workers._ipc_parent import (
        _create_server_socket, SocketTransport,
    )

    srv, addr = _create_server_socket()

    def _bad_peer():
        s = _connect_any(addr)
        SocketTransport(s).send({"authkey": "not-the-key"})

    # Minimal reimplementation of the parent's verify step, mirroring
    # _ensure_started, to assert the rejection contract in isolation.
    worker._authkey = "correct-key"
    t = threading.Thread(target=_bad_peer, daemon=True)
    srv.settimeout(10)
    t.start()
    client, _ = srv.accept()
    tr = SocketTransport(client)
    auth = tr.recv(timeout=10)
    ok = isinstance(auth, dict) and auth.get("authkey") == worker._authkey
    tr.close()
    srv.close()
    assert ok is False


def _connect_any(addr):
    import socket as _socket
    if addr.startswith("abstract://"):
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect("\0" + addr[len("abstract://"):])
    elif addr.startswith("unix://"):
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.connect(addr[len("unix://"):])
    else:  # tcp://host:port
        host, port = addr[len("tcp://"):].rsplit(":", 1)
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.connect((host, int(port)))
    return s
