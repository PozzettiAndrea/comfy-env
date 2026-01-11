"""
Transport Layer - Pluggable IPC transports for host-worker communication.

Supports:
- UnixSocketTransport: Unix Domain Sockets with length-prefixed JSON (recommended)
- StdioTransport: Legacy stdin/stdout JSON lines (fallback)

The transport abstraction allows swapping IPC mechanisms without changing
the serialization protocol.
"""

import json
import os
import socket
import struct
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Protocol for IPC transport mechanisms."""

    def send(self, obj: Any) -> None:
        """Send a JSON-serializable object to the remote endpoint."""
        ...

    def recv(self) -> Any:
        """Receive a JSON object from the remote endpoint. Blocks until available."""
        ...

    def close(self) -> None:
        """Close the transport. Further send/recv calls may fail."""
        ...


class UnixSocketTransport:
    """
    Transport using Unix Domain Sockets with length-prefixed JSON messages.

    This is the recommended transport as it:
    - Doesn't interfere with stdout/stderr (no C library output issues)
    - Supports binary-safe length-prefixed framing
    - Is more efficient than line-based JSON

    Message format: [4-byte big-endian length][JSON payload]
    """

    def __init__(self, sock: socket.socket):
        """
        Initialize with an already-connected socket.

        Args:
            sock: Connected Unix domain socket
        """
        self._sock = sock
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()

    @classmethod
    def create_server(cls, sock_path: str) -> tuple[socket.socket, "UnixSocketTransport"]:
        """
        Create a server socket, wait for one connection, return transport.

        Args:
            sock_path: Path for the Unix domain socket

        Returns:
            Tuple of (server_socket, transport) - caller should close server_socket
        """
        # Remove existing socket file if present
        if os.path.exists(sock_path):
            os.unlink(sock_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(sock_path)
        server.listen(1)

        conn, _ = server.accept()
        return server, cls(conn)

    @classmethod
    def connect(cls, sock_path: str, timeout: float = 30.0) -> "UnixSocketTransport":
        """
        Connect to an existing Unix domain socket.

        Args:
            sock_path: Path to the Unix domain socket
            timeout: Connection timeout in seconds

        Returns:
            Connected transport
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect(sock_path)
        sock.settimeout(None)  # Back to blocking mode
        return cls(sock)

    def send(self, obj: Any) -> None:
        """Send a JSON-serializable object with length prefix."""
        data = json.dumps(obj).encode('utf-8')
        header = struct.pack('>I', len(data))

        with self._send_lock:
            self._sock.sendall(header + data)

    def recv(self) -> Any:
        """Receive a length-prefixed JSON message."""
        with self._recv_lock:
            # Read 4-byte length header
            header = self._recvall(4)
            if not header or len(header) < 4:
                raise ConnectionError("Socket closed or incomplete header")

            msg_len = struct.unpack('>I', header)[0]

            # Sanity check - 100MB limit
            if msg_len > 100 * 1024 * 1024:
                raise ValueError(f"Message too large: {msg_len} bytes")

            # Read payload
            data = self._recvall(msg_len)
            if len(data) < msg_len:
                raise ConnectionError(f"Incomplete message: {len(data)}/{msg_len} bytes")

            return json.loads(data.decode('utf-8'))

    def _recvall(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(min(remaining, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def close(self) -> None:
        """Close the socket."""
        try:
            self._sock.close()
        except Exception:
            pass

    def fileno(self) -> int:
        """Return socket file descriptor for select()."""
        return self._sock.fileno()


class StdioTransport:
    """
    Legacy transport using stdin/stdout with JSON lines.

    This transport has issues with C libraries that print to stdout,
    which is why UnixSocketTransport is preferred. Kept for compatibility.

    Note: Requires fd-level redirection during method execution to prevent
    C library output from corrupting the JSON stream.
    """

    def __init__(
        self,
        stdin: Any = None,
        stdout: Any = None,
    ):
        """
        Initialize with file handles.

        Args:
            stdin: Input stream (default: sys.stdin)
            stdout: Output stream (default: sys.stdout)
        """
        self._stdin = stdin or sys.stdin
        self._stdout = stdout or sys.stdout
        self._send_lock = threading.Lock()

    def send(self, obj: Any) -> None:
        """Send a JSON object as a single line."""
        with self._send_lock:
            line = json.dumps(obj) + '\n'
            self._stdout.write(line)
            self._stdout.flush()

    def recv(self) -> Any:
        """Receive a JSON object from a single line."""
        line = self._stdin.readline()
        if not line:
            raise ConnectionError("stdin closed")
        return json.loads(line.strip())

    def close(self) -> None:
        """No-op for stdio transport."""
        pass


def get_socket_path(env_name: str, pid: Optional[int] = None) -> str:
    """
    Generate a unique socket path for an isolated environment.

    Args:
        env_name: Name of the isolated environment
        pid: Process ID (default: current process)

    Returns:
        Path string for Unix domain socket
    """
    if pid is None:
        pid = os.getpid()

    # Use /tmp on Linux, or a temp directory on other platforms
    if sys.platform == 'linux':
        base = '/tmp'
    else:
        import tempfile
        base = tempfile.gettempdir()

    return str(Path(base) / f"comfyui-isolation-{env_name}-{pid}.sock")


def cleanup_socket(sock_path: str) -> None:
    """Remove a socket file if it exists."""
    try:
        if os.path.exists(sock_path):
            os.unlink(sock_path)
    except OSError:
        pass
