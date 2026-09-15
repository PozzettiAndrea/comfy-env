"""Filesystem, platform, logging, and subprocess utilities for install/."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Callable



def _patch_uv_platform_py(log: Callable[[str], None] = print) -> None:
    """Patch uv-managed Python's platform.py to handle conda-forge version strings.

    conda-forge Python embeds '| packaged by conda-forge |' in sys.version.
    When pixi's uv creates build-isolation venvs it may use a standard CPython
    whose platform.py can't parse that string, crashing setuptools.  Apply the
    same one-line regex fix that conda-forge ships in their own builds.
    """
    if sys.platform != "win32":
        return
    search_dirs = [
        Path.home() / "AppData" / "Roaming" / "uv" / "python",
        Path.home() / "AppData" / "Local" / "rattler" / "cache" / "python",
    ]
    MARKER = r"r'([\w.+]+)\s*'"
    REPLACEMENT = r"r'([\w.+]+)\s*(?:\ \|\ packaged\ by\ conda\-forge\ \|)?\s*'"
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for py_dir in search_dir.iterdir():
            if not py_dir.name.startswith("cpython-"):
                continue
            platform_py = py_dir / "Lib" / "platform.py"
            if not platform_py.exists():
                continue
            content = platform_py.read_text(encoding="utf-8")
            if "packaged by conda" in content:
                continue
            idx = content.find(MARKER)
            if idx == -1:
                continue
            patched = content[:idx] + REPLACEMENT + content[idx + len(MARKER):]
            platform_py.write_text(patched, encoding="utf-8")
            log(f"[comfy-env] Patched {platform_py} for conda-forge compat")


class InstallLog:
    """The install log: one file per env, at `<env manifest dir>/install.log`.

    Every line goes to the console. Lines emitted while an env's section is
    open also go to that env's file. Lines emitted before any section opens
    -- workspace discovery, GPU detection, the wheel combo -- are the shared
    preamble, replayed into each env's file the first time its section
    opens, so every file reads as the complete story of that one env from
    the top. Workspace-level lines after sections have begun (the orphan
    notice, legacy cleanup) reach the console only; they are about the
    workspace, not any env.

    A section is opened with `begin(env, manifest_dir)` and closed with
    `end()`. Each env is written in several passes -- manifest, pixi
    install, stamp, identity -- so `begin` truncates on the first open and
    appends after that. `file` is the open handle, for callers that want
    the file and not the console (the post-mortem subprocess dump).
    """

    def __init__(self, console: Callable[[str], None]):
        self._console = console
        self._preamble: list[str] = []
        self._fh = None
        self._opened: set[str] = set()
        self.paths: dict[str, Path] = {}

    def __call__(self, msg: str) -> None:
        self._console(msg)
        sys.stdout.flush()
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()
        elif not self._opened:
            self._preamble.append(msg)

    @property
    def file(self):
        return self._fh

    def begin(self, env_name: str, manifest_dir: Path) -> None:
        import datetime
        self.end()
        path = Path(manifest_dir) / "install.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        first = env_name not in self._opened
        self._fh = open(path, "w" if first else "a", encoding="utf-8")
        if first:
            self._opened.add(env_name)
            self.paths[env_name] = path
            self._fh.write(f"# comfy-env install log for {env_name} - {datetime.datetime.now().isoformat()}\n")
            self._fh.write(f"# Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})\n")
            self._fh.write(f"# Platform: {sys.platform}\n\n")
            for line in self._preamble:
                self._fh.write(line + "\n")
        self._fh.flush()

    def end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    close = end


def _log_subprocess(log: Callable, result, label: str = "") -> None:
    """Write subprocess stdout/stderr to the log file (verbose, file-only)."""
    fh = getattr(log, "file", None)
    if fh is None:
        return
    if label:
        fh.write(f"\n--- {label} (exit {result.returncode}) ---\n")
    if result.stdout and result.stdout.strip():
        fh.write(f"[stdout]\n{result.stdout}\n")
    if result.stderr and result.stderr.strip():
        fh.write(f"[stderr]\n{result.stderr}\n")
    fh.flush()


_ESC_SEQ = re.compile(
    rb"\x1b\[([0-9;?]*)([A-Za-z])"        # CSI: ESC [ params final
    rb"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: ESC ] ... BEL or ST
    rb"|\x1b[()][A-Za-z0-9]"              # charset designation
    rb"|(\r)|(\n)|([^\x1b\r\n]+)"
)


class _FrameReducer:
    """Turn a terminal byte stream into the lines a log should keep.

    Built against pixi's real redraw grammar. A progress frame is drawn as
    `\r ESC[2K text ESC[1B` per line under an `ESC[nA` (cursor up), and
    NEVER ends in `\n`; only a real message does. So a line survives only
    when it is finished with `\n`; a CR, an erase, or any vertical cursor
    move (up or down: the buffer's text is on some other row now) drops
    whatever was pending as a frame.

    Two details the obvious version gets wrong, both measured:

    - The pty's line discipline translates every `\n` the child writes
      into `\r\n` (onlcr). A CR immediately followed by LF is therefore a
      line ending, not an overwrite; treating it as one ate every real line.
    - `os.read()` chunks split escape sequences. Eight of twenty-four 4 KB
      chunks of a real capture ended mid-sequence. An unfinished trailing
      ESC is carried into the next chunk rather than tokenised as text.
    """

    def __init__(self):
        self._buf = b""
        self._carry = b""
        self._pending_cr = False

    def feed(self, chunk: bytes) -> list[str]:
        data = self._carry + chunk
        self._carry = b""
        j = data.rfind(b"\x1b")
        if j != -1 and not _ESC_SEQ.match(data, j):
            data, self._carry = data[:j], data[j:]
        out: list[str] = []
        for m in _ESC_SEQ.finditer(data):
            if m.group(4):                            # LF: a real line
                self._emit(out)
                self._pending_cr = False
                continue
            if self._pending_cr:                      # CR then not-LF: overwritten
                self._buf = b""
                self._pending_cr = False
            if m.group(3):
                self._pending_cr = True
            elif m.group(2):
                if m.group(2) in b"ABEFKJ":           # any vertical move, or an erase
                    self._buf = b""
            elif m.group(5):
                self._buf += m.group(5)
        return out

    def finish(self) -> list[str]:
        out: list[str] = []
        self._emit(out)
        return out

    def _emit(self, out: list[str]) -> None:
        line = self._buf.decode("utf-8", errors="replace").rstrip()
        if line.strip():
            out.append(line)
        self._buf = b""


def _real_stderr_is_a_terminal() -> bool:
    try:
        return sys.__stderr__ is not None and sys.__stderr__.isatty()
    except (AttributeError, ValueError):
        return False


def _run_pixi(cmd, log: Callable, cwd=None, env=None, terminal_fd=None):
    """Run pixi so its native progress bar reaches the terminal, when there is one.

    pixi draws its bar only when stderr is a terminal; through a pipe it
    draws nothing at any verbosity. So when the process's real stderr is a
    terminal, give pixi a pty for stderr and relay the master end two ways:
    the raw bytes to the real terminal, so the bar renders as pixi drew it,
    and a reduced transcript (frames dropped, messages kept) to the log
    file only -- the console already saw it. Off a terminal (a service, an
    IDE, captured output), or on Windows where a pty needs ConPTY, fall
    back to _run_streaming and pixi's pipe behaviour: warnings, the phase
    lines under -v, the final line, and nothing drawn.

    `terminal_fd` is the fd the raw bytes go to; None means the real stderr.
    """
    if sys.platform == "win32" or not _real_stderr_is_a_terminal():
        return _run_streaming(cmd, log, cwd=cwd, env=env)

    import fcntl
    import pty
    import shutil
    import struct
    import subprocess
    import termios
    import threading

    if terminal_fd is None:
        terminal_fd = sys.__stderr__.fileno()
    fh = getattr(log, "file", None)

    master, slave = pty.openpty()
    cols, rows = shutil.get_terminal_size()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=slave, stdin=subprocess.DEVNULL, text=True,
    )
    os.close(slave)

    reduced: list[str] = []
    reducer = _FrameReducer()

    def _relay():
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError:            # EIO: the slave end closed
                break
            if not chunk:
                break
            try:
                os.write(terminal_fd, chunk)
            except OSError:
                pass
            for line in reducer.feed(chunk):
                reduced.append(line)
                if fh is not None:
                    fh.write(f"  {line}\n")
                    fh.flush()

    t = threading.Thread(target=_relay, daemon=True)
    t.start()

    stdout_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line_text = line.rstrip("\n")
        stdout_lines.append(line_text)
        if line_text.strip():
            log(f"  {line_text}")

    proc.wait()
    t.join(timeout=5)
    os.close(master)
    for line in reducer.finish():
        reduced.append(line)
        if fh is not None:
            fh.write(f"  {line}\n")
            fh.flush()

    return subprocess.CompletedProcess(
        cmd, proc.returncode, "\n".join(stdout_lines), "\n".join(reduced) + ("\n" if reduced else ""),
    )


def _run_streaming(cmd, log: Callable, cwd=None, env=None):
    """Run a subprocess, streaming stdout/stderr lines to log in real time."""
    import subprocess
    import threading

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        # stdin was INHERITED, so a child that decides to prompt (an auth
        # challenge, a confirmation) blocks forever against a console nobody
        # is watching, inside an install that looks hung. DEVNULL turns a
        # prompt into an immediate EOF, which the child handles as "no".
        stdin=subprocess.DEVNULL,
    )

    def _read_stderr():
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_lines.append(line)
            line_text = line.rstrip("\n")
            if line_text.strip():
                log(f"  {line_text}")

    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    assert proc.stdout is not None
    for line in proc.stdout:
        line_text = line.rstrip("\n")
        stdout_lines.append(line_text)
        if line_text.strip():
            log(f"  {line_text}")

    proc.wait()
    t.join(timeout=5)

    return subprocess.CompletedProcess(
        cmd, proc.returncode,
        "\n".join(stdout_lines),
        "".join(stderr_lines),
    )
