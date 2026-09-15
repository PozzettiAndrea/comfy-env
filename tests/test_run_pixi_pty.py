"""pixi's progress bar on the terminal, and a clean install.log, at the same time.

pixi draws its bar only when stderr is a terminal. _run_pixi gives it a pty
when the real stderr is one, relays the raw bytes to that terminal, and
reduces the same bytes to log lines. Off a terminal it falls back to a pipe.
"""
import os
import sys

import pytest

from comfy_env.install import helpers
from comfy_env.install.helpers import _FrameReducer, _run_pixi

ESC = "\x1b"
# pixi's real redraw grammar: cursor up, then per line CR, erase, text, cursor down
FRAME = f"{ESC}[3A\r{ESC}[2K  ⠴ installing [━━━━──] 12/329{ESC}[1B\r{ESC}[2K  ▪ preparing [━━━━━━] 329/329{ESC}[1B".encode()


def _reduce(*chunks):
    r = _FrameReducer()
    out = []
    for c in chunks:
        out += r.feed(c)
    return out + r.finish()


def test_reducer_keeps_lf_lines_and_drops_frames():
    """Catches: splitting on LF alone, which leaks a frame into the line it
    precedes; and dropping everything, which loses the messages."""
    out = _reduce(FRAME + FRAME + b" INFO pixi_install_pypi: Installed 48 packages in 180ms\r\n" + FRAME)
    assert out == [" INFO pixi_install_pypi: Installed 48 packages in 180ms"]


def test_reducer_crlf_is_a_line_ending_not_an_overwrite():
    """The pty's line discipline turns every LF the child writes into CRLF.
    Catches: 'CR discards the buffer' applied before checking for LF, which
    ate every real line -- measured: 94 KB of capture reduced to zero."""
    assert _reduce(b"hello\r\n") == ["hello"]
    assert _reduce(b"hello\r\nworld\r\n") == ["hello", "world"]


def test_reducer_bare_cr_overwrite_keeps_only_the_last_frame():
    """A tqdm-style single-line bar redraws with a bare CR and no erase.
    Catches: CR that does not discard, which concatenates every frame into
    one line -- 'frame 1frame 2'."""
    assert _reduce(b"\rframe 1\rframe 2\rframe 3\r\n") == ["frame 3"]


def test_reducer_erase_discards_the_stale_frame_before_a_message():
    """A frame left in the buffer must not prefix the next message.
    Catches: not treating ESC[K / ESC[J / ESC[A as 'what was here is gone'."""
    out = _reduce(b"\r  \xe2\xa0\xb4 installing 3/9" + f"{ESC}[2K WARN something\r\n".encode())
    assert out == [" WARN something"]
    out = _reduce(b"frame" + f"{ESC}[3A".encode() + b" INFO after up\r\n")
    assert out == [" INFO after up"]


def test_reducer_carries_an_escape_split_across_chunks():
    """os.read() splits escape sequences: 8 of 24 4 KB chunks of a real
    capture ended mid-sequence. Catches: tokenising a bare ESC as text,
    which leaks '[2' into the log."""
    out = _reduce(b"\r" + ESC.encode() + b"[2", b"K INFO x\r\n")
    assert out == [" INFO x"]
    out = _reduce(ESC.encode(), b"[3A INFO y\r\n")
    assert out == [" INFO y"]


posix = pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX only; Windows keeps the pipe path")


def test_pipe_path_when_the_real_stderr_is_not_a_terminal(monkeypatch, tmp_path):
    """Catches: forcing a pty unconditionally, which dumps escape bytes into
    whatever is capturing a service's output."""
    monkeypatch.setattr(helpers, "_real_stderr_is_a_terminal", lambda: False)
    calls = []
    monkeypatch.setattr(helpers, "_run_streaming", lambda cmd, log, cwd=None, env=None: calls.append(cmd) or "streamed")
    assert _run_pixi(["x"], log=lambda m: None) == "streamed"
    assert calls == [["x"]]


@posix
def test_child_sees_a_terminal_on_stderr(monkeypatch, tmp_path):
    """The whole point. Catches: handing the child a pipe, which makes pixi
    self-suppress the bar."""
    monkeypatch.setattr(helpers, "_real_stderr_is_a_terminal", lambda: True)
    r_fd, w_fd = os.pipe()
    seen = []
    result = _run_pixi(
        [sys.executable, "-c", "import sys; print('TTY' if sys.stderr.isatty() else 'PIPE', file=sys.stderr)"],
        log=seen.append, terminal_fd=w_fd,
    )
    os.close(w_fd)
    raw = os.read(r_fd, 4096); os.close(r_fd)
    assert b"TTY" in raw, raw
    assert result.returncode == 0


@posix
def test_terminal_gets_raw_bytes_and_log_gets_reduced_lines(monkeypatch, tmp_path):
    """Catches: relaying the reduced text to the terminal (the bar stops
    animating); writing the raw bytes to the log (escape noise in a file);
    or sending pixi's lines through the console callback a second time."""
    monkeypatch.setattr(helpers, "_real_stderr_is_a_terminal", lambda: True)
    from comfy_env.install.helpers import InstallLog
    console = []
    log = InstallLog(console.append)
    log.begin("e", tmp_path / "e")
    r_fd, w_fd = os.pipe()
    script = (
        "import sys\n"
        f"sys.stderr.write({FRAME.decode()!r}); sys.stderr.flush()\n"
        "sys.stderr.write(' INFO real line\\n'); sys.stderr.flush()\n"
        f"sys.stderr.write({FRAME.decode()!r}); sys.stderr.flush()\n"
    )
    result = _run_pixi([sys.executable, "-c", script], log=log, terminal_fd=w_fd)
    os.close(w_fd)
    raw = b""
    while True:
        try:
            c = os.read(r_fd, 65536)
        except OSError:
            break
        if not c:
            break
        raw += c
    os.close(r_fd)
    log.end()
    text = (tmp_path / "e" / "install.log").read_text()
    assert b"\x1b[3A" in raw and b"329/329" in raw, "the terminal must get the frames verbatim"
    assert "\x1b" not in text and "329/329" not in text, "the log must not get frames or escapes"
    assert "   INFO real line" in text, text
    assert console == [], "pixi's stderr must not be echoed through the console callback; the terminal already has it"
    assert result.stderr == " INFO real line\n", "the post-mortem block gets the reduced transcript"
