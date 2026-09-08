"""Live progress for `pixi install`, derived from the filesystem.

pixi will not tell us. Its own `--no-progress` is "always turned on if stderr
is not a terminal", and comfy-env runs it with `stderr=subprocess.PIPE` so it
can tee to `install.log` -- so pixi self-suppresses, and what reaches the pipe
is warnings plus one line, `The default environment has been installed.`.
Raising verbosity does not help: `-v` adds phase timings and `-vv` adds
internal DEBUG, neither naming a package. There is no `--json`.

Giving pixi a pty would get its native bar back, but only on Unix (Windows
needs ConPTY, so a C dependency in the HOST env, which the host-env principle
forbids) and it would fill install.log with ANSI redraw noise.

So count the result instead of parsing the narration. `pixi.lock` declares
what the env will contain, and the env fills in as it installs: one
`conda-meta/<pkg>.json` per conda package, one `site-packages/*.dist-info`
per pypi package. Measured on a real env: 160 conda entries in the lock, 160
files in conda-meta. That is a real per-package signal, it costs a directory
listing, and it works cross-platform with stderr piped.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path

#: A conda entry in pixi.lock looks like `  - conda: https://...`.
#:
#: CONDA ONLY, and the restriction is load bearing. Counting pypi entries too
#: and pairing them with `site-packages/*.dist-info` double counts, because a
#: conda package that is a Python library ships a dist-info AS WELL AS its
#: conda-meta record: measured on a 140 package env with zero pypi entries in
#: the lock, site-packages still held 32 dist-info directories, every one of
#: them belonging to a conda package. Conda is also the bulk and the slow
#: part (download plus extract), so a conda-only denominator is both exact
#: and the one worth watching.
_LOCK_CONDA = re.compile(r"^\s+- conda:", re.M)


def lock_package_count(manifest_dir: Path) -> int:
    """How many conda packages `pixi.lock` says this env will contain, or 0.

    0 means "no target", and every caller must treat that as "render a
    spinner, not a fraction". A lock that cannot be read is not an error
    here: progress is decoration, and decoration never fails an install.
    """
    try:
        return len(_LOCK_CONDA.findall(
            (Path(manifest_dir) / "pixi.lock").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return 0


def installed_package_count(manifest_dir: Path) -> int:
    """How many conda packages are on disk right now.

    One `conda-meta/<name>-<version>-<build>.json` per installed package,
    written as each is linked. Verified exact against the lock on two real
    envs: 140/140 and 160/160.

    Deliberately tolerant: the directory does not exist until pixi creates
    it, and being called before that is normal.
    """
    try:
        return sum(1 for _ in (Path(manifest_dir) / ".pixi" / "envs" /
                               "default" / "conda-meta").glob("*.json"))
    except OSError:
        return 0


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return ""
    filled = min(width, max(0, round(width * done / total)))
    return "█" * filled + "░" * (width - filled)


class InstallProgress:
    """Render `[3/26] name  ████░░ 141/226`, live, on a terminal only.

    Two output channels on purpose. The terminal gets a redrawing line via
    `\\r`; the log callback gets nothing at all until the end, then one
    discrete line. `install.log` is read after the fact by someone debugging
    a failure, and a thousand redraws of a bar is worse than useless there.

    Off a tty there is still progress, just not a redrawing one: a discrete
    line every `line_interval` seconds while the count moves. Gating the whole
    feature on `isatty` meant nobody ever saw it, because ComfyUI's stdout is
    routinely not a terminal (a service, an IDE, anything that captures
    output), and a sixty second install then printed one line at the end.
    Discrete lines are also what `install.log` wants, since that is where a
    reader goes to find out where a slow install spent its time.
    """

    def __init__(self, manifest_dir, env_name: str, index: int, count: int,
                 log=None, interval: float = 0.25,
                 line_interval: float = 10.0):
        self.manifest_dir = Path(manifest_dir)
        self.env_name = env_name
        self.prefix = f"[{index}/{count}] {env_name}"
        self.log = log
        self.interval = interval
        self.line_interval = line_interval
        # Seeded to the start time, NOT 0.0. At 0.0 the first gate check is
        # `monotonic() - 0 < interval`, which is false forever, so the first
        # line always fired however short the install was and a one second
        # cached env logged "0 package(s)" for nothing.
        self._last_line_at = 0.0
        self._last_line_count = -1
        # May legitimately be 0 here. A brand new env has no pixi.lock until
        # pixi writes one part way through the install, so reading it once at
        # construction gave every fresh env a denominator of zero and a
        # spinner for the whole run. The poller re-reads until it appears.
        self.total = lock_package_count(self.manifest_dir)
        self._started = time.monotonic()
        self._last_line_at = self._started
        self._stop = threading.Event()
        self._thread = None
        self._last = 0
        self._enabled = False
        try:
            self._enabled = sys.stdout.isatty()
        except (AttributeError, ValueError):
            self._enabled = False

    @property
    def _elapsed(self) -> str:
        s = time.monotonic() - self._started
        if s < 60:
            return f"{s:4.1f}s"
        return f"{int(s // 60)}m{int(s % 60):02d}s"

    def _render(self, done: int) -> None:
        if self.total:
            line = (f"  {self.prefix}  {_bar(done, self.total)} "
                    f"{done}/{self.total}  {self._elapsed}")
        else:
            line = f"  {self.prefix}  installing... {done} package(s)  {self._elapsed}"
        # Pad to overwrite a previously longer line; \r keeps it on one row.
        sys.stdout.write("\r" + line.ljust(self._width) + "\r" + line)
        sys.stdout.flush()

    @property
    def _width(self) -> int:
        try:
            import shutil
            return max(40, shutil.get_terminal_size((80, 20)).columns - 1)
        except Exception:
            return 80

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not self.total:      # the lock arrives mid install
                    self.total = lock_package_count(self.manifest_dir)
                done = installed_package_count(self.manifest_dir)
            except Exception:
                return          # progress must never take down an install
            self._last = done
            if self._enabled:
                # Repaint on a clock, not only on a change: the elapsed time
                # is part of the line, and a stalled install is exactly when
                # a reader most needs to see the seconds still moving.
                self._render(done)
            else:
                self._maybe_line(done)

    def _maybe_line(self, done: int) -> None:
        """One discrete line, at most every `line_interval`, off a tty.

        Rate limited AND change gated together: a fast cached install adds
        nothing beyond its summary, and a stalled one does not repeat an
        identical line forever.
        """
        if self.log is None or done == self._last_line_count:
            return
        now = time.monotonic()
        if now - self._last_line_at < self.line_interval:
            return
        self._last_line_at = now
        self._last_line_count = done
        frac = f"{done}/{self.total}" if self.total else f"{done}"
        self.log(f"  {self.prefix}: {frac} package(s), {self._elapsed.strip()} elapsed")

    def wrap_log(self, log):
        """A log callback that does not get scribbled over by the bar.

        pixi is nearly silent through a pipe, but not entirely: a manifest
        warning arrives mid install, and without this it lands on the same
        terminal row the bar is redrawing. Clear the row, let the message
        through, and let the next poll repaint.
        """
        if not self._enabled or log is None:
            return log

        def _log(msg):
            sys.stdout.write("\r" + " " * self._width + "\r")
            sys.stdout.flush()
            log(msg)
            self._render(self._last)

        return _log

    def __enter__(self):
        # The poller runs either way. On a tty it repaints a bar; off one it
        # emits a line every line_interval. Starting it only when isatty was
        # the bug that made this feature invisible in normal ComfyUI use.
        if self._enabled:
            self._render(installed_package_count(self.manifest_dir))
        if self._enabled or self.log is not None:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._enabled:
            sys.stdout.write("\r" + " " * self._width + "\r")
            sys.stdout.flush()
        if self.log is not None:
            done = installed_package_count(self.manifest_dir)
            total = self.total or lock_package_count(self.manifest_dir)
            frac = f"{done}/{total}" if total else str(done)
            self.log(f"  {self.prefix}: {frac} package(s) in {self._elapsed.strip()}")
        return False
