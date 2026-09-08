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

    A no-op when stdout is not a tty, which covers CI, a piped install, and
    ComfyUI's own captured startup. The install must not behave differently
    because someone is watching it.
    """

    def __init__(self, manifest_dir, env_name: str, index: int, count: int,
                 log=None, interval: float = 0.25):
        self.manifest_dir = Path(manifest_dir)
        self.env_name = env_name
        self.prefix = f"[{index}/{count}] {env_name}"
        self.log = log
        self.interval = interval
        self.total = lock_package_count(self.manifest_dir)
        self._stop = threading.Event()
        self._thread = None
        self._last = 0
        self._enabled = False
        try:
            self._enabled = sys.stdout.isatty()
        except (AttributeError, ValueError):
            self._enabled = False

    def _render(self, done: int) -> None:
        if self.total:
            line = f"  {self.prefix}  {_bar(done, self.total)} {done}/{self.total}"
        else:
            line = f"  {self.prefix}  installing... {done} package(s)"
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
                done = installed_package_count(self.manifest_dir)
            except Exception:
                return          # progress must never take down an install
            if done != self._last:
                self._last = done
                self._render(done)

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
        if self._enabled:
            self._render(installed_package_count(self.manifest_dir))
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
            if self.total:
                self.log(f"  {self.prefix}: {done}/{self.total} package(s) on disk")
            else:
                self.log(f"  {self.prefix}: {done} package(s) on disk")
        return False
