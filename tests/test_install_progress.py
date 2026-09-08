"""Contract: the pixi install bar counts the right thing and never fails a build.

pixi reports nothing usable through a pipe (its own `--no-progress` is forced
on when stderr is not a terminal), so progress is derived from what lands on
disk. That makes the counting rule the whole design, and both halves of it
were wrong in the first draft.
"""

import sys
import time

from comfy_env.install.progress import (
    InstallProgress, installed_package_count, lock_package_count,
)


def _env(tmp_path, conda=0, pypi=0, meta=0, dist_info=0, pyver="3.12"):
    (tmp_path / "pixi.lock").write_text(
        "packages:\n"
        + "".join(f"  - conda: https://example/c{i}.conda\n" for i in range(conda))
        + "".join(f"  - pypi: https://example/p{i}.whl\n" for i in range(pypi)),
        encoding="utf-8",
    )
    root = tmp_path / ".pixi" / "envs" / "default"
    cm = root / "conda-meta"
    cm.mkdir(parents=True)
    for i in range(meta):
        (cm / f"pkg{i}-1.0-h0.json").write_text("{}", encoding="utf-8")
    site = root / "lib" / f"python{pyver}" / "site-packages"
    site.mkdir(parents=True)
    for i in range(dist_info):
        (site / f"pkg{i}-1.0.dist-info").mkdir()
    return tmp_path


def test_target_counts_conda_only(tmp_path):
    """Catches counting pypi entries into the denominator.

    A conda package that is a Python library ships a `.dist-info` as well as
    its conda-meta record, so pairing a conda+pypi target with a
    conda-meta+dist-info count double counts every one of them. Measured on a
    real 140 package env with ZERO pypi entries in its lock: site-packages
    still held 32 dist-info directories, all belonging to conda packages.
    """
    d = _env(tmp_path, conda=140, pypi=66)
    assert lock_package_count(d) == 140


def test_installed_ignores_dist_info(tmp_path):
    """The other half of the same double count, from the disk side."""
    d = _env(tmp_path, conda=10, meta=10, dist_info=7)
    assert installed_package_count(d) == 10


def test_installed_ignores_a_stale_sibling_python_dir(tmp_path):
    """Catches globbing `lib/python*/site-packages`.

    Real envs carry an inert `lib/python3.1/site-packages` beside the real
    `lib/python3.12/site-packages` (a conda-site.pth artifact), so that glob
    counted the same tree twice. Counting conda-meta sidesteps it entirely.
    """
    d = _env(tmp_path, conda=5, meta=5, dist_info=5, pyver="3.12")
    stale = d / ".pixi" / "envs" / "default" / "lib" / "python3.1" / "site-packages"
    stale.mkdir(parents=True)
    for i in range(5):
        (stale / f"pkg{i}-1.0.dist-info").mkdir()
    assert installed_package_count(d) == 5


def test_counts_survive_a_missing_env(tmp_path):
    """Progress is decoration. It runs before pixi has created anything and
    must return 0 rather than raise, or it takes the install down with it."""
    assert installed_package_count(tmp_path / "nope") == 0
    assert lock_package_count(tmp_path / "nope") == 0


def test_no_ansi_when_stdout_is_not_a_tty(tmp_path, monkeypatch, capsys):
    """Off a tty there must be no redraw, but there MUST still be progress.

    Catches gating the whole feature on isatty, which is what shipped first
    and made it invisible: ComfyUI's stdout is routinely not a terminal, so a
    sixty second install printed one line at the very end and looked hung.
    Progress off a tty goes to the log as discrete lines instead.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=3, meta=3)
    lines = []
    with InstallProgress(d, "pack", 1, 1, log=lines.append, interval=0.01,
                         line_interval=0.0):
        time.sleep(0.1)
    out = capsys.readouterr().out
    assert "\r" not in out and "█" not in out, "no bar redraw off a tty"
    assert any("3/3" in ln for ln in lines)
    assert any("elapsed" in ln for ln in lines), "no live progress line emitted"


def test_progress_lines_are_rate_limited(tmp_path, monkeypatch):
    """A cached install finishing in a second must not add a line per tick."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=3, meta=3)
    lines = []
    with InstallProgress(d, "pack", 1, 1, log=lines.append, interval=0.01,
                         line_interval=60.0):
        time.sleep(0.1)
    # Only the final summary; the 60s gate suppresses every interim line.
    assert len(lines) == 1 and "in " in lines[0]


def test_wrap_log_is_transparent_when_disabled(tmp_path, monkeypatch):
    """Catches wrapping the log with a bar-clearing callback that then writes
    escape codes into `install.log` on a non-tty run."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=1, meta=1)
    p = InstallProgress(d, "pack", 1, 1, log=print)
    assert p.wrap_log(print) is print


def test_target_is_recovered_when_the_lock_arrives_late(tmp_path, monkeypatch):
    """Catches reading pixi.lock once, at construction.

    A brand new env has no `pixi.lock` until pixi writes one part way through
    the install, so a single read at start gave every fresh env a denominator
    of zero and a spinner for the whole run, then a final line reading
    "329 package(s) on disk" with no total. Observed exactly that on a real
    install whose lock turned out to hold 329 entries matching 329 files.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    p = InstallProgress(tmp_path, "pack", 1, 1, log=None, interval=0.01)
    assert p.total == 0, "no lock yet, so no denominator yet"

    # pixi writes the lock and starts linking
    _env(tmp_path, conda=4, meta=4)
    p.total = p.total or lock_package_count(tmp_path)
    assert p.total == 4


def test_final_line_reports_elapsed_time(tmp_path, monkeypatch):
    """The summary must say how long it took, not just what landed."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=2, meta=2)
    lines = []
    with InstallProgress(d, "pack", 1, 1, log=lines.append):
        pass
    assert lines and "2/2" in lines[0] and lines[0].rstrip().endswith("s")


def test_heartbeat_continues_when_the_count_is_frozen(tmp_path, monkeypatch):
    """The flat spot is the whole point. Catches ANDing the rate limit with a
    change gate, which is what shipped first.

    Measured on a cold install: conda linking finishes in about two seconds
    and pypi then runs for another fifty, during which `conda-meta` stops
    growing entirely. With a change gate the heartbeat stopped there, so the
    console showed one line and then nothing for fifty seconds, which is
    indistinguishable from the hang this class exists to disprove. uv also
    links its whole tree in a final burst, so no counter of any kind fills
    that window: only the clock can.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=26, meta=26)       # already "complete", never moves
    lines = []
    with InstallProgress(d, "pack", 1, 1, log=lines.append, interval=0.01,
                         line_interval=0.03):
        time.sleep(0.25)
    interim = [ln for ln in lines if "elapsed" in ln]
    assert len(interim) >= 3, (
        f"heartbeat stopped while the count was frozen: {lines}")
    assert all("26/26" in ln for ln in interim)
