"""Contract: the pixi install bar counts the right thing and never fails a build.

pixi reports nothing usable through a pipe (its own `--no-progress` is forced
on when stderr is not a terminal), so progress is derived from what lands on
disk. That makes the counting rule the whole design, and both halves of it
were wrong in the first draft.
"""

import sys

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


def test_no_bar_when_stdout_is_not_a_tty(tmp_path, monkeypatch, capsys):
    """CI, a piped install and ComfyUI's captured startup must all be clean.

    An install must not behave differently because someone is watching it.
    """
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=3, meta=3)
    lines = []
    with InstallProgress(d, "pack", 1, 1, log=lines.append):
        pass
    assert capsys.readouterr().out == ""
    assert lines and "3/3" in lines[0]


def test_wrap_log_is_transparent_when_disabled(tmp_path, monkeypatch):
    """Catches wrapping the log with a bar-clearing callback that then writes
    escape codes into `install.log` on a non-tty run."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)
    d = _env(tmp_path, conda=1, meta=1)
    p = InstallProgress(d, "pack", 1, 1, log=print)
    assert p.wrap_log(print) is print
