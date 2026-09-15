"""InstallLog: one file per env, the shared preamble replayed into each.

The old shape was one workspace-level install.log for the whole run, so a
machine with twenty envs had one file that only ever held the last install,
and finding out what happened to env X meant scrolling past nineteen others.
"""
from comfy_env.install.helpers import InstallLog, _log_subprocess


def test_each_env_gets_its_own_file_with_the_preamble(tmp_path):
    """Catches: a single workspace-level file; or per-env files that start
    at the env's own lines and lose the discovery/GPU/combo context above."""
    seen = []
    log = InstallLog(seen.append)
    log("[comfy-env] Workspace: /w")            # preamble: before any section
    log("[comfy-env] GPU: something")
    log.begin("alpha", tmp_path / "alpha")
    log("alpha line")
    log.end()
    log.begin("beta", tmp_path / "beta")
    log("beta line")
    log.end()

    a = (tmp_path / "alpha" / "install.log").read_text()
    b = (tmp_path / "beta" / "install.log").read_text()
    assert a.startswith("# comfy-env install log for alpha")
    assert "[comfy-env] Workspace: /w" in a and "[comfy-env] GPU: something" in a
    assert "alpha line" in a and "beta line" not in a
    assert "[comfy-env] Workspace: /w" in b and "beta line" in b and "alpha line" not in b
    assert seen == ["[comfy-env] Workspace: /w", "[comfy-env] GPU: something", "alpha line", "beta line"]
    assert log.paths == {"alpha": tmp_path / "alpha" / "install.log",
                         "beta": tmp_path / "beta" / "install.log"}


def test_reopening_an_env_appends_rather_than_truncates(tmp_path):
    """Each env is written in several passes (manifest, pixi, stamp,
    identity). Catches: open("w") on every begin, which would leave only
    the last pass in the file."""
    log = InstallLog(lambda m: None)
    log.begin("e", tmp_path / "e"); log("pass one"); log.end()
    log.begin("e", tmp_path / "e"); log("pass two"); log.end()
    text = (tmp_path / "e" / "install.log").read_text()
    assert "pass one" in text and "pass two" in text
    assert text.count("# comfy-env install log for e") == 1, "the header was written twice"


def test_lines_between_sections_reach_the_console_only(tmp_path):
    """Workspace-level lines after sections have begun (the orphan notice)
    are about the workspace, not any env. Catches: a preamble that keeps
    accumulating and replays stale workspace lines into the next env's
    file."""
    seen = []
    log = InstallLog(seen.append)
    log.begin("a", tmp_path / "a"); log("a line"); log.end()
    log("[comfy-env] 20 env(s) in the workspace are not declared")
    log.begin("b", tmp_path / "b"); log("b line"); log.end()
    assert "not declared" not in (tmp_path / "b" / "install.log").read_text()
    assert "not declared" not in (tmp_path / "a" / "install.log").read_text()
    assert any("not declared" in m for m in seen)


def test_file_property_carries_file_only_writes_to_the_open_env(tmp_path):
    """_log_subprocess writes the post-mortem block through log.file and
    silently does nothing if the attribute is missing. Catches: a log with
    no .file, which would drop every subprocess dump without a trace."""
    import subprocess
    seen = []
    log = InstallLog(seen.append)
    log.begin("e", tmp_path / "e")
    _log_subprocess(log, subprocess.CompletedProcess(["x"], 3, "out text", "err text"), "pixi install (e)")
    log.end()
    text = (tmp_path / "e" / "install.log").read_text()
    assert "--- pixi install (e) (exit 3) ---" in text and "err text" in text
    assert seen == [], "the post-mortem block must not reach the console"
    assert log.file is None, "end() must drop the handle"
