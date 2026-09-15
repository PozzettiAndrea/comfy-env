

def subprocess_env(**extra):
    """A minimal, ISOLATED environment for tests that re-import comfy_env in a
    child interpreter.

    Two Windows traps this exists to avoid:

    * Python on Windows cannot initialize without ``SystemRoot`` -- it dies with
      ``_Py_HashRandomization_Init: failed to get random numbers`` before running
      a single line, so every assertion about the child's output fails for a
      reason unrelated to the test.
    * ``Path.home()`` reads ``USERPROFILE`` on Windows, not ``HOME``. Setting
      only ``HOME`` leaves the child reading the developer's REAL
      ``~/.comfy-env/debug.env``, which is what these tests exist to exclude.
    """
    import os
    import sys as _sys
    from pathlib import Path as _Path

    nowhere = "C:\\nonexistent-so-settings-env-is-absent" if _sys.platform == "win32" \
        else "/nonexistent-so-settings-env-is-absent"
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": nowhere,
        "USERPROFILE": nowhere,
        "PYTHONPATH": str(_Path(__file__).resolve().parents[1] / "src"),
    }
    if _sys.platform == "win32":
        # Required by the interpreter itself, or inherited by tooling it shells out to.
        for k in ("SystemRoot", "SYSTEMROOT", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                  "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"):
            if k in os.environ:
                env[k] = os.environ[k]
    env.update(extra)
    return env


def fake_python_env(tmp_path):
    """An env dir with a runnable interpreter where fetch_metadata looks for one.

    POSIX: bin/python, a symlink to this interpreter. The shared library is
    found through rpath, so a lone symlink runs.

    Windows: python.exe at the root, as pixi lays it out. A symlink does NOT
    run there. The loader resolves python3XX.dll from the symlink's own
    directory, so a bare dir fails on launch with STATUS_DLL_NOT_FOUND
    (0xC0000135) -- on every GitHub runner, and on any box where Python's
    directory is off PATH. Copying the DLLs beside a copied exe is not
    enough either: conda's Python ships its own api-ms-win-crt-* forwarders
    there, and other builds ship other things. What runs anywhere is venv's
    launcher, a tiny exe importing only kernel32 that reads the pyvenv.cfg
    beside it and starts the real python from `home`, where every DLL
    already lives. That is how every Windows venv works, so the env is built
    from one: the launcher and its pyvenv.cfg, moved to the root.
    """
    import shutil
    import subprocess
    import sys
    env_dir = tmp_path / "env"
    if sys.platform != "win32":
        (env_dir / "bin").mkdir(parents=True)
        (env_dir / "bin" / "python").symlink_to(sys.executable)
        return env_dir
    venv = tmp_path / "_venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                   check=True, capture_output=True)
    env_dir.mkdir()
    shutil.copy2(venv / "Scripts" / "python.exe", env_dir / "python.exe")
    shutil.copy2(venv / "pyvenv.cfg", env_dir / "pyvenv.cfg")
    return env_dir
