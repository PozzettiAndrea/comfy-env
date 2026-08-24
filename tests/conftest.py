

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
      ``~/.comfy-env/settings.env``, which is what these tests exist to exclude.
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
