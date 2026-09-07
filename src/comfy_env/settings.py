"""Settings for comfy-env: environment variables, and nothing else.

Precedence has one tier. A setting is an environment variable (``COMFY_ENV_*``)
or it is a default; there is no settings file.

There used to be a second tier -- ``~/.comfy-env/settings.env``, written by the
General tab of ``comfy-env settings`` and loaded here at import with
``os.environ.setdefault``. It never worked. Nothing on the ComfyUI runtime path
imported this module, so a key the TUI wrote reached ``os.environ`` only in the
two processes that DID import it (the CLI itself, and the installer's
cuda-wheels lookup). Toggling Pool IPC in the TUI therefore changed nothing
about how workers ran, which is the one thing it looked like it was for.

The file, the loader and the General tab were removed together rather than
wired up. The consumers of these settings are workers, which cannot import
comfy_env at all (different venv) and parse ``os.environ`` directly, so an env
var is the only tier that can reach every reader; a file tier can only ever
serve the subset of readers that happen to import this module.
``~/.comfy-env/debug.env`` is unaffected and still works, because
``comfy_env.debug`` IS imported on the runtime path -- that difference, not
the file format, is why one tier was real and the other was decoration.

What survives here is the removed-variable tombstones, and the facade imports
this module so they actually run. They did not, before: with the only two
importers off the runtime path, a machine still exporting
``COMFY_ENV_ISOLATE=0`` got no error from ComfyUI at all.
"""

import os

# Removed settings (0.4.25). A falsy env var is a semantic inversion (the
# machine was told to run un-isolated and no longer will) and fails loudly; a
# truthy one matches the only behavior that exists now and is ignored.
_REMOVED_DISABLE_VARS = ("COMFY_ENV_ISOLATE", "COMFY_ENV_INSTALL_ISOLATED")

# Removed in 0.4.25. Opposite polarity to the two above: this one did nothing
# when falsy (it was already the default) and DID something when truthy, so the
# truthy setting is the semantic inversion here -- the machine was told to
# self-heal missing envs at startup and no longer will.
_REMOVED_ENABLE_VARS = ("COMFY_ENV_AUTO_INSTALL",)

_REMOVED_ENV_VARS = _REMOVED_DISABLE_VARS + _REMOVED_ENABLE_VARS


def _check_removed_env_vars():
    for _var in _REMOVED_DISABLE_VARS:
        _val = os.environ.get(_var)
        if _val is None:
            continue
        if _val.strip().lower() in ("0", "false", "no", "off"):
            raise RuntimeError(
                f"[comfy-env] {_var}={_val} was removed in 0.4.25 and can no "
                f"longer disable anything: isolation is always on, and missing "
                f"envs fall back per-env automatically. Unset this variable "
                f"(check your shell profile, Dockerfile/CI, systemd units). "
                f"In a container, run `comfy-env install --dir <pack>` at image "
                f"build time so the runtime never needs the network."
            )
        # Truthy matches the only behavior that exists now: silent ignore.

    for _var in _REMOVED_ENABLE_VARS:
        _val = os.environ.get(_var)
        if _val is None:
            continue
        if _val.strip().lower() not in ("0", "false", "no", "off"):
            raise RuntimeError(
                f"[comfy-env] {_var}={_val} was removed in 0.4.25. Envs are no "
                f"longer materialized lazily at startup: install() is the only "
                f"builder. A second builder could not be kept in agreement with "
                f"it -- it silently skipped the macOS libomp dedupe and uv's "
                f"python pinning, leaving envs that every later install then "
                f"SKIPPED as up to date. Unset this variable and build envs "
                f"with `comfy-env install --dir <pack>` (in a container, at "
                f"image build time)."
            )
        # Falsy always was a no-op (the default was off): silent ignore.


_check_removed_env_vars()
