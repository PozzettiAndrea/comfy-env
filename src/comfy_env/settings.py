"""General settings for comfy-env.

Precedence (most specific wins):
  1. Environment variables (COMFY_ENV_*)
  2. Persistent ~/.comfy-env/settings.env (loaded at import with
     os.environ.setdefault -- it fills UNSET env vars, so it can never
     override an explicitly-set one)
  3. Defaults

All settings are machine-global: the per-pack [settings] section was
removed in 0.4.25 (its one wired key served an experiment that a global
env var covers; its other key was parsed but never consulted).

Workers can't import this module (different venv), so they parse env vars directly.
"""

import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".comfy-env" / "settings.env"

# Removed settings (0.4.25). Checked BEFORE the persistent file loads: after
# the loader's setdefault, a Dockerfile/CI export and a TUI-written file line
# are indistinguishable, and only the former is user intent. The TUI wrote
# EVERY settings key to settings.env on save, so file keys are residue for
# every user who ever opened `comfy-env settings` -- they are skipped quietly
# and disappear on the next save. A falsy env var is a semantic inversion
# (the machine was told to run un-isolated and no longer will) and fails
# loudly; a truthy one matches the only behavior that exists now and warns.
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

# Load persistent settings (simple KEY=VALUE file) -- env vars always override
if SETTINGS_FILE.exists():
    try:
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                if k in _REMOVED_ENV_VARS:
                    continue  # TUI residue; next save drops it

                os.environ.setdefault(k, v.strip())
    except RuntimeError:
        raise
    except Exception:
        pass


# General settings
GENERAL_SETTINGS = [
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

GENERAL_DEFAULTS = {
    "COMFY_ENV_POOL_IPC": False,
}

# String settings. Unlike the boolean/numeric tables these are read at their
# point of use (each has a non-trivial default to compute), so this table is
# documentation for the settings surface rather than a resolution mechanism.
# Numeric settings
# Empty since 0.4.25 -- COMFY_ENV_WORKER_VRAM_BUDGET (manual per-worker VRAM
# cap) was removed: the budget-negotiation callback computes the honest number
# automatically, nobody ever set the manual override, and its origin predates
# anyone's memory. Kept as an empty container so the CLI settings surface
# stays stable (same pattern as PATCH_SETTINGS).
NUMERIC_SETTINGS = []


def get_numeric(var: str, default: float = 0) -> float:
    val = os.environ.get(var, "")
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default



# Patches (monkey-patching ComfyUI internals). Empty since 0.4.21 -- the
# only entry, COMFY_ENV_PATCH_SHAREABLE_POOL (parent-side CUDA shareable
# pool), was removed: experimental, default-off, and the cause of an
# environment->isolation import cycle. Kept as empty containers so the CLI
# settings surface stays stable.
PATCH_SETTINGS = []
PATCH_DEFAULTS = {}




