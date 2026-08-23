"""General settings for comfy-env.

Precedence (most specific wins -- a per-pack declaration is more specific
than a global environment variable):
  1. Per-pack `[settings]` in comfy-env-root.toml (via resolve_bool/
     resolve_numeric with node_settings)
  2. Environment variables (COMFY_ENV_*)
  3. Persistent ~/.comfy-env/settings.env (loaded at import with
     os.environ.setdefault -- it fills UNSET env vars, so it can never
     override an explicitly-set one)
  4. Defaults

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
_REMOVED_ENV_VARS = ("COMFY_ENV_ISOLATE", "COMFY_ENV_INSTALL_ISOLATED")


def _check_removed_env_vars():
    import sys as _sys
    for _var in _REMOVED_ENV_VARS:
        _val = os.environ.get(_var)
        if _val is None:
            continue
        if _val.strip().lower() in ("0", "false", "no", "off"):
            raise RuntimeError(
                f"[comfy-env] {_var}={_val} was removed in 0.4.25 and can no "
                f"longer disable anything: isolation is always on, and missing "
                f"envs fall back per-env automatically. Unset this variable "
                f"(check your shell profile, Dockerfile/CI, systemd units). "
                f"For containers, materialize envs at image build time -- see "
                f"the 'Containers, CI & air-gapped' page in the docs."
            )
        print(
            f"[comfy-env] WARNING: {_var} was removed in 0.4.25 and is "
            f"ignored (isolation is always on) -- unset it.",
            file=_sys.stderr, flush=True,
        )


_check_removed_env_vars()

# Load persistent settings (simple KEY=VALUE file) -- env vars always override
if SETTINGS_FILE.exists():
    try:
        _skipped_removed = False
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                if k in _REMOVED_ENV_VARS:
                    _skipped_removed = True  # TUI residue; next save drops it
                    continue
                os.environ.setdefault(k, v.strip())
        if _skipped_removed:
            import sys as _sys
            print(
                f"[comfy-env] Note: {SETTINGS_FILE} contains removed settings "
                f"(isolate/install_isolated); rerun `comfy-env settings` to "
                f"clean it up.",
                file=_sys.stderr, flush=True,
            )
    except RuntimeError:
        raise
    except Exception:
        pass


def _is_on(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes")


# General settings
GENERAL_SETTINGS = [
    ("COMFY_ENV_AUTO_INSTALL", "Auto-materialize missing envs on first node load (blocks startup)"),
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

GENERAL_DEFAULTS = {
    "COMFY_ENV_AUTO_INSTALL": False,   # OFF by default -- explicit opt-in (multi-min installs)
    "COMFY_ENV_POOL_IPC": False,
}

# Numeric settings
NUMERIC_SETTINGS = [
    ("COMFY_ENV_WORKER_VRAM_BUDGET", "Worker VRAM budget (GB, 0=auto)", 0, "GB"),
]


def get_numeric(var: str, default: float = 0) -> float:
    val = os.environ.get(var, "")
    if val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


WORKER_VRAM_BUDGET = get_numeric("COMFY_ENV_WORKER_VRAM_BUDGET", 0)

# Patches (monkey-patching ComfyUI internals). Empty since 0.4.21 -- the
# only entry, COMFY_ENV_PATCH_SHAREABLE_POOL (parent-side CUDA shareable
# pool), was removed: experimental, default-off, and the cause of an
# environment->isolation import cycle. Kept as empty containers so the CLI
# settings surface stays stable.
PATCH_SETTINGS = []
PATCH_DEFAULTS = {}

# Mapping from short TOML key names to env var names (for [settings] in comfy-env-root.toml)
SETTINGS_KEY_MAP = {
    "auto_install": "COMFY_ENV_AUTO_INSTALL",
    "pool_ipc": "COMFY_ENV_POOL_IPC",
    "worker_vram_budget": "COMFY_ENV_WORKER_VRAM_BUDGET",
}
_ENV_TO_SHORT = {v: k for k, v in SETTINGS_KEY_MAP.items()}


def resolve_bool(var: str, node_settings: dict = None, default: bool = False) -> bool:
    """Resolve a boolean setting with per-node override support."""
    if node_settings:
        short_key = _ENV_TO_SHORT.get(var)
        if short_key and short_key in node_settings:
            return bool(node_settings[short_key])
    return _is_on(var, default)


def resolve_numeric(var: str, node_settings: dict = None, default: float = 0) -> float:
    """Resolve a numeric setting with per-node override support."""
    if node_settings:
        short_key = _ENV_TO_SHORT.get(var)
        if short_key and short_key in node_settings:
            try:
                return float(node_settings[short_key])
            except (ValueError, TypeError):
                pass
    return get_numeric(var, default)
