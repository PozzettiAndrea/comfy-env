"""General settings for comfy-env.

Settings can be configured via:
  1. Environment variables (highest priority)
  2. Persistent settings in ~/.comfy-env/settings.env
  3. Defaults

Workers can't import this module (different venv), so they parse env vars directly.
"""

import os
from pathlib import Path

SETTINGS_FILE = Path.home() / ".comfy-env" / "settings.env"

# Load persistent settings (simple KEY=VALUE file) — env vars always override
if SETTINGS_FILE.exists():
    try:
        for line in SETTINGS_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

# General settings: (env_var, label)
GENERAL_SETTINGS = [
    ("COMFY_ENV_INSTALL_ISOLATED", "Install to isolated envs (pixi/venv)"),
    ("COMFY_ENV_INSTALL_MAIN", "Install to main Python env (pip)"),
    ("COMFY_ENV_ISOLATE", "Run nodes in subprocess workers"),
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

# Defaults (True = on when env var is unset)
GENERAL_DEFAULTS = {
    "COMFY_ENV_INSTALL_ISOLATED": True,
    "COMFY_ENV_INSTALL_MAIN": False,
    "COMFY_ENV_ISOLATE": True,
    "COMFY_ENV_POOL_IPC": False,
}


def _is_on(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "")
    if val == "":
        return default
    return val.lower() in ("1", "true", "yes")


INSTALL_ISOLATED = _is_on("COMFY_ENV_INSTALL_ISOLATED", GENERAL_DEFAULTS["COMFY_ENV_INSTALL_ISOLATED"])
INSTALL_MAIN = _is_on("COMFY_ENV_INSTALL_MAIN", GENERAL_DEFAULTS["COMFY_ENV_INSTALL_MAIN"])
ISOLATE = _is_on("COMFY_ENV_ISOLATE", GENERAL_DEFAULTS["COMFY_ENV_ISOLATE"])
