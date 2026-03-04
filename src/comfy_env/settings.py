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

# General settings: (env_var, label, default_on)
# default_on=True means the feature is ON by default (env var unset or "1")
GENERAL_SETTINGS = [
    ("USE_COMFY_ENV", "Enable process isolation"),
    ("COMFY_ENV_POOL_IPC", "Pool IPC (zero-copy GPU tensor transfer)"),
]

# Defaults for general settings (True = feature is on when env var is unset)
GENERAL_DEFAULTS = {
    "USE_COMFY_ENV": True,
    "COMFY_ENV_POOL_IPC": False,
}
